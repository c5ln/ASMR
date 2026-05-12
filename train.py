import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import KEYS, get_dataloaders
from model import build_model

# ── QWERTY 2D 좌표 (row, col) — 행 stagger 반영 ─────────────────────────────
# 숫자행: 0~9 (offset 0)
# Q행   : Q~P (offset 0)
# A행   : A~L (offset 0.5)
# Z행   : Z~M (offset 1.0)
_KEY_POS = {
    '1':(0,0),'2':(0,1),'3':(0,2),'4':(0,3),'5':(0,4),
    '6':(0,5),'7':(0,6),'8':(0,7),'9':(0,8),'0':(0,9),
    'Q':(1,0),'W':(1,1),'E':(1,2),'R':(1,3),'T':(1,4),
    'Y':(1,5),'U':(1,6),'I':(1,7),'O':(1,8),'P':(1,9),
    'A':(2,0.5),'S':(2,1.5),'D':(2,2.5),'F':(2,3.5),'G':(2,4.5),
    'H':(2,5.5),'J':(2,6.5),'K':(2,7.5),'L':(2,8.5),
    'Z':(3,1.0),'X':(3,2.0),'C':(3,3.0),'V':(3,4.0),'B':(3,5.0),
    'N':(3,6.0),'M':(3,7.0),
}


def _build_soft_label_matrix(keys: list, sigma: float) -> torch.Tensor:
    """keys 순서대로 (n, n) Gaussian soft-label 행렬 반환."""
    pos = np.array([_KEY_POS[k] for k in keys], dtype=np.float32)   # (n, 2)
    diff = pos[:, None, :] - pos[None, :, :]                         # (n, n, 2)
    d2   = (diff ** 2).sum(-1)                                        # (n, n)
    w    = np.exp(-d2 / (2 * sigma ** 2))
    w   /= w.sum(axis=1, keepdims=True)
    return torch.tensor(w, dtype=torch.float32)


class GaussianLabelSmoothingLoss(nn.Module):
    """
    정답 키를 중심으로 Gaussian 분포를 갖는 soft label로 KL divergence 계산.
    인접 키로 오분류할수록 적은 penalty, 먼 키로 오분류할수록 큰 penalty.
    """
    def __init__(self, keys: list, sigma: float = 1.0):
        super().__init__()
        mat = _build_soft_label_matrix(keys, sigma)
        self.register_buffer('label_dist', mat)   # (n_classes, n_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft     = self.label_dist[targets]               # (B, n_classes)
        log_prob = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_prob, soft, reduction='batchmean')


# ── Metrics ──────────────────────────────────────────────────────────────────

def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / B).item()
                for k in topk]


# ── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for specs, labels in tqdm(loader, desc='  train', leave=False):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = criterion(model(specs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = top1_sum = top5_sum = n = 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(specs)
        total_loss += criterion(out, labels).item()
        t1, t5 = topk_accuracy(out, labels)
        top1_sum += t1 * labels.size(0)
        top5_sum += t5 * labels.size(0)
        n += labels.size(0)
    return total_loss / len(loader), top1_sum / n, top5_sum / n


# ── LR schedule: linear warmup + cosine decay ────────────────────────────────

def _lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device}  |  AMP: {use_amp}')

    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        args.wav_dir, batch_size=args.batch_size
    )

    model = build_model(num_classes=n_classes).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    criterion = GaussianLabelSmoothingLoss(KEYS, sigma=args.sigma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: _lr_lambda(e, args.warmup, args.epochs)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'lr={lr_now:.2e} | '
            f'train={train_loss:.4f} | '
            f'val={val_loss:.4f} | '
            f'top1={val_top1:.1f}% | '
            f'top5={val_top5:.1f}%'
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(), 'val_top1': val_top1},
                os.path.join(args.ckpt_dir, 'best_model.pt')
            )

    # ── 최종 테스트 평가 ──────────────────────────────────────────────────────
    ckpt = torch.load(
        os.path.join(args.ckpt_dir, 'best_model.pt'), map_location=device
    )
    model.load_state_dict(ckpt['model'])
    _, test_top1, test_top5 = evaluate(model, test_loader, criterion, device, use_amp)
    print(f'\n[Test] top1={test_top1:.1f}%  top5={test_top5:.1f}%  '
          f'(best val epoch={ckpt["epoch"]})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='ConvNextV2 Acoustic Side-Channel Attack')
    p.add_argument('--wav_dir',    default='MBPWavs',      help='wav 파일 디렉터리')
    p.add_argument('--ckpt_dir',   default='checkpoints',  help='체크포인트 저장 경로')
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--warmup',     type=int,   default=20,  help='Linear warmup 에폭 수')
    p.add_argument('--sigma',      type=float, default=1.0, help='Gaussian smoothing σ')
    main(p.parse_args())
