"""
CAFormer-S36 Best Hyperparameter Training
JH. HA
"""

import argparse, json, csv, os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm

# setting
MODEL_NAME        = "caformer_s36.sail_in22k_ft_in1k"
N_CLASSES         = 36
BACKBONE_FEAT_DIM = 512
POOL_OUT          = (2, 2)                                             # 추가된 layer 1, CAFormer의 아웃풋은 4,4,512로 pooling을 통해 2,2,512로 변경
FC_IN             = BACKBONE_FEAT_DIM * POOL_OUT[0] * POOL_OUT[1]      # 추가된 layer 2, 위에서 나온 2,2,512를 36개의 classificaiton을 위해 36,2048로 flatten 진행

BEST_PARAMS = {
    "lr":              4.4e-05,    # 연속값을 반올림 - 원본 숫자는 hyper parameter csv 파일 참조
    "weight_decay":    0.0027,
    "warmup_epochs":   3,
    "label_smoothing": 0.018,
    "grad_clip":       2.0,
    "batch_size":      64,
}

FINETUNE_LR     = 5e-6   # revised LR
FINETUNE_ETA_MIN = 1e-8  # min cos LR
# ─────────────────────────────────────────────────────────────────────────────
# Data

def load_dataset(pt_path: str):
    data      = torch.load(pt_path, map_location="cpu", weights_only=False)
    specs     = data["specs"].float()
    labels    = data["labels"].long()
    n_classes = int(labels.max().item()) + 1
    print(f"[Dataset] specs={tuple(specs.shape)}  labels={tuple(labels.shape)}  classes={n_classes}")
    return specs, labels, n_classes


class MelDataset(Dataset):
    def __init__(self, specs: torch.Tensor, labels: torch.Tensor):
        self.specs  = specs
        self.labels = labels
    def __len__(self): return len(self.specs)
    def __getitem__(self, idx): return self.specs[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model

class CAFormerClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)      # 기존 CAFormer
        self.pool2d   = nn.AdaptiveAvgPool2d(POOL_OUT)          # 추가 layer 1
        self.fc       = nn.Linear(FC_IN, n_classes)             # 추가 layer 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # forward (process)
        feat = self.backbone.forward_features(x)
        feat = self.pool2d(feat)
        feat = feat.flatten(1)
        return self.fc(feat)

# ─────────────────────────────────────────────────────────────────────────────
# Options

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")          # expected (Apple GPU)
    elif torch.cuda.is_available():
        return torch.device("cuda")         # can not used on MacOS
    return torch.device("cpu")              # log에 이거 뜨면 환경 잘못 된 것이니 다시 설정


def build_amp_context(device: torch.device):
    if device.type == "cuda":
        scaler   = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        scaler   = None
        autocast = nullcontext              # 현재 맥 버전이 autocast를 불러오지 못하는 중(버전 충돌이 계속 발생) -- 해결이 된다면 적용 예정
    return scaler, autocast


def _clear_cache(device: torch.device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

# 결과 csv 파일에 저장 및 plot 생성
def save_results(save_dir, history):
    if not history:
        return

    epochs       = [h["epoch"]      for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_accs     = [h["val_acc"]    for h in history if h["val_acc"] is not None]
    val_epochs   = [h["epoch"]      for h in history if h["val_acc"] is not None]
    lrs          = [h["lr"]         for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.plot(epochs, train_losses, color="tab:blue", label="Train Loss")                 # epoch에 따른 train loss - 파란색
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("LR", color="tab:green")                                              # epoch에 따른 LR - 초록색
    ax3.plot(epochs, lrs, color="tab:green", linewidth=0.8, alpha=0.6, label="LR")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    if val_accs:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Accuracy", color="tab:red")                            # epoch에 따른 validation - 빨간색
        ax2.plot(val_epochs, val_accs, "r-o", linewidth=2, label="Val Acc")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(0, 1.0)

    fig.tight_layout()
    plt.title("Training Progress")
    fig.savefig(save_dir / "train_curve.png", dpi=120)                                    # 결과 plot 저장
    plt.close(fig)

    with open(save_dir / "train_log.csv", "w", newline="") as f:                          # 결과 csv 저장
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc", "lr"])
        for h in history:
            writer.writerow([h["epoch"], h["train_loss"], h.get("val_acc", ""), h.get("lr", "")])


# ────────────────────────────────────────────────────────────────────────
# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path",       default="./dataset_128.pt")
    parser.add_argument("--save_dir",      default="./final_train_results")
    parser.add_argument("--n_epochs",      type=int, default=100)                       # 진행 상황을 보고 100, 150, 200 250 ~ 순으로 적당히 조정
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--patience",      type=int, default=30)                         # 전체 epoch에 맞춰서 같이 조절
    parser.add_argument("--resume",        type=str, default=None,
                        help="풀 체크포인트 파일 경로 (epoch/optimizer 포함, 이어서 학습 시)") # 사용 지양 - 오래 걸림 (최대한 best에서 로드하도록)
    parser.add_argument("--resume_best",   type=str, default=None,
                        help="best_model.pth 경로: 가중치만 로드 후 낮은 LR로 재시작")        # check point file : .pth 경로 기입
    parser.add_argument("--mps_fallback",  action="store_true")
    args = parser.parse_args()

    if args.mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("[경고] MPS fallback 활성화")

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[Device] {device}  (torch {torch.__version__})")

    # load data
    specs, labels, n_classes = load_dataset(args.pt_path)
    indices = np.arange(len(specs))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.1, stratify=labels.numpy(), random_state=42          # split은 9:1로 지정
    )

    train_ds = MelDataset(specs[torch.from_numpy(train_idx)], labels[torch.from_numpy(train_idx)])
    test_ds  = MelDataset(specs[torch.from_numpy(test_idx)],  labels[torch.from_numpy(test_idx)])

    pin = device.type in ["cuda", "mps"]
    nw  = 6                                    # search 단계에서 4로 했다가 문제가 컸던 이유로 6으로 조정

    train_loader = DataLoader(
        train_ds, batch_size=BEST_PARAMS["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True, persistent_workers=(nw > 0),
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BEST_PARAMS["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0),
        prefetch_factor=4,
    )

    # model & loss
    model     = CAFormerClassifier(n_classes=n_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=BEST_PARAMS["label_smoothing"])
    scaler, autocast_fn = build_amp_context(device)

    start_epoch      = 0
    best_acc         = 0.0
    epochs_no_improve = 0
    history          = []

    # resume_best
    if args.resume_best:
        if not os.path.isfile(args.resume_best):
            print(f"[오류] 파일 없음: {args.resume_best}")
            return

        print(f"[*] best_model.pth 로드 중... ({args.resume_best})")
        model.load_state_dict(torch.load(args.resume_best, map_location=device))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FINETUNE_LR,
            weight_decay=BEST_PARAMS["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.n_epochs,
            eta_min=FINETUNE_ETA_MIN,
        )
        best_acc = 0.6255                                                               # 주의!!! - 롤백시마다 최고 임계점 갱신 필요 - !!!
        print(f"[*] best_model.pth 로드 완료. LR={FINETUNE_LR:.1e}로 재시작합니다.")
        print(f"    (기준 best_acc={best_acc:.4f}, 이걸 넘어야 best_model.pth 갱신)")

    # 사용 지양
    elif args.resume:
        if not os.path.isfile(args.resume):
            print(f"[오류] 파일 없음: {args.resume}")
            return

        print(f"[*] 체크포인트 로드 중... ({args.resume})")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BEST_PARAMS["lr"],
            weight_decay=BEST_PARAMS["weight_decay"],
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch       = checkpoint["epoch"]
        best_acc          = checkpoint.get("best_acc", 0.0)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        history           = checkpoint.get("history", [])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.n_epochs,
            eta_min=FINETUNE_ETA_MIN,
        )
        for _ in range(start_epoch):
            scheduler.step()

        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"[*] Epoch {start_epoch}부터 재시작. (best_acc={best_acc:.4f})")

    # fill-training
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BEST_PARAMS["lr"],
            weight_decay=BEST_PARAMS["weight_decay"],
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=BEST_PARAMS["warmup_epochs"],
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.n_epochs - BEST_PARAMS["warmup_epochs"],
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[BEST_PARAMS["warmup_epochs"]],
        )
        print(f"[*] 처음부터 학습 시작. LR={BEST_PARAMS['lr']:.1e}")

    print(f"\n[Train Start] n_epochs={args.n_epochs}, best_acc 기준={best_acc:.4f}")

    epoch_bar = tqdm(
        range(start_epoch, args.n_epochs),
        desc="Training", unit="ep",
        initial=start_epoch, total=args.n_epochs,
    )

    for epoch in epoch_bar:
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_fn():
                output = model(x)
                loss   = criterion(output, y)

            if scaler:
                scaler.scale(loss).backward()
                if BEST_PARAMS["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), BEST_PARAMS["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if BEST_PARAMS["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), BEST_PARAMS["grad_clip"])
                optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        scheduler.step()

        val_acc = None
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with autocast_fn():
                        pred = model(x).argmax(1)
                    correct += (pred == y).sum().item()
                    total   += y.size(0)
            val_acc = correct / total

            if val_acc > best_acc:
                best_acc          = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                tqdm.write(f"\n[★ Best] Epoch {epoch+1}: val_acc={val_acc:.4f} → best_model.pth 갱신")
            else:
                epochs_no_improve += args.eval_interval

            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
                lr=f"{current_lr:.2e}",
                patience=f"{epochs_no_improve}/{args.patience}",
            )
        else:
            epoch_bar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{current_lr:.2e}")

        history.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        })

        # checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"checkpoint_ep{epoch+1}.pth"
            ckpt_data = {
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc":             best_acc,
                "epochs_no_improve":    epochs_no_improve,
                "history":              history,
            }
            if scaler:
                ckpt_data["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)
            tqdm.write(f"\n[Checkpoint] Saved → {ckpt_path}")

        # early stopping
        if epochs_no_improve >= args.patience:
            tqdm.write(f"\n[Early Stopping] {args.patience} epochs 동안 개선 없음. Epoch {epoch+1}에서 종료.")
            break

    save_results(save_dir, history)
    print(f"\n[완료] 최고 정확도: {best_acc:.4f}")
    print(f"[저장] {save_dir}")
    _clear_cache(device)


if __name__ == "__main__":
    main()
