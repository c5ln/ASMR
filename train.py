import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import KEYS, get_dataloaders
from model import build_maxvit_s


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train MaxViT-S on MBP keystroke data')
    p.add_argument('--wav_dir',         default='MBPWavs')
    p.add_argument('--epochs',          type=int,   default=100)
    p.add_argument('--batch_size',      type=int,   default=16)
    p.add_argument('--lr',              type=float, default=5e-4)
    p.add_argument('--val_every',       type=int,   default=5)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--checkpoint_dir',  default='checkpoints')
    return p.parse_args()


# ── Train / Eval Helpers ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * specs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += specs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * specs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += specs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for specs, labels in loader:
        preds = model(specs.to(device)).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(log_path, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs, train_acc, val_acc = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_acc.append(float(row['train_acc']))
            val_acc.append(float(row['val_acc']) if row['val_acc'] else None)

    val_x = [e for e, v in zip(epochs, val_acc) if v is not None]
    val_y = [v for v in val_acc if v is not None]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_acc, label='train acc', alpha=0.7)
    plt.plot(val_x, val_y, label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MaxViT-S Training on MBP Keystrokes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Training curves saved → {out_path}")


def plot_confusion_matrix(cm, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(KEYS)))
    ax.set_yticks(range(len(KEYS)))
    ax.set_xticklabels(KEYS, fontsize=8)
    ax.set_yticklabels(KEYS, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (test set)')

    for i in range(len(KEYS)):
        for j in range(len(KEYS)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Confusion matrix saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        args.wav_dir, batch_size=args.batch_size, seed=args.seed
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_maxvit_s(num_classes=n_classes, in_chans=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MaxViT-S  params : {n_params / 1e6:.1f}M")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Linear decay: lr goes from 5e-4 → 5e-6 over all epochs (×0.01 factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=args.epochs,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    log_path = os.path.join(args.checkpoint_dir, 'training_log.csv')

    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_acc']
                       ).writeheader()

    print(f"\nTraining for {args.epochs} epochs  (validate every {args.val_every})\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_acc = 0.0
        if epoch % args.val_every == 0 or epoch == 1:
            _, val_acc = evaluate(model, val_loader, criterion, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoint_dir, 'best.pt'),
                )

            print(
                f"[{epoch:5d}/{args.epochs}]  "
                f"loss={train_loss:.4f}  train={train_acc:.4f}  "
                f"val={val_acc:.4f}  best={best_val_acc:.4f}  "
                f"lr={current_lr:.2e}"
            )

        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_acc']
                           ).writerow({
                               'epoch':      epoch,
                               'train_loss': round(train_loss, 6),
                               'train_acc':  round(train_acc, 6),
                               'val_acc':    round(val_acc, 6),
                           })

    # ── Step 7: Final Evaluation ──────────────────────────────────────────────
    print(f"\nBest validation accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("Loading best checkpoint for test evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint_dir, 'best.pt'), map_location=device)
    )

    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy            : {test_acc:.4f} ({test_acc*100:.2f}%)\n")

    preds, labels = collect_predictions(model, test_loader, device)
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=KEYS, digits=4))

    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(args.checkpoint_dir, 'confusion_matrix.npy'), cm)

    # Plots (require matplotlib; skip if unavailable)
    try:
        plot_training_curves(
            log_path,
            os.path.join(args.checkpoint_dir, 'training_curves.png'),
        )
        plot_confusion_matrix(
            cm,
            os.path.join(args.checkpoint_dir, 'confusion_matrix.png'),
        )
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == '__main__':
    main()
