# Keyboard Acoustic Side Channel Attack with MaxViT

Replication of **"A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards"** (Harrison et al., 2023) with **CoAtNet replaced by MaxViT-S** (Tu et al., 2022).

Given a phone-recorded audio of MacBook Pro keystrokes, the model classifies which of 36 keys (0–9, A–Z) was pressed — achieving this with no line-of-sight access to the keyboard.

---

## Overview

### Original Paper vs. This Implementation

| | Harrison et al. (2023) | This Repo |
|---|---|---|
| Model | CoAtNet | **MaxViT-S** |
| Input | 2-ch Mel-spectrogram 64×64 | Same, resized to 224×224 |
| Classes | 36 (0–9, A–Z) | Same |
| Dataset | MacBook Pro (phone-recorded) | Same (`MBPWavs/`) |
| Augmentation | Time shift + SpecAugment | Same |

### Pipeline

```
WAV files
  → Keystroke Isolation (adaptive threshold)
  → Time Shift Augmentation (±40%)
  → 2-Channel Mel-Spectrogram (64×64)
  → SpecAugment (frequency + time masking)
  → Resize 224×224
  → MaxViT-S (36 classes)
```

---

## MaxViT-S Architecture

MaxViT (Multi-Axis Vision Transformer) interleaves MBConv blocks with **local Block-Attention** and **global Grid-Attention** at each stage, allowing both local and global feature extraction without quadratic cost.

| Stage | Output | Channels | Heads | Blocks |
|-------|--------|----------|-------|--------|
| Stem  | 112×112 | 64  | —  | Conv ×2 |
| S1    | 56×56   | 96  | 3  | ×2 |
| S2    | 28×28   | 192 | 6  | ×2 |
| S3    | 14×14   | 384 | 12 | ×5 |
| S4    | 7×7     | 768 | 24 | ×2 |
| Head  | —       | —   | —  | GlobalAvgPool → FC(36) |

- Partition size P = Grid size G = 7 → requires 224×224 input
- ~69M trainable parameters
- Trained from scratch (no pretrained weights)

---

## Dataset

`MBPWavs/` contains 36 stereo `.wav` files (44100 Hz), one per key.  
Each file holds 25 keystrokes recorded via phone microphone.

```
MBPWavs/
├── 0.wav  ← 25 presses of key '0'
├── 1.wav
├── ...
├── A.wav
├── ...
└── Z.wav
```

Total: **900 samples** — split 80 / 10 / 10 (train / val / test).

---

## Preprocessing

### 1. Keystroke Isolation

Energy-based peak detection with an adaptive threshold loop:

```python
# Algorithm 1 from the paper
prom = 0.06
while len(strokes) != 25:
    strokes = isolator(signal, threshold=prom, before=2400, after=12000)
    if len(strokes) < 25: prom -= step
    else:                  prom += step
```

Each isolated keystroke: `(2 channels, 14400 samples)` ≈ 0.33 s

### 2. Mel-Spectrogram

```python
librosa.feature.melspectrogram(y, sr=44100, n_mels=64, n_fft=1024, hop_length=225)
# → (64, 64) per channel → stack → (2, 64, 64) → resize → (2, 224, 224)
```

### 3. Augmentation (train only)

- **Time shift**: random roll by ±40% of keystroke length
- **SpecAugment**: 2 frequency masks + 2 time masks, each up to 10% of axis width

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. GPU recommended.

---

## Usage

### Pre-generate Augmented Dataset (optional)

`store_data.py` pre-generates augmented mel-spectrograms and saves them to disk as a single `.pt` file containing both spectrograms and ground-truth labels.

```bash
python store_data.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--wav_dir` | `MBPWavs` | Path to keystroke WAV files |
| `--n_augments` | `10` | Augmented versions per keystroke |
| `--save_dir` | `augment_data` | Output directory |
| `--seed` | `42` | Random seed for reproducibility |

Output file: `augment_data/dataset.pt`

```python
data = torch.load('augment_data/dataset.pt')
data['specs']   # torch.Tensor  shape (N, 3, 224, 224)  — L / R / mono-mix channels
data['labels']  # torch.Tensor  shape (N,)  dtype=int64  — class index per sample
```

> **Note:** `store_data.py` produces **3-channel** spectrograms (left, right, mono mix), whereas the main training pipeline (`train.py`) uses **2-channel** (left, right only). If you load `dataset.pt` for training, adjust `in_chans` accordingly.

Estimated file sizes for common `--n_augments` values (36 keys × 25 × N, float32):

| `--n_augments` | Samples | File size |
|---|---|---|
| 10 | 9,000 | ~5 GB |
| 20 | 18,000 | ~10 GB |
| 30 | 27,000 | ~15 GB |

---

### Train

```bash
python train.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--wav_dir` | `MBPWavs` | Path to keystroke WAV files |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `5e-4` | Initial learning rate (linear decay to ×0.01) |
| `--val_every` | `5` | Validation interval (epochs) |
| `--checkpoint_dir` | `checkpoints` | Output directory |

Example — replicate paper settings (1100 epochs):

```bash
python train.py --epochs 1100 --lr 5e-4
```

### Outputs

```
checkpoints/
├── best.pt                 # Best checkpoint (by val accuracy)
├── training_log.csv        # Per-epoch loss and accuracy
├── training_curves.png     # Loss / accuracy plot
└── confusion_matrix.png    # 36×36 confusion matrix (test set)
```

### Verify model

```bash
python model.py
# MaxViT-S  trainable params: 69.0M
# Input:  torch.Size([2, 2, 224, 224])  →  Output: torch.Size([2, 36])
```

---

## File Structure

```
.
├── MBPWavs/                # Raw keystroke recordings (36 × .wav)
├── dataset.py              # Keystroke isolation, mel-spectrogram, Dataset/DataLoader
├── model.py                # MaxViT-S wrapper (timm)
├── train.py                # Training loop, evaluation, plots
├── store_data.py           # Pre-generate augmented spectrograms → augment_data/dataset.pt
├── augment_data/
│   └── dataset.pt          # Pre-generated dataset: {'specs': (N,3,224,224), 'labels': (N,)}
├── requirements.txt
└── checkpoints/            # Auto-created during training
```

---

## References

- Harrison, J., Toreini, E., & Mehrnezhad, M. (2023). *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*. IEEE EuroS&P.
- Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., & Li, Y. (2022). *MaxViT: Multi-Axis Vision Transformer*. ECCV 2022.
