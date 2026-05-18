# Keyboard Acoustic Side Channel Attack with CAFormer-S36

Replication of **"A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards"** (Harrison et al., 2023) with **CoAtNet replaced by CAFormer-S36** (Yu et al., 2022).

Given a microphone-recorded audio of MacBook Pro keystrokes, the model classifies which of **36 keys (0–9, A–Z)** was pressed — with no line-of-sight access to the keyboard.

---

## Original Paper vs. This Implementation

| | Harrison et al. (2023) | This Repo |
|---|---|---|
| Model | CoAtNet | **CAFormer-S36** |
| Pretrained | No | ImageNet-22k → ImageNet-1k |
| Input | 2-ch Mel-spectrogram 64×64 | **3-ch** (L / R / mono) 128×128 |
| Classes | 36 (0–9, A–Z) | Same |
| Dataset | MacBook Pro (phone-recorded) | Same (`MBPWavs/`) |
| Augmentation | Time shift + SpecAugment | Same |

---

## Pipeline

```
WAV files (stereo, 44100 Hz)
  → Keystroke Isolation   (adaptive energy threshold, 25 strokes/key)
  → Time Shift            (±40%, train only)
  → 3-Channel Mel-Spec    L / R / mono  →  (3, 128, 128)  dB scale
  → Min-Max Normalize     per channel → [0, 1]
  → SpecAugment           (train only)
  → CAFormer-S36 backbone + AdaptiveAvgPool2d(2×2) + Linear(2048, 36)
```

---

## Model Architecture

CAFormer (MetaFormer variant) replaces the token-mixer with **pooling** in early stages and **self-attention** in later stages, providing a strong pretrained backbone without the fixed-resolution constraint of window-attention models.

```
Input  (B, 3, 128, 128)
  └── CAFormer-S36 backbone  [pretrained: ImageNet-22k → 1k]
        └── forward_features()  →  (B, 512, 4, 4)
  └── AdaptiveAvgPool2d(2, 2)   →  (B, 512, 2, 2)
  └── Flatten                   →  (B, 2048)
  └── Linear(2048, 36)          →  (B, 36)
```

---

## Dataset

`MBPWavs/` contains 36 stereo `.wav` files (44100 Hz), one per key.
Each file holds 25 keystrokes recorded via phone microphone next to a 16-inch M1 Pro MacBook Pro.

```
MBPWavs/
├── 0.wav  ← 25 presses of key '0'
├── 1.wav
├── ...
├── A.wav
├── ...
└── Z.wav
```

`Zoom/` contains the same 36 keys recorded via Zoom's built-in meeting recorder.

Total raw samples: **900** (36 keys × 25 strokes).
With offline augmentation (`store_data.py`): up to ~27,000 samples.

---

## Preprocessing

### 1. Keystroke Isolation

Energy-based peak detection with an adaptive threshold loop (same algorithm as the original paper):

```python
prom, step = 0.06, 0.005
while len(strokes) != 25:
    strokes = detect(signal[1s:], threshold=prom, before=2400, after=12000)
    if len(strokes) < 25: prom -= step
    else:                  prom += step
    step *= 0.99
```

Each isolated keystroke: `(2, 14400)` ≈ 0.33 s @ 44100 Hz

### 2. Mel-Spectrogram

```python
# per channel (L, R, mono)
librosa.feature.melspectrogram(y, sr=44100, n_mels=128, n_fft=1024, hop_length=112)
# → power_to_db → crop/pad → (128, 128)
# stack 3 channels → (3, 128, 128)
# min-max normalize per channel → [0, 1]
```

### 3. Augmentation (train only)

- **Time shift**: random circular roll ±40% of keystroke length
- **SpecAugment**: 2 frequency masks + 2 time masks, each up to 10% of axis length

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. CUDA or Apple MPS recommended.

---

## Usage

### Step 1 — Pre-generate Augmented Dataset

Offline augmentation writes a single `.pt` file to avoid re-computing spectrograms every epoch:

```bash
python store_data.py
# → augment_data/dataset_128.pt
```

Output format:

```python
data = torch.load('augment_data/dataset_128.pt')
data['specs']   # Tensor  (N, 3, 128, 128)  float32
data['labels']  # Tensor  (N,)               int64
```

| `--n_augments` | Samples | Approx. size |
|---|---|---|
| 10 (default) | 9,000 | ~1.4 GB |
| 20 | 18,000 | ~2.8 GB |
| 30 | 27,000 | ~4.2 GB |

---

### Step 2 — Train

```bash
python train.py --pt_path ./augment_data/dataset_128.pt
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--pt_path` | `./dataset_128.pt` | Pre-generated `.pt` dataset |
| `--save_dir` | `./final_train_results` | Output directory |
| `--n_epochs` | `100` | Total epochs |
| `--eval_interval` | `10` | Validation frequency (epochs) |
| `--save_interval` | `50` | Checkpoint save frequency (epochs) |
| `--patience` | `30` | Early stopping patience (epochs) |
| `--resume_best` | `None` | Resume from `best_model.pth` with lower LR |
| `--resume` | `None` | Resume from full checkpoint (use sparingly) |

Best hyperparameters (baked in):

| Parameter | Value |
|---|---|
| LR | 4.4e-5 |
| Weight Decay | 0.0027 |
| Warmup Epochs | 3 |
| Label Smoothing | 0.018 |
| Grad Clip | 2.0 |
| Batch Size | 64 |

Outputs:

```
final_train_results/
├── best_model.pth          # Best weights by val accuracy
├── train_curve.png         # Loss / val accuracy / LR plot
├── train_log.csv           # Per-epoch log
└── checkpoints/
    └── checkpoint_ep50.pth # Full checkpoint (weights + optimizer + history)
```

Fine-tuning from a saved best model:

```bash
python train.py \
  --pt_path ./augment_data/dataset_128.pt \
  --resume_best ./final_train_results/best_model.pth \
  --n_epochs 50
```

> **Note:** When using `--resume_best`, update `best_acc` in `train.py:226` to match the current best before running.

---

### Step 3 — Inference

Run on a single audio file (short isolated clip or a longer multi-keystroke recording):

```bash
python infer.py --audio ./MBPWavs/A.wav --model ./final_train_results/best_model.pth
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--audio` | *(required)* | Input audio file (`.wav`, `.mp3`, ...) |
| `--model` | `./final_train_results/best_model.pth` | Trained weights |
| `--top_k` | `3` | Number of top candidates to show |

Example output:

```
[Device] cuda
[*] 모델 로드 완료.

[*] 오디오 길이: 14.93s
[*] 감지된 키스트로크: 25개

──────────────────────────────
[keystroke 1]
  ★ 1위: A  ( 91.3%)
    2위: S  (  4.8%)
    3위: Q  (  1.7%)

[keystroke 2]
  ★ 1위: A  ( 88.6%)
  ...
```

`infer.py` is fully standalone — it does not import `dataset.py` or `train.py`.

---

## File Structure

```
.
├── MBPWavs/                    # Phone-recorded keystrokes (36 × .wav)
├── Zoom/                       # Zoom-recorded keystrokes (36 × .wav)
├── augment_data/
│   └── dataset_128.pt          # Pre-generated augmented dataset
├── dataset.py                  # Keystroke isolation, mel-spec, Dataset/DataLoader
├── store_data.py               # Offline augmentation → augment_data/*.pt
├── train.py                    # Training loop, checkpointing, early stopping
├── infer.py                    # Standalone inference on single audio files
├── requirements.txt
└── final_train_results/        # Created during training
    ├── best_model.pth
    ├── train_curve.png
    ├── train_log.csv
    └── checkpoints/
```

---

## References

- Harrison, J., Toreini, E., & Mehrnezhad, M. (2023). *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*. IEEE EuroS&P.
- Yu, W., Si, C., Zhou, P., Luo, M., Zhou, Y., Feng, J., Yan, S., & Pan, X. (2022). *MetaFormer Baselines for Vision*. TPAMI 2024.
- Original dataset: [github.com/gaborvecsei/keystroke-acoustic-emanation](https://github.com/Rohul1997/dl-keyboard-sounds) (Harrison et al.)
