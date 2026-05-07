# Implementation Plan: Keyboard Acoustic Side Channel Attack with MaxViT-S

## Overview

Replicate "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (Harrison et al., 2023) using MBPWavs dataset, replacing CoAtNet with MaxViT-S for improved performance.

---

## Pipeline Summary

```
WAV files → Keystroke Isolation → Time Shift Augmentation
→ 2-Channel Audio → Mel-Spectrogram (64×64) → Resize 224×224
→ SpecAugment → MaxViT-S (num_classes=36) → Train/Eval
```

---

## Step 1: Data Loading

**File:** `train.py`

- Load all 36 `.wav` files from `MBPWavs/` (keys: 0-9, A-Z)
- Use `librosa.load(path, sr=None, mono=False)` to preserve stereo (2 channels, 44100 Hz)
- Label each file by its filename (e.g., `A.wav` → label `A`)
- Map labels to integers: sort keys, assign 0–35

---

## Step 2: Keystroke Isolation

**Reference:** `README.md` `isolator()` function

Parameters for MBP phone-recorded data:
```python
isolator(
    signal=samples_ch[:, 1*sample_rate:],  # skip first 1 sec
    sample_rate=44100,
    size=48,        # FFT size for energy detection
    scan=24,        # hop for energy
    before=2400,    # samples before peak
    after=12000,    # samples after peak → total 14400 samples (0.33s)
    threshold=prom,
    show=False
)
```

Adaptive threshold loop (Algorithm 1 in paper):
```python
prom = 0.06
step = 0.005
while len(strokes) != 25:
    strokes = isolator(...)
    if len(strokes) < 25: prom -= step
    if len(strokes) > 25: prom += step
    if prom <= 0: break
    step *= 0.99
```

Apply independently to each stereo channel.

**Output:** Per key: 25 keystrokes, each shape `(2, 14400)` (2 channels × 14400 samples)

---

## Step 3: Data Augmentation & Preprocessing

Following the paper's pipeline (Figure 3 and Table 3):

### 3a. Time Shift Augmentation (applied at training time)
- Randomly shift signal by up to ±40% of keystroke length
- `shift = random.randint(-0.4 * 14400, 0.4 * 14400)`
- Use `np.roll()` or slice-and-pad

### 3b. Mel-Spectrogram Generation
For each of the 2 channels:
```python
librosa.feature.melspectrogram(
    y=signal,
    sr=44100,
    n_mels=64,
    n_fft=1024,
    hop_length=225
)
# Convert to dB scale: librosa.power_to_db(S, ref=np.max)
```
- Output per channel: ~(64, 64) (crop/pad time axis to exactly 64)
- Stack channels → shape `(2, 64, 64)`

### 3c. Normalization
- Normalize each spectrogram to [0, 1] or z-score normalize (paper: "Normalised Data: Yes")

### 3d. Resize for MaxViT-S
- MaxViT-S uses P=G=7 (block/grid partition size 7×7)
- Minimum input for 4 downsampling stages with P=7: 224×224
- Resize: `torchvision.transforms.Resize((224, 224))`
- Shape: `(2, 224, 224)`

### 3e. SpecAugment Masking (applied at training time only)
- 2 masks per axis (time and frequency)
- Each mask: random width up to 10% of axis → up to 6 pixels on 64-dim axis
- Set masked region to mean value
- Apply on the mel-spectrogram (before resize)

---

## Step 4: Dataset Preparation

Total samples: 36 keys × 25 keystrokes = **900 samples**

Random split (paper: "Data Split: Random"):
- Train: 80% → 720 samples
- Validation: 10% → 90 samples
- Test: 10% → 90 samples

Use `torch.utils.data.Dataset` + `DataLoader`:
```python
DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## Step 5: MaxViT-S Model

Use `timm` library for MaxViT-S implementation:
```python
import timm
model = timm.create_model(
    'maxvit_small_tf_224',
    pretrained=False,     # train from scratch
    num_classes=36,       # 36 keys
    in_chans=2            # 2-channel mel-spectrogram
)
```

**MaxViT-S architecture** (from Table 11, MaxViT paper):
| Stage | Output Size | Config |
|-------|-------------|--------|
| Stem  | 112×112     | 3×3 Conv ×2, C=64 |
| S1    | 56×56       | MBConv C=96, Rel-MSA P=7 H=3, Grid-SA G=7 H=3, ×2 |
| S2    | 28×28       | MBConv C=192, Rel-MSA P=7 H=6, Grid-SA G=7 H=6, ×2 |
| S3    | 14×14       | MBConv C=384, Rel-MSA P=7 H=12, Grid-SA G=7 H=12, ×5 |
| S4    | 7×7         | MBConv C=768, Rel-MSA P=7 H=24, Grid-SA G=7 H=24, ×2 |
| Head  | 1×1         | Global Avg Pool → FC(36) |

Total: ~69M parameters

---

## Step 6: Training

Hyperparameters from Table 3 (MacBook phone classifier):

| Parameter | Value |
|-----------|-------|
| Epochs | 1100 |
| Batch Size | 16 |
| Loss | Cross Entropy |
| Optimizer | Adam |
| Max Learning Rate | 5e-4 |
| LR Schedule | Linear decay |
| Data Split | Random |

Training loop:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.0, total_iters=1100
)
criterion = nn.CrossEntropyLoss()

for epoch in range(1100):
    # train
    model.train()
    for batch in train_loader:
        ...
    scheduler.step()

    # validate every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        ...
```

---

## Step 7: Evaluation

- Log train loss, train accuracy, val accuracy per epoch
- Save best model checkpoint (by validation accuracy)
- Final test set evaluation:
  - Top-1 accuracy
  - Confusion matrix (36×36)
  - Classification report (precision, recall, F1 per key)

---

## File Structure

```
keyboard-acoustic-side-channel-attack-coatnet/
├── MBPWavs/              # Raw .wav files
├── train.py              # Main training script
├── dataset.py            # Dataset class + preprocessing
├── model.py              # MaxViT-S wrapper
├── requirements.txt
├── PLAN.md
└── checkpoints/          # Saved model weights (auto-created)
```

---

## Notes & Considerations

1. **Small dataset (900 samples):** Time shift and SpecAugment augmentation are critical to prevent overfitting on 720 training samples.
2. **MaxViT-S input size:** The original MaxViT-S was designed for 224×224. We resize 64×64 spectrograms to 224×224. Alternatively, use `timm.create_model('maxvit_small_tf_64', ...)` with custom partition size P=G=4 if timm supports it.
3. **2-channel input:** `timm` automatically adapts the first conv layer when `in_chans=2`.
4. **Training instability:** The paper found models could "collapse" around epoch 300–400 at LR=1e-3. Using LR=5e-4 + linear decay avoids this.
5. **Convergence:** Based on the paper (Fig. 7), expect meaningful accuracy around epoch 200–300.
