import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
KEYS = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
LABEL_MAP = {k: i for i, k in enumerate(KEYS)}

SAMPLE_RATE   = 44100
BEFORE        = 2400
AFTER         = 12000
KEYSTROKE_LEN = BEFORE + AFTER   # 14400 samples ≈ 0.33 s

N_MELS         = 64
N_FFT          = 1024
HOP_LENGTH     = 225
N_TIME_FRAMES  = 64     # crop mel-spectrogram time axis to this
TARGET_SIZE    = 256   # pretrained MobileViT-XXS 기대 입력 크기

TIMESHIFT_FRAC    = 0.1   # ± 10 % — BEFORE=2400이므로 max_shift=1440 < 2400, wrap-around 없음
MASK_MAX_FRAC     = 0.15  # each mask up to 15 % of axis length
N_MASKS_PER_AXIS  = 2


# ── Step 2: Keystroke Isolation ───────────────────────────────────────────────

def _detect_timestamps(signal, sample_rate, size, scan, before, after, threshold):
    """Return (start, end) sample pairs for each detected keystroke."""
    fft    = librosa.stft(signal.astype(np.float32), n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(np.float64)
    peaks  = np.where(energy > threshold)[0]

    prev_end   = -sample_rate * 0.1
    timestamps = []
    for peak in peaks:
        t = int(peak * scan + size // 2)
        if t > prev_end + 0.1 * sample_rate:
            s, e = t - before, t + after
            if s >= 0 and e <= len(signal):
                timestamps.append((s, e))
                prev_end = t + after
    return timestamps


def isolate_keystrokes(stereo, sample_rate, target=25,
                       size=48, scan=24, before=BEFORE, after=AFTER):
    """
    stereo : (2, N) float32 ndarray
    Returns list of (2, KEYSTROKE_LEN) float32 ndarrays (exactly `target` items).
    Detection is done on the mono mix; extraction uses the full stereo signal.
    """
    # Use mean of channels for peak detection; skip first second
    mono    = stereo.mean(axis=0)
    mono_tr = mono[sample_rate:]

    prom, step = 0.06, 0.005
    timestamps = []
    while len(timestamps) != target:
        timestamps = _detect_timestamps(mono_tr, sample_rate, size, scan,
                                        before, after, prom)
        if len(timestamps) < target:
            prom -= step
        elif len(timestamps) > target:
            prom += step
        if prom <= 0:
            print(f"  Warning: could not reach {target} keystrokes "
                  f"(found {len(timestamps)})")
            break
        step *= 0.99

    offset  = sample_rate          # timestamps are relative to trimmed signal
    strokes = []
    for s, e in timestamps:
        sg, eg = s + offset, e + offset
        if sg >= 0 and eg <= stereo.shape[1]:
            strokes.append(stereo[:, sg:eg].copy())
    return strokes


# ── Step 1: Data Loading ──────────────────────────────────────────────────────

def load_all_keystrokes(wav_dir):
    """
    Load all 36 .wav files from `wav_dir`, isolate 25 keystrokes each.
    Returns list of (label_int, ndarray(2, KEYSTROKE_LEN)).
    """
    all_data = []
    for key in KEYS:
        path = os.path.join(wav_dir, f'{key}.wav')
        if not os.path.exists(path):
            print(f"  Skipping missing file: {path}")
            continue

        signal, sr = librosa.load(path, sr=None, mono=False)

        # librosa returns (N,) for mono files — duplicate to get 2 channels
        if signal.ndim == 1:
            signal = np.stack([signal, signal], axis=0)

        signal = signal.astype(np.float32)
        strokes = isolate_keystrokes(signal, sr)
        label   = LABEL_MAP[key]
        for stroke in strokes:
            all_data.append((label, stroke))
        print(f"  {key}: {len(strokes)} keystrokes isolated")

    print(f"Total: {len(all_data)} keystrokes ({len(KEYS)} keys × ~25)")
    return all_data


# ── Step 3: Augmentation & Preprocessing ─────────────────────────────────────

def _time_shift(stroke):
    """Circular shift by a random fraction of signal length (± TIMESHIFT_FRAC)."""
    max_shift = int(TIMESHIFT_FRAC * stroke.shape[1])
    shift = random.randint(-max_shift, max_shift)
    return np.roll(stroke, shift, axis=1)


def _melspec(signal_1d):
    """
    signal_1d : (KEYSTROKE_LEN,) float32
    Returns    : (N_MELS, N_TIME_FRAMES) float32  in dB, cropped to N_TIME_FRAMES
    """
    S    = librosa.feature.melspectrogram(
        y=signal_1d, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    t    = S_db.shape[1]
    if t >= N_TIME_FRAMES:
        return S_db[:, :N_TIME_FRAMES].astype(np.float32)
    # pad if somehow shorter
    return np.pad(S_db, ((0, 0), (0, N_TIME_FRAMES - t))).astype(np.float32)


def _normalize(spec):
    """Min-max normalize each channel of spec (C, H, W) independently to [0, 1]."""
    for c in range(spec.shape[0]):
        lo, hi = spec[c].min(), spec[c].max()
        if hi > lo:
            spec[c] = (spec[c] - lo) / (hi - lo)
        else:
            spec[c] = 0.0
    return spec


def _spec_augment(spec):
    """
    spec : (C, N_MELS, N_TIME_FRAMES) float32 tensor
    Apply N_MASKS_PER_AXIS frequency + time masks; fill with channel mean.
    """
    spec = spec.clone()
    _, F, T = spec.shape

    for _ in range(N_MASKS_PER_AXIS):
        # frequency mask — fill with 0 (묵음, 정규화 후 최솟값)
        f_w = random.randint(1, max(1, int(MASK_MAX_FRAC * F)))
        f0  = random.randint(0, F - f_w)
        spec[:, f0:f0 + f_w, :] = 0.0

        # time mask — fill with 0
        t_w = random.randint(1, max(1, int(MASK_MAX_FRAC * T)))
        t0  = random.randint(0, T - t_w)
        spec[:, :, t0:t0 + t_w] = 0.0

    return spec


# ── Step 4: Dataset Class ─────────────────────────────────────────────────────

class KeystrokeDataset(Dataset):
    """
    Wraps pre-isolated raw keystrokes.
    Per-sample pipeline:
      train : time_shift → melspec → normalize → tensor → spec_augment → resize 224
      val/test :            melspec → normalize → tensor              → resize 224
    """

    def __init__(self, data, mode='train', aug_factor=1):
        """
        data       : list of (label_int, ndarray(2, KEYSTROKE_LEN))
        mode       : 'train' | 'val' | 'test'
        aug_factor : train 샘플을 몇 배로 늘릴지 (train 전용, val/test는 무시)
        """
        self.data       = data
        self.mode       = mode
        self.aug_factor = aug_factor if mode == 'train' else 1

    def __len__(self):
        return len(self.data) * self.aug_factor

    def __getitem__(self, idx):
        label, stroke = self.data[idx % len(self.data)]
        stroke = stroke.copy()   # (2, KEYSTROKE_LEN)

        if self.mode == 'train':
            stroke = _time_shift(stroke)

        # mel-spectrogram per channel → (2, 64, 64)
        specs = np.stack([_melspec(stroke[c]) for c in range(stroke.shape[0])], axis=0)

        _normalize(specs)                   # in-place [0, 1]
        specs = torch.from_numpy(specs)     # (2, 64, 64)

        if self.mode == 'train':
            specs = _spec_augment(specs)

        # resize to (2, TARGET_SIZE, TARGET_SIZE) for ConvNextV2
        specs = torch.nn.functional.interpolate(
            specs.unsqueeze(0),
            size=(TARGET_SIZE, TARGET_SIZE),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        return specs, torch.tensor(label, dtype=torch.long)


def get_dataloaders(wav_dir, batch_size=16, val_ratio=0.1, test_ratio=0.1, seed=42,
                    aug_factor=25, num_workers=0):
    """
    Load → isolate → split randomly (paper: "Data Split: Random") → return DataLoaders.
    Returns (train_loader, val_loader, test_loader, num_classes).
    """
    all_data = load_all_keystrokes(wav_dir)

    train_data, tmp_data = train_test_split(
        all_data, test_size=val_ratio + test_ratio,
        random_state=seed
    )
    val_data, test_data = train_test_split(
        tmp_data, test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )
    print(f"Split → train:{len(train_data)}  val:{len(val_data)}  test:{len(test_data)}")
    print(f"Aug   → train:{len(train_data) * aug_factor} (×{aug_factor})  val:{len(val_data)}  test:{len(test_data)}")

    loaders = {}
    for name, data, mode, shuffle, factor in [
        ('train', train_data, 'train', True,  aug_factor),
        ('val',   val_data,   'val',   False, 1),
        ('test',  test_data,  'test',  False, 1),
    ]:
        loaders[name] = DataLoader(
            KeystrokeDataset(data, mode=mode, aug_factor=factor),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return loaders['train'], loaders['val'], loaders['test'], len(KEYS)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    wav_dir = sys.argv[1] if len(sys.argv) > 1 else 'MBPWavs'

    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        wav_dir, batch_size=4
    )

    batch, labels = next(iter(train_loader))
    print(f"\nSanity check:")
    print(f"  batch shape : {batch.shape}")   # (4, 2, 64, 64)
    print(f"  label shape : {labels.shape}")  # (4,)
    print(f"  value range : [{batch.min():.3f}, {batch.max():.3f}]")
    print(f"  labels      : {labels.tolist()}")
    print(f"  num_classes : {n_classes}")
