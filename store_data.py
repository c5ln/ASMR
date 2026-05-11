import argparse
import os
import random

import numpy as np
import torch

from dataset import (
    MASK_MAX_FRAC,
    N_MASKS_PER_AXIS,
    TARGET_SIZE,
    _melspec,
    _normalize,
    _time_shift,
    load_all_keystrokes,
)


def _spec_augment_corrected(spec: torch.Tensor) -> torch.Tensor:
    """
    Corrected SpecAugment: fills masked regions with the GLOBAL spectrogram mean,
    matching the paper ("setting all values within those ranges to the mean of
    the spectrogram") — unlike dataset.py which uses the local region mean.

    spec : (C, N_MELS, N_TIME_FRAMES) float32 tensor
    """
    spec = spec.clone()
    _, F, T = spec.shape
    global_mean = spec.mean()

    for _ in range(N_MASKS_PER_AXIS):
        f_w = random.randint(1, max(1, int(MASK_MAX_FRAC * F)))
        f0  = random.randint(0, F - f_w)
        spec[:, f0:f0 + f_w, :] = global_mean

        t_w = random.randint(1, max(1, int(MASK_MAX_FRAC * T)))
        t0  = random.randint(0, T - t_w)
        spec[:, :, t0:t0 + t_w] = global_mean

    return spec


def build_augmented_dataset(wav_dir: str, n_augments: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    raw = load_all_keystrokes(wav_dir)
    n_base = len(raw)
    total = n_base * n_augments
    print(f"\nGenerating {n_augments} augmented version(s) per keystroke "
          f"({n_base} base × {n_augments} = {total} total)...")

    all_specs, all_labels = [], []

    for i, (label, stroke) in enumerate(raw):
        if i % 100 == 0:
            print(f"  [{i + 1}/{n_base}] processing...")
        for _ in range(n_augments):
            s = _time_shift(stroke.copy())

            mono = (s[0] + s[1]) / 2
            specs = np.stack([_melspec(s[0]), _melspec(s[1]), _melspec(mono)], axis=0)
            _normalize(specs)

            t = torch.from_numpy(specs.copy())   # .copy() avoids shared-memory aliasing
            t = _spec_augment_corrected(t)

            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(TARGET_SIZE, TARGET_SIZE),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            all_specs.append(t)
            all_labels.append(label)

    specs_tensor  = torch.stack(all_specs)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    return specs_tensor, labels_tensor


def main():
    p = argparse.ArgumentParser(
        description='Pre-generate augmented mel-spectrograms and save to disk.'
    )
    p.add_argument('--wav_dir',    default='MBPWavs',      help='Directory of raw .wav files')
    p.add_argument('--n_augments', type=int, default=10,   help='Augmented versions per keystroke')
    p.add_argument('--save_dir',   default='augment_data', help='Output directory')
    p.add_argument('--seed',       type=int, default=42,   help='Random seed for reproducibility')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'dataset.pt')

    specs, labels = build_augmented_dataset(args.wav_dir, args.n_augments, args.seed)

    torch.save({'specs': specs, 'labels': labels}, save_path)

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"\n{'─' * 54}")
    print(f"  Saved     : {save_path}")
    print(f"  Samples   : {len(labels):,}  (36 keys × 25 × {args.n_augments})")
    print(f"  Spec shape: {tuple(specs.shape)}  (N, 3, {TARGET_SIZE}, {TARGET_SIZE})")
    print(f"  Value range: [{specs.min():.3f}, {specs.max():.3f}]")
    print(f"  File size : {size_mb:.1f} MB")
    print(f"{'─' * 54}")


if __name__ == '__main__':
    main()
