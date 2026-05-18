"""
ASMR Keystroke Classifier - Inference
JH. HA
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import librosa
import timm

# ── Constants (must match dataset.py / train.py exactly) ──────────────────────
KEYS       = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
IDX_TO_KEY = {i: k for i, k in enumerate(KEYS)}

SAMPLE_RATE   = 44100
BEFORE        = 2400
AFTER         = 12000
KEYSTROKE_LEN = BEFORE + AFTER   # 14400 samples ≈ 0.33 s

N_MELS        = 128
N_FFT         = 1024
HOP_LENGTH    = 112
N_TIME_FRAMES = 128
TARGET_SIZE   = 128

MODEL_NAME        = "caformer_s36.sail_in22k_ft_in1k"
N_CLASSES         = 36
BACKBONE_FEAT_DIM = 512
POOL_OUT          = (2, 2)
FC_IN             = BACKBONE_FEAT_DIM * POOL_OUT[0] * POOL_OUT[1]   # 2048


# ── Model (identical to train.py) ─────────────────────────────────────────────

class CAFormerClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.pool2d   = nn.AdaptiveAvgPool2d(POOL_OUT)
        self.fc       = nn.Linear(FC_IN, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        feat = self.pool2d(feat)
        feat = feat.flatten(1)
        return self.fc(feat)


# ── Preprocessing (test mode pipeline from dataset.py) ────────────────────────

def _melspec(signal_1d: np.ndarray) -> np.ndarray:
    """(KEYSTROKE_LEN,) float32 → (N_MELS, N_TIME_FRAMES) float32 dB"""
    S    = librosa.feature.melspectrogram(
        y=signal_1d, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    t    = S_db.shape[1]
    if t >= N_TIME_FRAMES:
        return S_db[:, :N_TIME_FRAMES].astype(np.float32)
    return np.pad(S_db, ((0, 0), (0, N_TIME_FRAMES - t))).astype(np.float32)


def _normalize(spec: np.ndarray) -> np.ndarray:
    """(3, H, W) → min-max normalize per channel in-place to [0, 1]"""
    for c in range(spec.shape[0]):
        lo, hi = spec[c].min(), spec[c].max()
        if hi > lo:
            spec[c] = (spec[c] - lo) / (hi - lo)
        else:
            spec[c] = 0.0
    return spec


def preprocess(stroke: np.ndarray) -> torch.Tensor:
    """
    stroke : (2, KEYSTROKE_LEN) stereo float32
    Returns: (1, 3, TARGET_SIZE, TARGET_SIZE) tensor  — no augmentation (test mode)
    """
    L    = stroke[0]
    R    = stroke[1]
    mono = (L + R) / 2.0

    specs = np.stack([_melspec(L), _melspec(R), _melspec(mono)], axis=0)  # (3, 128, 128)
    specs = _normalize(specs)
    t     = torch.from_numpy(specs)

    t = torch.nn.functional.interpolate(
        t.unsqueeze(0),
        size=(TARGET_SIZE, TARGET_SIZE),
        mode='bilinear',
        align_corners=False,
    )  # (1, 3, TARGET_SIZE, TARGET_SIZE)
    return t


# ── Audio loading ──────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio resampled to SAMPLE_RATE as stereo (2, N) float32."""
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if signal.ndim == 1:
        signal = np.stack([signal, signal], axis=0)
    return signal.astype(np.float32)


# ── Keystroke isolation ────────────────────────────────────────────────────────

def _detect_peaks(mono: np.ndarray, size=48, scan=24, threshold=0.06) -> list:
    fft    = librosa.stft(mono.astype(np.float32), n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(np.float64)
    peaks  = np.where(energy > threshold)[0]

    prev_end   = -SAMPLE_RATE * 0.1
    timestamps = []
    for peak in peaks:
        t = int(peak * scan + size // 2)
        if t > prev_end + 0.1 * SAMPLE_RATE:
            s, e = t - BEFORE, t + AFTER
            if s >= 0 and e <= len(mono):
                timestamps.append((s, e))
                prev_end = t + AFTER
    return timestamps


def _isolate_from_long(stereo: np.ndarray) -> list:
    """Peak detection on recordings longer than 2×KEYSTROKE_LEN."""
    mono    = stereo.mean(axis=0)
    mono_tr = mono[SAMPLE_RATE:]   # skip first second (same as dataset.py)
    offset  = SAMPLE_RATE

    prom, step = 0.06, 0.005
    timestamps = []
    for _ in range(1000):
        timestamps = _detect_peaks(mono_tr, threshold=prom)
        if timestamps:
            break
        prom -= step
        if prom <= 0:
            break

    if not timestamps:
        return []

    strokes = []
    for s, e in timestamps:
        sg, eg = s + offset, e + offset
        if sg >= 0 and eg <= stereo.shape[1]:
            strokes.append(stereo[:, sg:eg].copy())
    return strokes


def extract_strokes(stereo: np.ndarray) -> list:
    """
    Short clip (≤ 2×KEYSTROKE_LEN) → single keystroke with pad/trim.
    Long recording                  → automatic peak detection.
    Returns list of (2, KEYSTROKE_LEN) ndarrays.
    """
    n_samples = stereo.shape[1]

    if n_samples <= KEYSTROKE_LEN * 2:
        if n_samples >= KEYSTROKE_LEN:
            stroke = stereo[:, :KEYSTROKE_LEN]
        else:
            pad    = KEYSTROKE_LEN - n_samples
            stroke = np.pad(stereo, ((0, 0), (0, pad)))
        return [stroke.copy()]

    return _isolate_from_long(stereo)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> CAFormerClassifier:
    model = CAFormerClassifier(n_classes=N_CLASSES)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: CAFormerClassifier, tensor: torch.Tensor,
            device: torch.device, top_k: int) -> list:
    """Returns [(key_char, confidence), ...] sorted by confidence descending."""
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0]

    k               = min(top_k, N_CLASSES)
    values, indices = torch.topk(probs, k)
    return [(IDX_TO_KEY[idx.item()], val.item()) for idx, val in zip(indices, values)]


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASMR Keystroke Classifier - Inference")
    parser.add_argument("--audio", required=True,
                        help="입력 오디오 파일 경로 (.wav, .mp3, ...)")
    parser.add_argument("--model", default="./final_train_results/best_model.pth",
                        help="학습된 모델 가중치 경로 (default: ./final_train_results/best_model.pth)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="출력할 상위 후보 수 (default: 3)")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"[오류] 오디오 파일 없음: {args.audio}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"[오류] 모델 파일 없음: {args.model}")
        sys.exit(1)

    device = get_device()
    print(f"[Device] {device}")

    print(f"[*] 모델 로드 중... ({args.model})")
    model = load_model(args.model, device)
    print("[*] 모델 로드 완료.\n")

    print(f"[*] 오디오 로드 중... ({args.audio})")
    stereo  = load_audio(args.audio)
    dur_sec = stereo.shape[1] / SAMPLE_RATE
    print(f"[*] 오디오 길이: {dur_sec:.2f}s")

    strokes = extract_strokes(stereo)
    n       = len(strokes)

    if n == 0:
        print("[오류] 키스트로크를 감지하지 못했습니다. 오디오를 확인하세요.")
        sys.exit(1)

    print(f"[*] 감지된 키스트로크: {n}개\n")
    print("─" * 30)

    for i, stroke in enumerate(strokes, 1):
        tensor  = preprocess(stroke)
        results = predict(model, tensor, device, args.top_k)

        print(f"[keystroke {i}]")
        for rank, (key, conf) in enumerate(results, 1):
            marker = "★" if rank == 1 else " "
            print(f"  {marker} {rank}위: {key}  ({conf * 100:5.1f}%)")
        print()


if __name__ == "__main__":
    main()
