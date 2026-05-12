import argparse

import numpy as np
import torch
import librosa
import soundfile as sf

from dataset import isolate_keystrokes, _melspec, _normalize, KEYS, TARGET_SIZE, SAMPLE_RATE
from model import build_model


def preprocess(stroke: np.ndarray) -> torch.Tensor:
    """(2, KEYSTROKE_LEN) → (1, 2, TARGET_SIZE, TARGET_SIZE) 텐서"""
    specs = np.stack([_melspec(stroke[c]) for c in range(stroke.shape[0])], axis=0)
    _normalize(specs)
    t = torch.from_numpy(specs).unsqueeze(0)   # (1, 2, 64, 64)
    return torch.nn.functional.interpolate(
        t, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False
    )


@torch.no_grad()
def predict_strokes(model, strokes, device, topk=5):
    """
    strokes : list of (2, KEYSTROKE_LEN) ndarray
    Returns : list of [(key, prob), ...] top-k 결과
    """
    results = []
    for stroke in strokes:
        x      = preprocess(stroke).to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1)[0]
        tk     = probs.topk(topk)
        results.append([(KEYS[i], p.item()) for i, p in zip(tk.indices, tk.values)])
    return results


def load_model(ckpt_path: str, device: torch.device):
    model = build_model().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'체크포인트 로드 완료  (epoch={ckpt["epoch"]}, val_top1={ckpt["val_top1"]:.1f}%)')
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.ckpt, device)

    # 헤더 손상 WAV 대응: 청크 단위로 읽어 EOF에서 자동 종료
    chunks = []
    with sf.SoundFile(args.wav) as f:
        native_sr = f.samplerate
        block_size = native_sr * 60   # 1분씩 읽기
        for block in f.blocks(blocksize=block_size, dtype='float32', always_2d=True):
            chunks.append(block)
    raw = np.concatenate(chunks, axis=0).T   # (channels, frames)
    if raw.ndim == 1:
        raw = np.stack([raw, raw], axis=0)

    # 학습 시 사용한 SAMPLE_RATE로 리샘플링
    if native_sr != SAMPLE_RATE:
        raw = np.stack([
            librosa.resample(raw[c], orig_sr=native_sr, target_sr=SAMPLE_RATE)
            for c in range(raw.shape[0])
        ], axis=0)
    signal = raw.astype(np.float32)
    sr = SAMPLE_RATE

    strokes = isolate_keystrokes(signal, sr)
    print(f'\n감지된 키스트로크: {len(strokes)}개\n')

    results  = predict_strokes(model, strokes, device, topk=args.topk)
    sequence = ''.join(r[0][0] for r in results)

    for i, preds in enumerate(results, 1):
        top1_key, top1_conf = preds[0]
        candidates = '  '.join(f'{k}({p*100:.1f}%)' for k, p in preds)
        print(f'  [{i:2d}]  Top-1: {top1_key} ({top1_conf*100:.1f}%)   |   {candidates}')

    print(f'\n예측 시퀀스: {sequence}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Acoustic Side-Channel — 실시간 추론')
    p.add_argument('wav',               help='분석할 .wav 파일 경로')
    p.add_argument('--ckpt',  default='checkpoints/best_model.pt')
    p.add_argument('--topk',  type=int, default=5, help='출력할 후보 수')
    main(p.parse_args())
