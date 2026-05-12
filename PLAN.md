# 구현 계획: Acoustic Side Channel Attack with ConvNextV2

## 현재 상태

- `dataset.py` : 완성 (키스트로크 분리 → 멜 스펙트로그램 → 증강 → DataLoader)
- `model.py` : 삭제됨 → 재구현 필요
- `train.py` : 삭제됨 → 재구현 필요
- 데이터 : `MBPWavs/` — 0-9, A-Z (36개 키 × 25회 = 900 샘플)

---

## Phase 1: dataset.py 검토 및 수정

**파일**: `dataset.py`

**현황**: 이미 논문 전처리 파이프라인을 따름.  
논문 설정과 일치 여부 최종 확인:

| 파라미터 | 논문 | 현재 코드 | 조치 |
|---|---|---|---|
| 샘플 길이 | ~0.33 s | BEFORE=2400, AFTER=12000 → 14400 samples @ 44100 Hz ≈ 0.33 s | 일치, 유지 |
| 멜 빈 | 64 | N_MELS=64 | 일치, 유지 |
| FFT 윈도우 | 1024 | N_FFT=1024 | 일치, 유지 |
| Hop Length | 225 | HOP_LENGTH=225 | 일치, 유지 |
| 출력 크기 | 64×64 | TARGET_SIZE=64 | 일치 (pretrained 없이 64×64 사용) |
| 채널 | 2 (stereo) | stereo 처리 후 (2, 64, 64) | 유지 |

**수정 사항**:
- `TARGET_SIZE = 64` 유지 (ConvNextV2는 FCN 구조라 임의 해상도 입력 가능)
- 주석에서 "MaxViT-S" 참조 → "ConvNextV2" 로 교체

---

## Phase 2: model.py 구현

**파일**: `model.py` (새로 작성)

### ConvNextV2 Atto 아키텍처

`timm` 라이브러리의 `convnextv2_atto` 사용.  
사전학습 가중치 없이 처음부터 학습 (in_chans=2 이므로).

```
입력: (B, 2, 64, 64)
  ↓ Stem (4×4 Conv, stride 4, →40ch) → (B, 40, 16, 16)
  ↓ Stage 1 (2 blocks, 40ch)
  ↓ Downsample → Stage 2 (2 blocks, 80ch)
  ↓ Downsample → Stage 3 (6 blocks, 160ch)
  ↓ Downsample → Stage 4 (2 blocks, 320ch)
  ↓ GlobalAvgPool → LayerNorm → Linear(320, 36)
출력: (B, 36) logits
```

**ConvNextV2 블록 구조** (GRN 포함):
```
Depthwise 7×7 Conv → LayerNorm → Linear(4× expand) → GELU → GRN → Linear(1× project)
```

**GRN (Global Response Normalization)**:
```
X' = X * (X_norm / mean(X_norm))    where X_norm = L2-norm over spatial dims
```

**구현 방식**: `timm.create_model('convnextv2_atto', pretrained=False, in_chans=2, num_classes=36)`

파라미터 수 목표: ~3.7M (CoAtNet 69M 대비 약 95% 절감)

---

## Phase 3: Gaussian Label Smoothing 구현

**파일**: `train.py` 내 `GaussianLabelSmoothingLoss` 클래스

### 키보드 2D 좌표 (QWERTY 기준 행/열 인덱스)

```
행 0 (숫자행): 0 1 2 3 4 5 6 7 8 9
행 1 (Q행):    Q W E R T Y U I O P
행 2 (A행):     A S D F G H J K L
행 3 (Z행):      Z X C V B N M
```

각 키에 (row, col) 좌표 할당 → 36×36 거리 행렬 D 사전 계산.

### Soft Label 생성

```
target_dist[i][j] = exp(-D[i][j]² / (2σ²))
target_dist[i] = target_dist[i] / sum(target_dist[i])   # 정규화
```

σ = 1.0 (키보드 격자 단위 — 인접 키에 적당한 확률 분배)

### 손실 함수

```python
loss = F.kl_div(
    F.log_softmax(logits, dim=-1),
    soft_labels,          # Gaussian 분포로 만든 타겟
    reduction='batchmean'
)
```

---

## Phase 4: train.py 구현

**파일**: `train.py` (새로 작성)

### 구성 요소

1. **옵티마이저**: AdamW (weight_decay=0.05)
2. **스케줄러**: CosineAnnealingLR (T_max=epochs)
3. **Mixed Precision**: `torch.cuda.amp.autocast` + `GradScaler`
4. **체크포인트**: `checkpoints/best_model.pt` (val top-1 기준)
5. **로깅**: epoch별 train_loss, val_loss, val_top1, val_top5

### 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|---|---|---|
| Batch size | 32 | Colab T4 VRAM 15GB 기준 |
| Epochs | 200 | 소규모 데이터셋 (900 샘플) |
| Learning Rate | 1e-3 | 처음부터 학습 (no pretrained) |
| Weight Decay | 0.05 | ConvNextV2 논문 권장값 |
| σ (Gaussian) | 1.0 | 키보드 격자 단위 |
| Warmup Epochs | 20 | Linear warmup |

### 학습 루프 흐름

```
for epoch in epochs:
    train: forward → GaussianLoss → backward → optimizer.step
    val:   forward → top1/top5 accuracy 계산
    if val_top1 > best: save checkpoint
    lr_scheduler.step()
```

---

## Phase 5: 평가 계획

논문 목표 성능과 비교:

| 지표 | 논문 (CoAtNet) | 목표 (ConvNextV2) |
|---|---|---|
| Top-1 Accuracy | 95% | ≥ 90% |
| Top-5 Accuracy | ~99% | ≥ 98% |
| 파라미터 수 | 69M | ~3.7M |
| Colab 학습 시간 | 불가 | 30분 이내 |

---

## 구현 순서

```
[1] dataset.py  — 주석 수정, TARGET_SIZE=64 확인           (소요: 10분)
[2] model.py    — ConvNextV2 Atto 구현 (timm 활용)         (소요: 30분)
[3] train.py    — GaussianLabelSmoothing + 학습 루프       (소요: 60분)
[4] 로컬 sanity check — CPU로 1 epoch 돌려보기             (소요: 10분)
[5] Colab 업로드 후 본 학습                               (소요: 30분)
```

---

## 디렉터리 구조 (완성 후)

```
ASMR/
├── MBPWavs/          # 원본 .wav 파일 (36개 키)
├── checkpoints/      # 학습된 모델 가중치
├── dataset.py        # 전처리 파이프라인
├── model.py          # ConvNextV2 Atto 정의
├── train.py          # 학습 루프 + Gaussian Label Smoothing
├── requirements.txt
└── PLAN.md
```
