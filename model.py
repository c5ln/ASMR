import timm
import torch
import torch.nn as nn

NUM_CLASSES = 36
IN_CHANS    = 2


class ConvNextV2Classifier(nn.Module):
    """
    Input : (B, 2, 224, 224)  — stereo mel-spectrogram
    Output: (B, num_classes)

    구조:
      adapter  : Conv2d(2→3, 1×1)  — 2ch stereo를 3ch RGB 공간으로 projection
      backbone : ConvNextV2 Atto pretrained on ImageNet-21k
      head     : backbone 내장 (GlobalAvgPool → LayerNorm → Linear)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # 2ch → 3ch 어댑터 (파라미터 6개)
        self.adapter = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        # 초기화: 두 채널을 균등 평균 → pretrained 입력 분포에 가깝게 시작
        nn.init.constant_(self.adapter.weight, 1 / 2)

        # pretrained backbone (3ch, 224×224 기대)
        self.backbone = timm.create_model(
            'convnextv2_atto',
            pretrained=True,
            in_chans=3,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adapter(x))


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    return ConvNextV2Classifier(num_classes)


if __name__ == '__main__':
    m = build_model()
    x = torch.randn(2, IN_CHANS, 224, 224)
    y = m(x)
    print(f'Input : {tuple(x.shape)}')
    print(f'Output: {tuple(y.shape)}')
    total = sum(p.numel() for p in m.parameters())
    print(f'Total params: {total / 1e6:.2f}M')
    adapter_params = sum(p.numel() for p in m.adapter.parameters())
    print(f'Adapter params: {adapter_params}')
