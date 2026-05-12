import timm
import torch
import torch.nn as nn

NUM_CLASSES = 36
IN_CHANS    = 2


class MobileViTClassifier(nn.Module):
    """
    Input : (B, 2, 256, 256)  — stereo mel-spectrogram
    Output: (B, num_classes)

    구조:
      adapter  : Conv2d(2→3, 1×1)  — 2ch stereo를 3ch RGB 공간으로 projection
      backbone : MobileViT-XXS pretrained on ImageNet
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # 2ch → 3ch 어댑터 (파라미터 6개)
        self.adapter = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.adapter.weight, 1 / 2)

        # pretrained backbone
        self.backbone = timm.create_model(
            'mobilevit_xxs',
            pretrained=True,
            in_chans=3,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adapter(x))


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    return MobileViTClassifier(num_classes)


if __name__ == '__main__':
    m = build_model()
    x = torch.randn(2, IN_CHANS, 256, 256)
    y = m(x)
    print(f'Input : {tuple(x.shape)}')
    print(f'Output: {tuple(y.shape)}')
    total = sum(p.numel() for p in m.parameters())
    print(f'Total params   : {total / 1e6:.2f}M')
    print(f'Adapter params : {sum(p.numel() for p in m.adapter.parameters())}')
