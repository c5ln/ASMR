import timm
import torch.nn as nn


def build_maxvit_s(num_classes: int = 36, in_chans: int = 2) -> nn.Module:
    """
    MaxViT-S from Tu et al. (2022), adapted for our task.

    Architecture (Table 11, MaxViT paper):
      Stem : 3×3 Conv ×2, C=64
      S1   : MBConv C=96,  Block-SA P=7 H=3, Grid-SA G=7 H=3  ×2
      S2   : MBConv C=192, Block-SA P=7 H=6, Grid-SA G=7 H=6  ×2
      S3   : MBConv C=384, Block-SA P=7 H=12,Grid-SA G=7 H=12 ×5
      S4   : MBConv C=768, Block-SA P=7 H=24,Grid-SA G=7 H=24 ×2
      Head : GlobalAvgPool → FC(num_classes)

    Input  : (B, in_chans, 224, 224)  – mel-spectrograms resized to 224×224
    Output : (B, num_classes)
    """
    model = timm.create_model(
        'maxvit_small_tf_224',
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    return model


if __name__ == '__main__':
    import torch
    model = build_maxvit_s()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MaxViT-S  trainable params: {n/1e6:.1f}M")
    x = torch.zeros(2, 2, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}  →  Output: {out.shape}")  # (2, 36)
