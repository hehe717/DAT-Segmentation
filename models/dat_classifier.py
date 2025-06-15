import torch.nn as nn

from models.backbones.dat import DAT
from models.heads.cls_head import ClsHead

__all__ = ["DatClassifier"]


class DatClassifier(nn.Module):
    """DAT 백본 + ClsHead 분류 모델 (ImageNet 등)."""

    def __init__(self, num_classes: int = 1000, pretrained: str | None = None) -> None:
        super().__init__()
        self.backbone = DAT(num_classes=num_classes, init_cfg=dict(type="Pretrained", checkpoint=pretrained))
        # DAT 기본 설정에서 마지막 stage 출력 채널은 768
        self.head = ClsHead(in_channels=768, num_classes=num_classes)

    def forward(self, x):  # type: ignore
        feats = self.backbone(x)  # List[Tensor]
        return self.head(feats) 