# coding: utf-8
from __future__ import annotations

import importlib
import torch.nn as nn

from models.backbones.dat import DAT
from models.heads.cls_head import ClsHead

__all__ = ["DatClassifier"]


class DatClassifier(nn.Module):
    """DAT 백본 + ClsHead 분류 모델 (ImageNet 등)."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        backbone_cfg = self._load_backbone_cfg().copy()
        self.backbone = DAT(**backbone_cfg, num_classes=num_classes)
        self.head = ClsHead(in_channels=backbone_cfg["dims"], num_classes=num_classes)

    @staticmethod
    def _load_backbone_cfg() -> dict:
        cfg_module = importlib.import_module("configs.dat.upn_tiny_160k_dp03_lr6")
        return cfg_module.model["backbone"]

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats) 