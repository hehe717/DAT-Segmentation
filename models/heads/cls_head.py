import torch
import torch.nn as nn
from typing import Sequence

from models.utils.dat_blocks import LayerNormProxy

__all__ = ["ClsHead"]


class ClsHead(nn.Module):

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        *,
        dropout_ratio: float = 0.0,
    ) -> None:
        super().__init__()

        # 마지막 피쳐(가장 높은 stage 출력)만 사용
        last_channels = in_channels[-1]
        self.norm = LayerNormProxy(last_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.fc = nn.Linear(last_channels, num_classes)

    def forward(self, x: Sequence[torch.Tensor]):
        # 시퀀스의 마지막 피쳐만 활용
        feat = x[-1]
        feat = self.norm(feat)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits 