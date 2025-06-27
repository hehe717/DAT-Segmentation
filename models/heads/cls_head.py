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

        self.norms = nn.ModuleList([LayerNormProxy(c) for c in in_channels])
        total_channels = sum(in_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.fc = nn.Linear(total_channels, num_classes)

    def forward(self, x: Sequence[torch.Tensor]):
        outs = []
        for i, feat in enumerate(x):
            y = self.norms[i](feat)
            y = self.pool(y)
            y = torch.flatten(y, 1)
            outs.append(y)
        x = torch.cat(outs, dim=1)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits 