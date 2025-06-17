import torch
import torch.nn as nn
from typing import Sequence, Union

from models.utils.dat_blocks import LayerNormProxy

__all__ = ["ClsHead"]


class ClsHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        dropout_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = LayerNormProxy(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]):  # type: ignore
        if isinstance(x, (list, tuple)):
            x = x[-1]

        x = self.norm(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits 