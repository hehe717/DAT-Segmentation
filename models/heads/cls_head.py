import torch
import torch.nn as nn
from typing import Sequence, Union

from models.utils.dat_blocks import LayerNormProxy  # 동일 노멀라이즈 사용을 위해

__all__ = ["ClsHead"]


class ClsHead(nn.Module):
    """DAT 백본과 동일한 레이어‧노름을 사용하는 간단한 분류 헤드.

    처리 순서:
        1. LayerNormProxy (원본 코드와 동일)
        2. 글로벌 평균 풀링
        3. (선택) 드롭아웃
        4. FC 로짓 층

    입력 `x` 는 (B,C,H,W) 또는 feature map 리스트 모두 허용.
    이제 항상 로짓(`logits`) 텐서 하나만 반환한다.
    """

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
        # Allow passing list/tuple of feature maps
        if isinstance(x, (list, tuple)):
            x = x[-1]

        x = self.norm(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits 