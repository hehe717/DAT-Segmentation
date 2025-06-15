import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FCNHead"]


class FCNHead(nn.Module):
    """Minimal FCN head used as auxiliary head.

    Only one or two conv layers before classification, following MMSeg's
    default `num_convs=1` pattern.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: int = 256,
        num_convs: int = 1,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners

        convs = []
        last_channels = in_channels
        for _ in range(num_convs):
            convs.extend([
                nn.Conv2d(last_channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
            last_channels = channels
        self.convs = nn.Sequential(*convs)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.convs(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x 