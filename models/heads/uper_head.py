import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UPerHead"]


class UPerHead(nn.Module):
    """UPerHead (PPM + FPN) re-implementation closely following MMSegmentation.

    주요 구성
    1) PPM (Pyramid Pooling Module) – 최저 해상도 feature에 global context 추가
    2) top-down FPN – 레벨 간 정보를 합성해 세밀한 공간 정보를 복원
    3) Fuse head – 모든 레벨을 upsample & concat 후 3×3 conv → dropout → cls
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        *,
        channels: int = 512,
        pool_scales: tuple[int, int, int, int] = (1, 2, 3, 6),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        self.align_corners = align_corners
        self.num_levels = len(in_channels)

        # ------------------------------------------------------------------
        # 1. Pyramid Pooling Module (on deepest feature)
        # ------------------------------------------------------------------
        self.ppm_modules = nn.ModuleList()
        ppm_out_channels = channels // len(pool_scales)
        for scale in pool_scales:
            self.ppm_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels[-1], ppm_out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ppm_out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # bottleneck after PPM concat
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * ppm_out_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # 2. Lateral & FPN convs
        # ------------------------------------------------------------------
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ic in in_channels[:-1]:  # skip last, handled by PPM bottleneck
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ic, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # ------------------------------------------------------------------
        # 3. Fusing final feature maps
        # ------------------------------------------------------------------
        self.fuse_bottleneck = nn.Sequential(
            nn.Conv2d(self.num_levels * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def _ppm_forward(self, x: torch.Tensor) -> torch.Tensor:
        ppm_outs = [x]
        for ppm in self.ppm_modules:
            ppm_out = ppm(x)
            ppm_out = F.interpolate(ppm_out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)
            ppm_outs.append(ppm_out)
        ppm_out = torch.cat(ppm_outs, dim=1)
        return self.ppm_bottleneck(ppm_out)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == self.num_levels, "Number of input features mismatch."

        # feats: [C1(h), C2, C3, C4(l)]  (0=highest resolution)

        # 1) build laterals for levels 0~2 (skip last)
        laterals = [l_conv(feats[i]) for i, l_conv in enumerate(self.lateral_convs)]

        # 2) PPM on last level
        laterals.append(self._ppm_forward(feats[-1]))  # index = num_levels -1

        # 3) build top-down path
        for i in range(self.num_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsample = F.interpolate(laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners)
            laterals[i - 1] = laterals[i - 1] + upsample
            # refine
            if i - 1 < len(self.fpn_convs):
                laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # 4) Gather all features to the highest resolution
        target_size = laterals[0].shape[2:]
        outs = [laterals[0]]
        for i in range(1, self.num_levels):
            outs.append(
                F.interpolate(laterals[i], size=target_size, mode="bilinear", align_corners=self.align_corners)
            )

        x = torch.cat(outs, dim=1)
        x = self.fuse_bottleneck(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x 