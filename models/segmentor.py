from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """Lightweight segmentor that mimics MMSegmentation `EncoderDecoder`.

    It wraps a backbone, a main `decode_head`, and an optional `auxiliary_head`.
    Only the forward for simple supervised segmentation is implemented – no
    test-time augmentations or sliding window.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        auxiliary_head: nn.Module | None = None,
        *,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.auxiliary_head = auxiliary_head
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Returns
        -------
        If `self.training` and `auxiliary_head` is provided, returns a tuple
        `(main_logits, aux_logits)`. Otherwise returns only `main_logits`.
        """
        feats: List[torch.Tensor] = self.backbone(x)

        # Main decoder logits
        logits = self.decode_head(feats)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)

        # Optionally compute auxiliary logits; however, we always *return only* the
        # main logits to stay compatible with existing training scripts that
        # assume a single-tensor output. If users need the auxiliary branch they
        # can access `model.last_aux_logits` after a forward pass.

        if self.auxiliary_head is not None:
            aux_feat = feats[-2]  # usually in_index=2
            aux_logits = self.auxiliary_head(aux_feat)
            aux_logits = F.interpolate(aux_logits, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)
            # store for external usage (e.g., custom loss calculation)
            self.last_aux_logits = aux_logits.detach() if not self.training else aux_logits

            if self.training:
                return logits, aux_logits

        return logits 