import os
from collections import OrderedDict
from typing import Any, Dict

import torch

__all__ = ["load_checkpoint"]


def load_checkpoint(
    model: torch.nn.Module,
    filename: str,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
    logger: Any | None = None,
) -> Dict[str, Any] | OrderedDict:
    def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if sd and next(iter(sd)).startswith("module."):
            return {k[7:]: v for k, v in sd.items()}
        return sd

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _strip_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if logger is not None:
        if missing_keys:
            logger.warning("Missing keys when loading checkpoint: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)

    return checkpoint
