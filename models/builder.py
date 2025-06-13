import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch

from backbones.loading import load_checkpoint

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _import_config(config_path: str) -> Any:
    """Dynamically import a Python config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _resolve_pretrained_path(cfg_module, config_dir: Path) -> str | None:
    pretrained = getattr(cfg_module, "pretrained", None)
    if not pretrained or pretrained.startswith("<"):
        return None
    # If relative path, resolve against config directory
    pretrained_path = Path(pretrained)
    if not pretrained_path.is_absolute():
        pretrained_path = (config_dir / pretrained).resolve()
    return str(pretrained_path)


# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

def build_model_from_config(config_path: str) -> torch.nn.Module:
    device = "cuda"  # GPU 전용
    cfg_module = _import_config(config_path)
    cfg = cfg_module.model  # dict defined in the mmseg-style config

    # Dynamically import backbone class
    backbone_type = cfg["backbone"]["type"]
    if backbone_type != "DAT":
        raise NotImplementedError("Only DAT backbone is supported in this builder.")

    from models.backbones.dat import DAT  # local import to avoid heavy deps at startup

    backbone_cfg = cfg["backbone"].copy()
    init_cfg = backbone_cfg.pop("init_cfg", None)

    model = DAT(**backbone_cfg).to(device)

    # Load pretrained weights if provided
    pretrained_path = _resolve_pretrained_path(cfg_module, Path(config_path).parent)
    if pretrained_path:
        load_checkpoint(model, pretrained_path, map_location=device)

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Export utilities (ONNX 제거됨)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DAT segmentation model builder & exporter")
    parser.add_argument("config", help="Path to the config .py file (e.g., upn_tiny_160k_dp03_lr6.py)")
    args = parser.parse_args()

    model = build_model_from_config(args.config)
    print("Model built on GPU (cuda)")


if __name__ == "__main__":
    main() 