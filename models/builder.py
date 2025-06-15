import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa

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
# Import local modules lazily to avoid heavy deps at startup
# -----------------------------------------------------------------------------

from models.backbones.loading import load_checkpoint

# Head registry
_HEAD_REGISTRY = {}


def _register_head(name):
    """Decorator to register head classes by name (mimicking mmseg)."""

    def _wrapper(cls):
        _HEAD_REGISTRY[name] = cls
        return cls

    return _wrapper


# Local imports are placed inside try-except to allow builder import before files
try:
    from models.heads.uper_head import UPerHead  # type: ignore
    from models.heads.fcn_head import FCNHead  # type: ignore
    from models.heads.cls_head import ClsHead  # type: ignore

    _register_head("UPerHead")(UPerHead)
    _register_head("FCNHead")(FCNHead)
    _register_head("ClsHead")(ClsHead)
except ModuleNotFoundError:
    # Heads may not exist in some environments; builder can still create backbone only.
    pass

# Segmentor wrapper
try:
    from models.segmentor import EncoderDecoder
except ModuleNotFoundError:
    EncoderDecoder = None  # type: ignore


# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

def build_model_from_config(config_path: str) -> torch.nn.Module:
    """Build full segmentation model based on an mmseg-style config.

    Currently supports:
        • DAT backbone
        • UPerHead decode head
        • FCNHead auxiliary head (optional)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_module = _import_config(config_path)
    cfg = cfg_module.model  # dict defined in the mmseg-style config

    # ---------------- Backbone ----------------
    backbone_type = cfg["backbone"]["type"]
    if backbone_type != "DAT":
        raise NotImplementedError("Only DAT backbone is supported in this builder.")

    from models.backbones.dat import DAT  # local import to avoid heavy deps at startup

    backbone_cfg = cfg["backbone"].copy()
    _ = backbone_cfg.pop("init_cfg", None)  # init_cfg handled externally

    backbone = DAT(**backbone_cfg).to(device)

    # Load pretrained weights if provided
    pretrained_path = _resolve_pretrained_path(cfg_module, Path(config_path).parent)
    if pretrained_path:
        load_checkpoint(backbone, pretrained_path, map_location=device)

    # If no decode_head in cfg, return backbone only (for backward compatibility)
    if "decode_head" not in cfg:
        backbone.eval()
        return backbone

    # ---------------- Decode head ----------------
    decode_cfg = cfg["decode_head"].copy()
    head_type = decode_cfg.pop("type", "UPerHead")
    head_cls = _HEAD_REGISTRY.get(head_type)
    if head_cls is None:
        raise NotImplementedError(f"Unsupported decode head type: {head_type}")

    # Extract required params
    in_channels = decode_cfg.pop("in_channels")
    num_classes = decode_cfg.pop("num_classes")

    import inspect

    # Filter kwargs to only those accepted by the head's __init__
    head_sig = inspect.signature(head_cls.__init__)
    allowed = set(head_sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in decode_cfg.items() if k in allowed}

    decode_head = head_cls(in_channels=in_channels, num_classes=num_classes, **filtered_kwargs).to(device)

    # ---------------- Auxiliary head (optional) ----------------
    aux_head = None
    if "auxiliary_head" in cfg:
        aux_cfg = cfg["auxiliary_head"].copy()
        aux_type = aux_cfg.pop("type", "FCNHead")
        aux_cls = _HEAD_REGISTRY.get(aux_type)
        if aux_cls is None:
            raise NotImplementedError(f"Unsupported auxiliary head type: {aux_type}")

        in_channels_aux = aux_cfg.pop("in_channels")
        num_classes_aux = aux_cfg.pop("num_classes")

        aux_sig = inspect.signature(aux_cls.__init__)
        allowed_aux = set(aux_sig.parameters.keys()) - {"self"}
        filtered_aux_kwargs = {k: v for k, v in aux_cfg.items() if k in allowed_aux}

        aux_head = aux_cls(in_channels=in_channels_aux, num_classes=num_classes_aux, **filtered_aux_kwargs).to(device)

    # ---------------- Segmentor wrapper ----------------
    if EncoderDecoder is None:
        raise RuntimeError("models.segmentor.EncoderDecoder could not be imported.")

    model = EncoderDecoder(backbone, decode_head, aux_head).to(device)
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
    print("Model built successfully, {}".format(model))


if __name__ == "__main__":
    main() 