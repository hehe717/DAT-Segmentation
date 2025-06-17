from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random


import albumentations as A  # type: ignore

__all__ = ["ADE20KDataset"]


IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_CROP_SIZE: Tuple[int, int] = (512, 512)
_IMG_SCALE: Tuple[int, int] = (2048, 512)  # (W, H)


###############################################################################
# Dataset class
###############################################################################


class ADE20KDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        crop_size: Tuple[int, int] = _CROP_SIZE,
        ignore_index: int = 255,
    ) -> None:
        assert split in {"train", "val", "test", "validation", "training"}
        self.root = Path(root)
        # mapping for original ade folder names
        split_dir = {
            "train": "training",
            "training": "training",
            "val": "validation",
            "validation": "validation",
            "test": "validation",  # test images share same dir; list may differ
        }[split]

        img_dir = self.root / "images" / split_dir
        ann_dir = self.root / "annotations" / split_dir

        # If default ADE20K layout not found, support alternative common layout
        # where data is stored under `<root>/<train|val>/img` and
        # `<root>/<train|val>/annotations`. This allows training with
        # lightweight dataset extractions without reorganising files.
        if not img_dir.exists() or not ann_dir.exists():
            alt_root = self.root / split  # e.g. <root>/train or /val
            alt_img_dir = alt_root / "img"
            alt_ann_dir = alt_root / "annotations"

            if alt_img_dir.exists() and alt_ann_dir.exists():
                img_dir = alt_img_dir
                ann_dir = alt_ann_dir

        if not img_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(
                f"Dataset directories not found. Checked: \n"
                f" 1) {self.root}/images/{split_dir} and annotations/{split_dir} \n"
                f" 2) {self.root}/{split}/img and {self.root}/{split}/annotations"
            )

        self.images = sorted(img_dir.rglob("*.jpg"))
        self.masks = [ann_dir / (p.stem + ".png") for p in self.images]
        self.is_train = split in {"train", "training"}

        self.crop_size = crop_size
        global _CROP_SIZE  # override module-level size if user changes
        _CROP_SIZE = crop_size
        self.ignore_index = ignore_index

        # ------------------------------------------------------------------
        # Transformation mode: use manual pipeline to mimic mmcv RandomResize
        # ------------------------------------------------------------------
        self.transform = None  # albumentations pipeline disabled

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        mask_path = str(self.masks[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        # reduce_zero_label always (train/val/test)
        mask = mask.astype(np.uint8)
        mask[mask == 0] = self.ignore_index
        mask = mask - 1
        mask[mask == 254] = self.ignore_index  # newly wrapped values

        if self.is_train:
            img, mask = _train_transform(img, mask, self.crop_size, self.ignore_index)
        else:
            img, mask = _val_transform(img, mask, self.crop_size, self.ignore_index)

        # To tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, img_path


# -----------------------------------------------------------------------------
# Additional helper for val/test resize keep_ratio
# -----------------------------------------------------------------------------


def _resize(
    img: np.ndarray,
    mask: np.ndarray,
    scale: Tuple[int, int] = _IMG_SCALE,
    *,
    ratio_range: Tuple[float, float] | None = None,
    keep_ratio: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    base_w, base_h = scale

    if ratio_range is not None:
        ratio = random.uniform(*ratio_range)
        target_w = int(base_w * ratio)
        target_h = int(base_h * ratio)
    else:
        target_w, target_h = base_w, base_h

    if keep_ratio:
        # Compute new size preserving aspect ratio relative to *input* image.
        h, w = img.shape[:2]
        scale_factor = min(target_w / w, target_h / h)
        resize_w = int(w * scale_factor + 0.5)
        resize_h = int(h * scale_factor + 0.5)
    else:
        resize_w, resize_h = target_w, target_h

    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    return img, mask


# -----------------------------------------------------------------------------
# Manual augmentation helpers (mmseg style)
# -----------------------------------------------------------------------------


def _random_crop(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int],
    *,
    cat_max_ratio: float = 0.75,
    ignore_index: int = 255,
    num_attempts: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop with category ratio constraint (mimics mmseg RandomCrop).

    Ensures that no single class occupies more than ``cat_max_ratio`` of the
    cropped area. Falls back to the last sampled crop if the constraint cannot
    be satisfied after ``num_attempts`` trials.
    """

    ch, cw = crop_size[1], crop_size[0]
    h, w = mask.shape

    if h < ch or w < cw:
        img, mask = _pad_to_size(img, mask, crop_size, ignore_index)
        h, w = mask.shape  # 패딩 후 크기 갱신

    for attempt in range(num_attempts):
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img_crop = img[top : top + ch, left : left + cw]
        mask_crop = mask[top : top + ch, left : left + cw]

        valid = (mask_crop != ignore_index)
        if not np.any(valid):
            max_ratio = 0.0  # nothing but ignore pixels
        else:
            labels, counts = np.unique(mask_crop[valid], return_counts=True)
            max_ratio = counts.max() / counts.sum()  # ignore_index 완전히 제외

        if max_ratio < cat_max_ratio:
            return img_crop, mask_crop

    # fallback
    return img_crop, mask_crop


def _photo_metric_distortion(img: np.ndarray) -> np.ndarray:
    """Simplified PhotoMetricDistortion (brightness/contrast/hue/sat)."""

    img = img.astype(np.float32)

    # random brightness
    if random.random() < 0.5:
        delta = random.uniform(-32, 32)
        img += delta

    # mode: 0 → contrast 마지막, 1 → contrast 먼저 (mmseg 공식)
    mode = random.randint(0, 1)

    def _rand_contrast(inp: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            alpha = random.uniform(0.5, 1.5)
            return inp * alpha
        return inp

    if mode == 1:
        img = _rand_contrast(img)

    # convert to HSV for saturation & hue
    img_hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    # saturation
    if random.random() < 0.5:
        img_hsv[..., 1] *= random.uniform(0.5, 1.5)

    # hue shift
    if random.random() < 0.5:
        img_hsv[..., 0] += random.uniform(-18, 18)
        img_hsv[..., 0] = np.mod(img_hsv[..., 0], 180)

    img = cv2.cvtColor(np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    if mode == 0:
        img = _rand_contrast(img)

    return np.clip(img, 0, 255)


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    return img


def _pad_to_size(img: np.ndarray, mask: np.ndarray, crop_size: Tuple[int, int], ignore_index: int) -> Tuple[np.ndarray, np.ndarray]:
    ch, cw = crop_size[1], crop_size[0]
    h, w = mask.shape
    pad_h = max(ch - h, 0)
    pad_w = max(cw - w, 0)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=ignore_index)
    return img, mask


def _train_transform(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int],
    ignore_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # 1. Random scale w.r.t base scale 2048x512
    #    ratio: float ∈ [0.5, 2.0]
    #    img/mask dtype: uint8, value range 0–255 (unchanged)
    ratio = random.uniform(0.5, 2.0)
    new_w = int(_IMG_SCALE[0] * ratio)
    new_h = int(_IMG_SCALE[1] * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # ensure image >= crop_size
    if new_h < crop_size[1] or new_w < crop_size[0]:
        img, mask = _pad_to_size(img, mask, crop_size, ignore_index)
        new_h, new_w = img.shape[:2]

    # 2. Random crop with category max ratio (safe)
    img, mask = _random_crop(img, mask, crop_size, cat_max_ratio=0.75, ignore_index=ignore_index)
    # If random_crop returned same size because of constraints, ensure final size not smaller
    img, mask = _pad_to_size(img, mask, crop_size, ignore_index)

    # 3. Random horizontal flip
    #    probability: 0.5
    #    dtype/value range unchanged
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()

    # 4. Photo-metric distortion
    #    brightness delta: int ∈ [-32, 32]
    #    contrast alpha: float ∈ [0.5, 1.5]
    #    saturation factor: float ∈ [0.5, 1.5]
    #    hue shift: int ∈ [-18, 18] (HSV space)
    #    img converted to float32 during this step, value range 0–255 before clipping
    img = _photo_metric_distortion(img)

    # 5. Normalize
    #    mean: [123.675, 116.28, 103.53]
    #    std:  [58.395, 57.12, 57.375]
    #    output img dtype float32, roughly in range ~[-2, +2]
    img = _normalize(img)

    # 6. Pad to crop size (default 512×512)
    #    img pad value: 0 (float32 after norm), mask pad value: ignore_index (=255)
    img, mask = _pad_to_size(img, mask, crop_size, ignore_index)

    return img, mask


def _val_transform(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int],
    ignore_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    img, mask = _resize(img, mask)

    # normalize
    img = _normalize(img)

    # pad
    img, mask = _pad_to_size(img, mask, crop_size, ignore_index)

    return img, mask

 