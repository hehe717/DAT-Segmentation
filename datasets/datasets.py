
from __future__ import annotations

import random
import math
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ADE20KDataset"]


IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_CROP_SIZE: Tuple[int, int] = (512, 512)
_IMG_SCALE: Tuple[int, int] = (2048, 512)  # (W, H)
_RATIO_RANGE: Tuple[float, float] = (0.5, 2.0)


###############################################################################
# Utility augmentation helpers
###############################################################################

def _random_resize(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly resize image & mask following mmseg's `Resize` operation."""
    ratio = random.uniform(*_RATIO_RANGE)
    new_w = int(_IMG_SCALE[0] * ratio)
    new_h = int(_IMG_SCALE[1] * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized


def _random_crop(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop with cat_max_ratio=0.75 (at most 75% of any single class).

    If the constraint cannot be satisfied in 10 attempts we fall back to a
    simple centre crop.
    """
    # Always produce output of size _CROP_SIZE, following mmseg RandomCrop.
    ch, cw = _CROP_SIZE

    # Step 1: Pad if image is smaller than crop size (same logic as mmseg's
    #         pad_if_needed inside RandomCrop).
    h, w = img.shape[:2]
    pad_h = max(ch - h, 0)
    pad_w = max(cw - w, 0)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        h, w = img.shape[:2]  # update sizes after padding

    # Step 2: Try 10 times to satisfy cat_max_ratio=0.75
    for _ in range(10):
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        crop_mask = mask[top : top + ch, left : left + cw]

        labels, counts = np.unique(crop_mask, return_counts=True)
        if counts.size == 0:
            break  # empty? unlikely, but fallback to center crop
        if counts.max() / counts.sum() <= 0.75:
            return img[top : top + ch, left : left + cw], crop_mask

    # Step 3: fallback centre crop
    center_top = (h - ch) // 2
    center_left = (w - cw) // 2
    return (
        img[center_top : center_top + ch, center_left : center_left + cw],
        mask[center_top : center_top + ch, center_left : center_left + cw],
    )


def _random_flip(img: np.ndarray, mask: np.ndarray, p: float = 0.5):
    if random.random() < p:
        img = np.ascontiguousarray(img[:, ::-1, :])
        mask = np.ascontiguousarray(mask[:, ::-1])
    return img, mask


def _photometric_distortion(img: np.ndarray) -> np.ndarray:
    """Implement mmseg PhotoMetricDistortion."""
    img = img.astype(np.float32)

    # Random brightness
    if random.random() < 0.5:
        delta = random.uniform(-32, 32)
        img += delta

    # Contrast or not first (50% chance)
    if random.random() < 0.5:
        alpha = random.uniform(0.5, 1.5)
        img *= alpha

    # Saturation
    if random.random() < 0.5:
        img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        img_hsv[:, :, 1] *= random.uniform(0.5, 1.5)
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Hue
    if random.random() < 0.5:
        img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        img_hsv[:, :, 0] += random.uniform(-18, 18)
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Contrast again if not yet applied
    if random.random() < 0.5:
        alpha = random.uniform(0.5, 1.5)
        img *= alpha

    return img


def _normalize(img: np.ndarray) -> np.ndarray:
    return (img - IMG_NORM_MEAN) / IMG_NORM_STD


def _pad(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ch, cw = _CROP_SIZE
    h, w = img.shape[:2]
    pad_h = max(ch - h, 0)
    pad_w = max(cw - w, 0)
    if pad_h == 0 and pad_w == 0:
        return img, mask

    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
    return img, mask

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
            # --- training augmentation pipeline ---
            img, mask = _random_resize(img, mask)
            img, mask = _random_crop(img, mask)
            img, mask = _random_flip(img, mask)
            img = _photometric_distortion(img)
            img = _normalize(img)
            img, mask = _pad(img, mask)

        # To tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, img_path


# -----------------------------------------------------------------------------
# Additional helper for val/test resize keep_ratio
# -----------------------------------------------------------------------------


def _resize_keep_ratio(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize so that output height==512, width scaled <=2048 keeping aspect."""
    h, w = img.shape[:2]
    scale = _IMG_SCALE[1] / h  # target_h / h (target_h=512)
    new_h = _IMG_SCALE[1]
    new_w = int(w * scale)
    # ensure width does not exceed 2048, otherwise scale down further
    if new_w > _IMG_SCALE[0]:
        scale = _IMG_SCALE[0] / w
        new_w = _IMG_SCALE[0]
        new_h = int(h * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img, mask 