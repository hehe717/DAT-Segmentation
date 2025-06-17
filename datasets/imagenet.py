import os
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

__all__ = [
    "get_imagenet_dataloader",
    "MixupCutmix",
]

# --------------------------------------------------
# Augmentations (RandAugment + RandomErasing etc.)
# --------------------------------------------------

def _get_train_transforms() -> transforms.Compose:  # noqa: WPS430
    """Return the default training transforms following NVLabs recipe."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
            transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# --------------------------------------------------
# Mixup / Cutmix
# --------------------------------------------------

class Mixup:  # noqa: WPS110, WPS600
    """Implement Mixup augmentation."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, (labels_a, labels_b, lam)


class Cutmix:  # noqa: WPS110, WPS600
    """Implement Cutmix augmentation."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size, _, h, w = images.size()

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        index = torch.randperm(batch_size, device=images.device)

        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        return images, (labels, labels[index], lam)


class MixupCutmix:  # noqa: WPS110
    """Randomly apply Mixup or Cutmix based on switch_prob."""

    def __init__(self, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0, switch_prob: float = 0.5):
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.switch_prob = switch_prob

    def __call__(self, batch):
        if np.random.rand() < self.switch_prob:
            return self.mixup(batch)
        return self.cutmix(batch)


# --------------------------------------------------
# ImageNet dataset (folder structure)
# --------------------------------------------------

class ImageNetDataset(Dataset):
    """A simple folder-based ImageNet dataset loader compatible with custom augments."""

    def __init__(self, root_dir: str | Path, split: str = "train", transform: Optional[transforms.Compose] = None):
        self.data_dir = Path(root_dir) / split
        self.transform = transform

        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples: list[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:  # noqa: D401, WPS110
        return len(self.samples)

    def __getitem__(self, idx: int):  # noqa: WPS110
        img_path, label = self.samples[idx]
        with open(img_path, "rb") as fh:
            img = Image.open(fh).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# --------------------------------------------------
# Public helper
# --------------------------------------------------

def get_imagenet_dataloader(
    root_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    distributed: bool = False,
    use_mixup_cutmix: bool = False,
):
    """Create train/val dataloaders for ImageNet with optional Mixup/Cutmix."""
    train_ds = ImageNetDataset(root_dir, split="train", transform=_get_train_transforms())
    val_ds = ImageNetDataset(
        root_dir,
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    train_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    mixup_cutmix = MixupCutmix() if use_mixup_cutmix else None
    return train_loader, val_loader, mixup_cutmix 