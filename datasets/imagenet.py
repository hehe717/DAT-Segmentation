import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from timm.data import create_transform
from timm.data.mixup import Mixup

__all__ = ["get_imagenet_dataloader"]


def _get_train_transforms():
    return create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
    )


def _get_val_transforms():
    return create_transform(
        input_size=224,
        is_training=False,
        interpolation="bicubic",
    )


class _MixupAdapter:
    def __init__(
        self,
        num_classes: int = 1000,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        self._mixup = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=1.0,
            switch_prob=switch_prob,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )

    def __call__(self, batch):
        imgs, labels = batch
        return self._mixup(imgs, labels)


class ImageNetDataset(Dataset):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        with open(img_path, "rb") as fh:
            img = Image.open(fh).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_imagenet_dataloader(
    root_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    distributed: bool = False,
    use_mixup_cutmix: bool = False,
):
    train_ds = ImageNetDataset(root_dir, split="train", transform=_get_train_transforms())
    val_ds = ImageNetDataset(root_dir, split="val", transform=_get_val_transforms())

    train_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    mixup_cutmix = _MixupAdapter() if use_mixup_cutmix else None
    return train_loader, val_loader, mixup_cutmix 