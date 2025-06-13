import argparse
import importlib.util
import os
from pathlib import Path
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.builder import build_model_from_config
from datasets.datasets import ADE20KDataset


class WarmupPolyLRScheduler(optim.lr_scheduler._LRScheduler):
    """Poly LR with linear warm-up (iter-based) – mimics mmsegmentation."""

    def __init__(
        self,
        optimizer,
        max_iters: int,
        *,
        warmup_iters: int = 0,
        warmup_ratio: float = 1e-6,
        power: float = 0.9,
        min_lr: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warm-up
            alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
            factor = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        else:
            # Poly decay
            cur_iter = self.last_epoch - self.warmup_iters
            total_iter = self.max_iters - self.warmup_iters
            factor = (1 - cur_iter / float(total_iter)) ** self.power
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


def build_segmentation_model(backbone: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    """Wrap DAT backbone with a simple 1×1 conv head for semantic segmentation."""

    # Infer channel dimension with a dummy forward pass
    device = next(backbone.parameters()).device
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 512, 512, device=device)
        feats = backbone(dummy)
        in_channels = feats[-1].shape[1]

    head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    class _SegModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            feats = self.backbone(x)
            logits = self.head(feats[-1])  # use last stage feature
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
            return logits

    return _SegModel(backbone, head)


def parse_args():
    parser = argparse.ArgumentParser(description="DAT semantic segmentation training script")
    parser.add_argument("--config", default="configs/dat/upn_tiny_160k_dp03_lr6.py", help="Path to config file")
    parser.add_argument("--data_root", required=True, help="Path to ADE20K root directory (having images/, annotations/)")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=160000)
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=16000)
    parser.add_argument("--num_classes", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Model ----------------
    backbone = build_model_from_config(args.config)  # already on CUDA inside builder
    model = build_segmentation_model(backbone, args.num_classes).to(device)
    model.train()

    # ---------------- Data ----------------
    train_set = ADE20KDataset(args.data_root, split="train", crop_size=(512, 512), ignore_index=255)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------- Optim & LR ----------------
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ---------------- Parameter groups for AdamW (zero wd for norm & positional embeddings) ----------------
    no_decay_keys = ["absolute_pos_embed", "relative_position_bias_table", "rpe_table", "norm"]

    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay_keys) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay_keys) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    scheduler = WarmupPolyLRScheduler(
        optimizer,
        max_iters=args.max_iters,
        warmup_iters=1500,
        warmup_ratio=1e-6,
        power=1.0,
        min_lr=0.0,
    )

    # ---------------- Train loop ----------------
    data_iter = cycle(train_loader)
    iter_idx = 0

    while iter_idx < args.max_iters:
        imgs, masks, _ = next(data_iter)
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iter_idx + 1) % args.print_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iter [{iter_idx + 1}/{args.max_iters}]  Loss: {loss.item():.4f}  LR: {current_lr:.6f}")

        if (iter_idx + 1) % args.ckpt_interval == 0:
            ckpt_path = Path(args.save_dir) / f"iter_{iter_idx + 1}.pth"
            torch.save(
                {
                    "iter": iter_idx + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

        iter_idx += 1

    print("Training completed.")


if __name__ == "__main__":
    main() 