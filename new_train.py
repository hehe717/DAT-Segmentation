import argparse
import importlib.util
import os
from pathlib import Path
# from itertools import cycle  # 삭제: DDP 환경에선 사용하지 않음

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from models.builder import build_model_from_config
from datasets.datasets import ADE20KDataset
import torch.distributed as dist


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
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- Distributed init ----------------
    distributed = False
    if ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1) or args.local_rank != -1:
        distributed = True

    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # ---------------- Model ----------------
    model = build_model_from_config(args.config).to(device)

    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    model.train()

    # ---------------- Data ----------------
    train_set = ADE20KDataset(args.data_root, split="train", crop_size=(512, 512), ignore_index=255)
    sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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

    # ---------------- Iteration helpers ----------------
    sampler_epoch = 0
    if sampler is not None:
        sampler.set_epoch(sampler_epoch)
    train_loader_iter = iter(train_loader)
    iter_idx = 0

    while iter_idx < args.max_iters:
        try:
            imgs, masks, _ = next(train_loader_iter)
        except StopIteration:
            sampler_epoch += 1
            if sampler is not None:
                sampler.set_epoch(sampler_epoch)
            train_loader_iter = iter(train_loader)
            imgs, masks, _ = next(train_loader_iter)

        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iter_idx + 1) % args.print_interval == 0 and (not distributed or dist.get_rank() == 0):
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iter [{iter_idx + 1}/{args.max_iters}]  Loss: {loss.item():.4f}  LR: {current_lr:.6f}")

        if (iter_idx + 1) % args.ckpt_interval == 0 and (not distributed or dist.get_rank() == 0):
            ckpt_path = Path(args.save_dir) / f"iter_{iter_idx + 1}.pth"
            torch.save(
                {
                    "iter": iter_idx + 1,
                    "model_state": model.module.state_dict() if distributed else model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

        iter_idx += 1

    if not distributed or dist.get_rank() == 0:
        print("Training completed.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 