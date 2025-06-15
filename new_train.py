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
from models.backbones.loading import load_checkpoint
import numpy as np


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
    parser.add_argument("--eval_interval", type=int, default=16000, help="Interval (iters) to run validation evaluation")
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--pretrained", default="pretrained/upn_dat_t_160k.pth", help="Path to a pretrained checkpoint (.pth) to load before training")
    parser.add_argument("--aux_loss_weight", type=float, default=0.4, help="Weight for auxiliary loss (set 0 to disable)")
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

    # Disable gradient checkpointing (DAT backbone uses torch.utils.checkpoint)
    # when combined with DDP(find_unused_parameters=True) it can trigger
    # "Expected to mark a variable ready only once" errors.
    def _recursively_disable_checkpoint(module):
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False
        for child in module.children():
            _recursively_disable_checkpoint(child)

    _recursively_disable_checkpoint(model)

    # Optionally load a full-model checkpoint (strict=False: 구조 변동 허용)
    if args.pretrained and os.path.isfile(args.pretrained):
        try:
            load_checkpoint(model, args.pretrained, map_location=device, strict=False)
            if local_rank == 0:
                print(f"[Info] Loaded pretrained weights from {args.pretrained}")
        except Exception as e:
            if local_rank == 0:
                print(f"[Warning] Failed to load pretrained weights: {e}")

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

    # Validation set & loader
    val_set = ADE20KDataset(args.data_root, split="val", crop_size=(512, 512), ignore_index=255)
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
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

    # ---------------- Metric accumulators ----------------
    train_area_inter = np.zeros(args.num_classes, dtype=np.int64)
    train_area_union = np.zeros(args.num_classes, dtype=np.int64)

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
        aux_logits = None
        if isinstance(logits, tuple):
            logits, aux_logits = logits  # type: ignore

        loss = criterion(logits, masks)

        # Auxiliary loss (if provided by model output and weight > 0)
        if args.aux_loss_weight > 0 and aux_logits is not None:
            aux_loss = criterion(aux_logits, masks)
            loss = loss + args.aux_loss_weight * aux_loss

        # Metric accumulation BEFORE gradients so it doesn't interfere with DDP sync
        inter, union = _intersection_and_union(logits, masks, args.num_classes, ignore_index=255)
        train_area_inter += inter
        train_area_union += union

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iter_idx + 1) % args.print_interval == 0 and (not distributed or dist.get_rank() == 0):
            current_lr = scheduler.get_last_lr()[0]
            train_miou = _compute_miou(train_area_inter, train_area_union)
            print(
                f"Iter [{iter_idx + 1}/{args.max_iters}]  Loss: {loss.item():.4f}  LR: {current_lr:.6f}  Train mIoU: {train_miou:.4f}"
            )
            # reset accumulators
            train_area_inter.fill(0)
            train_area_union.fill(0)

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

        # ---------------- Validation evaluation ----------------
        if (iter_idx + 1) % args.eval_interval == 0 and (not distributed or dist.get_rank() == 0):
            val_miou = _evaluate(model if not distributed else model.module, val_loader, device, args.num_classes)
            print(f"[Val] Iter {iter_idx + 1}: mIoU {val_miou:.4f}")

        iter_idx += 1

    if not distributed or dist.get_rank() == 0:
        print("Training completed.")

    if distributed:
        dist.destroy_process_group()


###############################################################################
# Metric helpers
###############################################################################


def _intersection_and_union(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute intersection and union CPU numpy arrays for a batch."""
    with torch.no_grad():
        if logits.dim() == 4:  # (B, C, H, W)
            preds = logits.argmax(1)
        else:
            preds = logits

        preds = preds.view(-1)
        masks = masks.view(-1)

        valid = masks != ignore_index
        preds = preds[valid]
        masks = masks[valid]

        intersection = preds[preds == masks]
        area_inter = torch.bincount(intersection, minlength=num_classes).cpu().numpy()
        area_pred = torch.bincount(preds, minlength=num_classes).cpu().numpy()
        area_mask = torch.bincount(masks, minlength=num_classes).cpu().numpy()
        area_union = area_pred + area_mask - area_inter
        return area_inter, area_union


def _compute_miou(area_inter: np.ndarray, area_union: np.ndarray) -> float:
    iou = area_inter / np.maximum(area_union, 1)
    valid = area_union > 0
    if valid.sum() == 0:
        return 0.0
    return float(iou[valid].mean())


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
):
    model.eval()
    area_inter = np.zeros(num_classes, dtype=np.int64)
    area_union = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            inter, union = _intersection_and_union(logits, masks, num_classes, ignore_index)
            area_inter += inter
            area_union += union
    miou = _compute_miou(area_inter, area_union)
    model.train()
    return miou


if __name__ == "__main__":
    main() 