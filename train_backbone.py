#!/usr/bin/env python
"""train_backbone.py
DAT 백본과 ClsHead 를 결합하여 ImageNet (ILSVRC-2012) 분류 학습을 수행하는 스크립트.
사용 예)
    python train_backbone.py --data /path/to/imagenet --batch-size 256 --epochs 300
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard import SummaryWriter

from models.dat_classifier import DatClassifier
from datasets.imagenet import get_imagenet_dataloader

class LabelSmoothingCrossEntropy(torch.nn.Module):
    """Cross-entropy with label smoothing and Mixup/Cutmix tuple support."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target):
        if isinstance(target, tuple):
            target_a, target_b, lam = target
            return lam * self._smooth_loss(pred, target_a) + (1 - lam) * self._smooth_loss(pred, target_b)
        return self._smooth_loss(pred, target)

    def _smooth_loss(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: WPS110
        num_classes = pred.size(-1)
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        smooth_target = torch.zeros_like(log_prob).fill_(self.smoothing / (num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-smooth_target * log_prob, dim=-1))

def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy"""
    with torch.no_grad():
        target = target[0]
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_weight_stats(model, writer, step):
    """Record weight statistics and update ratio to TensorBoard."""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        data = param.data
        writer.add_scalar(f"weights/{name}/mean", data.mean().item(), step)
        writer.add_scalar(f"weights/{name}/std", data.std().item(), step)
        writer.add_scalar(f"weights/{name}/L2_norm", data.norm().item(), step)
        if param.grad is not None:
            update_ratio = (param.grad.norm() / (data.norm() + 1e-8)).item()
            writer.add_scalar(f"grads/{name}/update_ratio", update_ratio, step)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, *, writer=None, global_step=0, mixup_cutmix=None, log_interval=100, scheduler=None):
    model.train()
    running_loss = 0.0
    running_acc1 = 0.0
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if mixup_cutmix is not None:
            imgs, labels = mixup_cutmix((imgs, labels))

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        acc1, _ = accuracy(outputs, labels, topk=(1, 5))
        running_loss += loss.item()
        running_acc1 += acc1.item()

        global_step += 1
        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc1", acc1.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if (i + 1) % log_interval == 0:
                log_weight_stats(model, writer, global_step)

        if (i + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch}] Step {i + 1}/{len(loader)}  "
                f"Loss: {running_loss / (i + 1):.4f}  "
                f"Acc@1: {running_acc1 / (i + 1):.2f}%"
            )

    return global_step


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc1_sum = 0.0
    acc5_sum = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            loss_sum += loss.item()
            acc1_sum += acc1.item()
            acc5_sum += acc5.item()
    n = len(loader)
    return loss_sum / n, acc1_sum / n, acc5_sum / n


def main():
    parser = argparse.ArgumentParser(description="DAT ImageNet Training")
    parser.add_argument("--data", required=True, help="ImageNet root directory")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.00006, type=float)
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--pretrained", default=None, help="Path to pretrained DAT checkpoint")
    parser.add_argument("--output", default="./logs", help="Checkpoint and log directory")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.distributed:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DatClassifier(num_classes=1000, pretrained=args.pretrained)
    model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)

    no_decay_keywords = [
        "absolute_pos_embed",
        "relative_position_bias_table",
        "rpe_table",
        "norm",
    ]

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": args.weight_decay},
    ]

    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    train_loader, val_loader, mixup_cutmix = get_imagenet_dataloader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed,
        use_mixup_cutmix=True,
    )

    train_sampler = train_loader.sampler if args.distributed else None

    total_iters = args.epochs * len(train_loader)

    # Poly learning rate schedule with linear warm-up (by iteration)
    warmup_iters = 1500
    warmup_ratio = 1e-6
    power = 1.0

    def lr_lambda(current_iter: int):  # noqa: WPS430
        if current_iter < warmup_iters:
            return warmup_ratio + (1 - warmup_ratio) * current_iter / warmup_iters
        else:
            progress = (current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
            return (1 - progress) ** power

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # TensorBoard writer
    if not args.distributed or torch.distributed.get_rank() == 0:
        writer = SummaryWriter(log_dir=args.output)
    else:
        writer = None

    best_acc1 = 0.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer=writer,
            global_step=global_step,
            mixup_cutmix=mixup_cutmix,
            scheduler=scheduler,
        )
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)

        if writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc1", val_acc1, epoch)
            writer.add_scalar("val/acc5", val_acc5, epoch)
            log_weight_stats(model, writer, epoch)
            writer.flush()

        if not args.distributed or torch.distributed.get_rank() == 0:
            print(
                f"[Epoch {epoch}] Val Loss: {val_loss:.4f}  Acc@1: {val_acc1:.2f}%  Acc@5: {val_acc5:.2f}%"
            )

        if epoch % 10 == 0:
            ckpt_path = Path(args.output) / f"epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "acc1": val_acc1,
                },
                ckpt_path,
            )
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_path = Path(args.output) / "best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best Acc@1: {best_acc1:.2f}%, checkpoint saved to {best_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main() 