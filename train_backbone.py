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

from models.dat_classifier import DatClassifier
from imagenet_dataloader import get_imagenet_dataloader, LabelSmoothingCrossEntropy


def build_dataloaders(args):
    """ImageNet 학습/검증 DataLoader 구성"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        normalize,
    ])

    val_trans = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = Path(args.data) / "train"
    val_dir = Path(args.data) / "val"

    train_set = datasets.ImageFolder(train_dir, transform=train_trans)
    val_set = datasets.ImageFolder(val_dir, transform=val_trans)

    if args.distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def accuracy(output, target, topk=(1,)):
    """top-k 정확도 산출"""
    with torch.no_grad():
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None, mixup_cutmix=None, log_interval=100):
    model.train()
    running_loss = 0.0
    running_acc1 = 0.0
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Apply Mixup/Cutmix after moving to device
        if mixup_cutmix is not None:
            imgs, labels = mixup_cutmix((imgs, labels))

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, _, _ = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _, _ = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc1, _ = accuracy(outputs, labels, topk=(1, 5))
        running_loss += loss.item()
        running_acc1 += acc1.item()

        if (i + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch}] Step {i + 1}/{len(loader)}  "
                f"Loss: {running_loss / (i + 1):.4f}  "
                f"Acc@1: {running_acc1 / (i + 1):.2f}%"
            )


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc1_sum = 0.0
    acc5_sum = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs, _, _ = model(imgs)
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
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--pretrained", default=None, help="Path to pretrained DAT checkpoint")
    parser.add_argument("--output", default="outputs", help="Checkpoint and log directory")
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

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    train_loader, val_loader, mixup_cutmix = get_imagenet_dataloader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed,
        use_mixup_cutmix=True,
    )

    train_sampler = train_loader.sampler if args.distributed else None

    best_acc1 = 0.0
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler=scaler, mixup_cutmix=mixup_cutmix
        )
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        if not args.distributed or torch.distributed.get_rank() == 0:
            print(
                f"[Epoch {epoch}] Val Loss: {val_loss:.4f}  Acc@1: {val_acc1:.2f}%  Acc@5: {val_acc5:.2f}%"
            )
            ckpt_path = Path(args.output) / f"epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler else None,
                    "acc1": val_acc1,
                },
                ckpt_path,
            )
            if val_acc1 > best_acc1:
                best_acc1 = val_acc1
                best_path = Path(args.output) / "best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"New best Acc@1: {best_acc1:.2f}%, checkpoint saved to {best_path}")


if __name__ == "__main__":
    main() 