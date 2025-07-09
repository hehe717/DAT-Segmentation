#!/usr/bin/env python

import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard import SummaryWriter

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from models.dat_classifier import DatClassifier
from datasets.imagenet import get_imagenet_dataloader



def load_state(model, ckpt_path):
    if ckpt_path is None:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded. Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        if isinstance(target, tuple):
            target_a, target_b, lam = target
        else:
            target_a = target
            target_b = None
            lam = 1.0

        if target_a.dim() > 1:
            target_a = target_a.argmax(dim=1)
        if target_b is not None and target_b.dim() > 1:
            target_b = target_b.argmax(dim=1)

        maxk = max(topk)
        batch_size = target_a.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        res = []
        for k in topk:
            correct_a = pred[:k].eq(target_a.view(1, -1).expand_as(pred[:k]))

            if target_b is not None:
                correct_b = pred[:k].eq(target_b.view(1, -1).expand_as(pred[:k]))
                correct = lam * correct_a.float() + (1.0 - lam) * correct_b.float()
            else:
                correct = correct_a.float()

            correct_k = correct.reshape(-1).sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

def log_weight_stats(model, writer, step):
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

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, *, writer=None, global_step=0, mixup_cutmix=None, log_interval=100, scheduler=None, clip_grad=None):
    model.train()
    if dist.is_available() and dist.is_initialized():
        is_main_process = dist.get_rank() == 0
    else:
        is_main_process = True

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

        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        acc1, _ = accuracy(outputs, labels, topk=(1, 5))
        running_loss += loss.item()
        running_acc1 += acc1.item()

        global_step += 1
        if is_main_process and writer is not None and (i + 1) % log_interval == 0:
            log_weight_stats(model, writer, global_step)

        if is_main_process and (i + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch}] Step {i + 1}/{len(loader)}  "
                f"Loss: {running_loss / (i + 1):.4f}  "
                f"Acc@1: {running_acc1 / (i + 1):.2f}%"
            )

    epoch_loss = running_loss / len(loader)
    epoch_acc1 = running_acc1 / len(loader)

    if is_main_process and writer is not None:
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/acc1", epoch_acc1, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        log_weight_stats(model, writer, epoch)

    if is_main_process:
        print(
            f"[Epoch {epoch}] Train Loss: {epoch_loss:.4f}  Train Acc@1: {epoch_acc1:.2f}%"
        )

    return global_step, epoch_loss, epoch_acc1


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

    if dist.is_available() and dist.is_initialized():
        metrics = torch.tensor([loss_sum, acc1_sum, acc5_sum, float(n)], dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        loss_sum, acc1_sum, acc5_sum, n = metrics.tolist()

    loss_avg = loss_sum / n
    acc1_avg = acc1_sum / n
    acc5_avg = acc5_sum / n

    return loss_avg, acc1_avg, acc5_avg


def main():
    parser = argparse.ArgumentParser(description="DAT ImageNet Training")
    parser.add_argument("--data", required=True, help="ImageNet root directory")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=5e-4, type=float, help="Base learning rate")
    parser.add_argument("--warmup-epochs", default=20, type=int, help="Number of warm-up epochs")
    parser.add_argument("--warmup-lr", default=5e-7, type=float, help="Starting LR for warm-up")
    parser.add_argument("--min-lr", default=5e-6, type=float, help="Minimum LR after decay")
    parser.add_argument("--poly-power", default=0.9, type=float, help="Power for polynomial LR decay")
    parser.add_argument("--clip-grad", default=5.0, type=float, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--weight-decay", default=0.05, type=float)
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

    model = DatClassifier(num_classes=1000)
    if args.pretrained:
        load_state(model, args.pretrained)
    model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    train_loader, val_loader, mixup_cutmix = get_imagenet_dataloader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed,
        use_mixup_cutmix=True,
    )

    if mixup_cutmix is not None:
        criterion_train = SoftTargetCrossEntropy().to(device)
    else:
        criterion_train = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)

    criterion_val = torch.nn.CrossEntropyLoss().to(device)

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

    train_sampler = train_loader.sampler if args.distributed else None

    num_steps_per_epoch = len(train_loader)
    warmup_iters = args.warmup_epochs * num_steps_per_epoch
    total_iters = args.epochs * num_steps_per_epoch

    warmup_ratio = args.warmup_lr / args.lr
    min_factor = args.min_lr / args.lr

    def lr_lambda(current_iter: int):
        if current_iter < warmup_iters:
            return warmup_ratio + (1 - warmup_ratio) * current_iter / warmup_iters
        progress = (current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_factor, cosine_factor)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    is_main_process = (not args.distributed) or (torch.distributed.get_rank() == 0)

    writer = SummaryWriter(log_dir=args.output) if is_main_process else None

    best_acc1 = 0.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        global_step, epoch_loss, epoch_acc1 = train_one_epoch(
            model,
            train_loader,
            criterion_train,
            optimizer,
            device,
            epoch,
            writer=writer,
            global_step=global_step,
            mixup_cutmix=mixup_cutmix,
            scheduler=scheduler,
            clip_grad=args.clip_grad,
        )
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion_val, device)

        if writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc1", val_acc1, epoch)
            writer.add_scalar("val/acc5", val_acc5, epoch)
            log_weight_stats(model, writer, epoch)
            writer.flush()

        if is_main_process:
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
            backbone_state_dict = (
                model.module.backbone if isinstance(model, nn.parallel.DistributedDataParallel) else model.backbone
            ).state_dict()
            backbone_ckpt_path = Path(args.output) / f"epoch_{epoch}_backbone.pth"
            torch.save(backbone_state_dict, backbone_ckpt_path)
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            if is_main_process:
                best_path = Path(args.output) / "best.pth"
                torch.save(model.state_dict(), best_path)
                backbone_best_path = Path(args.output) / "best_backbone.pth"
                backbone_state_dict = (
                    model.module.backbone if isinstance(model, nn.parallel.DistributedDataParallel) else model.backbone
                ).state_dict()
                torch.save(backbone_state_dict, backbone_best_path)
                print(
                    f"New best Acc@1: {best_acc1:.2f}%, checkpoint saved to {best_path} (backbone at {backbone_best_path})"
                )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main() 