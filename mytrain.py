#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from typing import Optional, Tuple
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import timm
from timm.data import create_transform
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD,
)
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
from datetime import datetime
import torch

import torch

def load_medclip_effb5_to_timm(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"]  # 你现在确认了就在这里

    # 只拿 image_encoder.*
    prefix = "image_encoder."
    enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    # 建一个 timm 的目标 state_dict
    tgt = model.state_dict()
    new = {}

    # -------- stem/head：固定映射 --------
    def assign(timm_k, src_k):
        if timm_k in tgt and src_k in enc and tgt[timm_k].shape == enc[src_k].shape:
            new[timm_k] = enc[src_k]

    assign("conv_stem.weight", "_conv_stem.weight")
    # timm 的 stem BN 叫 bn1.*，你的 ckpt 叫 _bn0.*
    for suf in ["weight","bias","running_mean","running_var"]:
        assign(f"bn1.{suf}", f"_bn0.{suf}")

    assign("conv_head.weight", "_conv_head.weight")
    # timm 的 head BN 叫 bn2.*，你的 ckpt 叫 _bn1.*
    for suf in ["weight","bias","running_mean","running_var"]:
        assign(f"bn2.{suf}", f"_bn1.{suf}")

    # classifier 通常不想载（任务类数不同），这里不 assign

    # -------- blocks：按顺序对齐 _blocks.N --------
    # 先把 timm blocks 展平成 (stage, idx) 顺序
    flat = []
    for s, stage in enumerate(model.blocks):
        for i, _ in enumerate(stage):
            flat.append((s, i))
    # ckpt 的 _blocks 是线性编号：_blocks.0, _blocks.1, ...
    # 我们假设两者数量相同或近似，按 min 对齐
    n = min(len(flat), max([int(k.split(".")[1]) for k in enc.keys() if k.startswith("_blocks.") and k.split(".")[1].isdigit()] + [-1]) + 1)

    def try_block_map(timm_prefix, src_prefix):
        # MBConv 常见字段映射：expand -> conv_pw；depthwise -> conv_dw；project -> conv_pwl
        # BN 对应：_bn0->_bn? 取决于实现，这里用 “优先匹配 shape” 的方式最稳
        candidates = [
            # conv
            ("conv_pw.weight",    f"{src_prefix}._expand_conv.weight"),
            ("conv_dw.weight",    f"{src_prefix}._depthwise_conv.weight"),
            ("conv_pwl.weight",   f"{src_prefix}._project_conv.weight"),
            # bn
            ("bn1.weight",        f"{src_prefix}._bn0.weight"),
            ("bn1.bias",          f"{src_prefix}._bn0.bias"),
            ("bn1.running_mean",  f"{src_prefix}._bn0.running_mean"),
            ("bn1.running_var",   f"{src_prefix}._bn0.running_var"),

            ("bn2.weight",        f"{src_prefix}._bn1.weight"),
            ("bn2.bias",          f"{src_prefix}._bn1.bias"),
            ("bn2.running_mean",  f"{src_prefix}._bn1.running_mean"),
            ("bn2.running_var",   f"{src_prefix}._bn1.running_var"),

            ("bn3.weight",        f"{src_prefix}._bn2.weight"),
            ("bn3.bias",          f"{src_prefix}._bn2.bias"),
            ("bn3.running_mean",  f"{src_prefix}._bn2.running_mean"),
            ("bn3.running_var",   f"{src_prefix}._bn2.running_var"),

            # se
            ("se.conv_reduce.weight", f"{src_prefix}._se_reduce.weight"),
            ("se.conv_reduce.bias",   f"{src_prefix}._se_reduce.bias"),
            ("se.conv_expand.weight", f"{src_prefix}._se_expand.weight"),
            ("se.conv_expand.bias",   f"{src_prefix}._se_expand.bias"),
        ]

        for dst_suf, src_k in candidates:
            dst_k = f"{timm_prefix}.{dst_suf}"
            if dst_k in tgt and src_k in enc and tgt[dst_k].shape == enc[src_k].shape:
                new[dst_k] = enc[src_k]

        # 有些 block 没有 expand_conv（例如 expansion=1），timm 的 conv_pw 可能对应不上
        # 再给一个 fallback：如果 conv_pw 没匹配到，尝试用 project_conv（shape 一致才会写入）
        dst_k = f"{timm_prefix}.conv_pw.weight"
        if dst_k in tgt and dst_k not in new:
            alt = f"{src_prefix}._project_conv.weight"
            if alt in enc and tgt[dst_k].shape == enc[alt].shape:
                new[dst_k] = enc[alt]

    for k in range(n):
        s, i = flat[k]
        timm_pref = f"blocks.{s}.{i}"
        src_pref  = f"_blocks.{k}"
        try_block_map(timm_pref, src_pref)

    msg = model.load_state_dict(new, strict=False)
    print("[load_medclip_effb5_to_timm] loaded:", len(new))
    print("[load_medclip_effb5_to_timm] missing:", len(msg.missing_keys))
    print("[load_medclip_effb5_to_timm] unexpected:", len(msg.unexpected_keys))
    # 如需看具体缺哪些：print(msg.missing_keys[:50])
    return msg

class TeeLogger:
    """Print to console and append to a txt log file (rank0 only)."""
    def __init__(self, log_path: str, enabled: bool = True):
        self.enabled = enabled
        self.log_path = log_path
        if self.enabled:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== Run start: {datetime.now().isoformat(timespec='seconds')} =====\n")

    def log(self, msg: str):
        # console
        print(msg, flush=True)
        # file
        if self.enabled:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


# ----------------------------
# DDP utils
# ----------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def ddp_cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

@torch.no_grad()
def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# ----------------------------
# Metrics
# ----------------------------
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)


@torch.no_grad()
def accuracy_top1(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = output.argmax(dim=1)
    correct = pred.eq(target).float().sum()
    return correct * (100.0 / target.size(0))



# ----------------------------
# Transform helpers (noise + insertion)
# ----------------------------
class AddGaussianNoise(nn.Module):
    """
    Add zero-mean Gaussian noise to a tensor image.
    Assumes input is a float tensor (C,H,W) after ToTensor(), before Normalize().
    """
    def __init__(self, std: float = 0.0, p: float = 0.7):
        super().__init__()
        self.std = float(std)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        if torch.rand(1).item() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise

def insert_before_normalize(transform: transforms.Compose, new_tf: nn.Module) -> transforms.Compose:
    """
    Insert `new_tf` before Normalize in a torchvision.transforms.Compose.
    If Normalize doesn't exist, append at end.
    """
    if not isinstance(transform, transforms.Compose):
        return transforms.Compose([transform, new_tf])

    tfs = list(transform.transforms)
    idx = None
    for i, t in enumerate(tfs):
        if isinstance(t, transforms.Normalize):
            idx = i
            break
    if idx is None:
        tfs.append(new_tf)
    else:
        tfs.insert(idx, new_tf)
    return transforms.Compose(tfs)


def build_transform(is_train: bool, args) -> transforms.Compose:
    """
    Mirror your provided logic:
    - train: timm create_transform with optional RandomCrop for small images
      + insert Gaussian noise only for training (before Normalize)
    - eval: Resize/CenterCrop (or warping for >=384) + ToTensor + Normalize
    """
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # If input_size <= 32 (CIFAR-style), replace first op with RandomCrop + padding
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)

        # add noise only for training
        if getattr(args, "noise_std", 0.0) > 0:
            transform = insert_before_normalize(
                transform,
                AddGaussianNoise(std=args.noise_std, p=getattr(args, "noise_p", 0.7))
            )
        return transform

    # Eval / test transform
    t = []
    if resize_im:
        if args.input_size >= 384:
            # warping (no cropping) for 384+
            t.append(
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
            if is_main_process():
                print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    logger: Optional[TeeLogger] = None,
    mixup_fn: Optional[Mixup] = None,
    log_interval: int = 50,
):
    model.train()
    loss_meter = AverageMeter()
    start = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        if logger is not None and is_main_process() and (step % log_interval == 0):
            elapsed = time.time() - start
            # print(f"[Train] Epoch {epoch} Step {step}/{len(loader)} Loss {loss_meter.avg:.4f} ({elapsed:.1f}s)")
            logger.log(f"[Train] Epoch {epoch} Step {step}/{len(loader)} Loss {loss_meter.avg:.4f}")


    loss_avg = reduce_mean(torch.tensor(loss_meter.avg, device=device)).item()
    return loss_avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1 = accuracy_top1(outputs, targets)

        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))

    loss_t = reduce_mean(torch.tensor(loss_meter.avg, device=device)).item()
    top1_t = reduce_mean(torch.tensor(top1_meter.avg, device=device)).item()
    return loss_t, top1_t


# ----------------------------
# Checkpoint
# ----------------------------
def save_ckpt(path: str, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best_acc: float):
    if not is_main_process():
        return
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_ckpt(path: str, model: nn.Module, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_acc = float(ckpt.get("best_acc", 0.0))
    return start_epoch, best_acc


# ----------------------------
# Data
# ----------------------------
def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, int]:
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise FileNotFoundError(f"Expected ImageFolder structure: {train_dir} and {val_dir}")

    train_tf = build_transform(True, args)
    val_tf = build_transform(False, args)

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)
    num_classes = len(train_set.classes)

    if is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_set, shuffle=True)
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
        drop_last=False,
    )
    return train_loader, val_loader, num_classes


def save_ckpt_limited(
    save_dir: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_acc: float,
    val_acc1: float,
    keep_last_k: int = 3,
    save_last: bool = True,
):
    """
    Save:
      - best.pth (only when improved)
      - epoch_{E}.pth for last K epochs only (older ones deleted)
      - optional last.pth (always overwritten)
    """
    if not is_main_process():
        return

    os.makedirs(save_dir, exist_ok=True)

    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }

    # (A) save last (optional)
    if save_last:
        torch.save(state, os.path.join(save_dir, "last.pth"))

    # (B) save rolling epoch ckpt
    epoch_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pth")
    torch.save(state, epoch_path)

    # delete ckpts older than last K epochs
    old_epoch = epoch - keep_last_k
    if old_epoch >= 0:
        old_path = os.path.join(save_dir, f"epoch_{old_epoch:04d}.pth")
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    # (C) save best only if improved
    # 注意：best_acc 是“历史 best”，val_acc1 是当前 epoch 的 top1
    if val_acc1 >= best_acc:  # >= ensures deterministic overwrite on ties
        torch.save(state, os.path.join(save_dir, "best.pth"))

import os
import torch

def _load_state_dict_any(ckpt_path: str, device="cpu"):
    ext = os.path.splitext(ckpt_path)[1].lower()

    if ext == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(ckpt_path, device=str(device))
        return state

    # .pth / .pt / .bin 等 torch.save 格式
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
        return ckpt
    return ckpt


def load_backbone_only(model, ckpt_path):
    import torch
    from safetensors.torch import load_file

    state = load_file(ckpt_path)  # 你如果原来不是这么读的，就用你原来的读法
    model_state = model.state_dict()

    filtered = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in state.items():
        if k not in model_state:
            skipped_missing.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    msg = model.load_state_dict(filtered, strict=False)
    print(f"[Pretrain] loaded={len(filtered)} | skipped_missing={len(skipped_missing)} | skipped_shape={len(skipped_shape)}")
    print("[Pretrain] example skipped_shape:", skipped_shape[:5])
    return msg



# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-txt", type=str, default="trainlog.txt", help="txt filename under save-dir")

    # data/model
    parser.add_argument("--data", type=str, required=True, help="dataset root containing train/ and val/")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="e.g. vit_base_patch16_224 or resnet50")
    parser.add_argument("--pretrained", type=str, default="", help="path to local pretrained weights (.pth/.bin/.safetensors)")

    # training
    # scheduler args (timm expects these names)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--cooldown-epochs', type=int, default=0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")

    # optimizer/scheduler
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--opt", type=str, default="adamw")
    parser.add_argument("--sched", type=str, default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # transforms args (match your build_transform signature)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--imagenet-default-mean-and-std", action="store_true")
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", help="auto augment policy string, or '' to disable")
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--reprob", type=float, default=0.25, help="random erasing prob")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--crop-pct", type=float, default=None, help="eval crop pct; default 224/256 if None")

    # noise (train only)
    parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise std (on tensor, before Normalize)")
    parser.add_argument("--noise_p", type=float, default=0.7, help="probability to apply Gaussian noise per sample")

    # mixup/cutmix/ls
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # ckpt/log
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=50)

    args = parser.parse_args()

    log_path = os.path.join(args.save_dir, args.log_txt)
    logger = TeeLogger(log_path, enabled=is_main_process())

    ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = build_dataloaders(args)

    model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
    model.to(device)
    print("[DEBUG] model type:", type(model))
    print("[DEBUG] embed_dim:", getattr(model, "embed_dim", None))
    print("[DEBUG] num_features:", getattr(model, "num_features", None))
    # Swin 特有：看第一层 downsample 前后的维度
    try:
        print("[DEBUG] stage0 dim:", model.layers[0].dim)
        print("[DEBUG] stage1 dim:", model.layers[1].dim)
        print("[DEBUG] stage2 dim:", model.layers[2].dim)
        print("[DEBUG] stage3 dim:", model.layers[3].dim)
    except Exception as e:
        print("[DEBUG] layer dims unavailable:", e)


    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"])
        )

    # Mixup/CutMix
    mixup_fn = None
    if args.mixup > 0.0 or args.cutmix > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )

    # Loss
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
    )
    
    from timm.scheduler import create_scheduler as timm_create_scheduler

    scheduler, _ = timm_create_scheduler(args, optimizer)


    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.pretrained:
        if args.model =='tf_efficientnet_b5':
            load_medclip_effb5_to_timm(model, args.pretrained)  # 用我前面给你的函数
        else:
            msg = load_backbone_only(model, args.pretrained)  # 或 .bin/.pth
            print(msg)  # 会显示 missing_keys 里包含 head，这是正常的

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        start_epoch, best_acc1 = load_ckpt(args.resume, model, optimizer, scheduler, scaler)
        if is_main_process():
            print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_acc1={best_acc1:.2f}")

    logger.log(f"Command: {' '.join(sys.argv)}")
    logger.log(f"Save dir: {args.save_dir}")
    logger.log(f"Model: {args.model}")
    logger.log(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, WD: {args.weight_decay}, AMP: {args.amp}")
    if getattr(args, "pretrained", ""):
        logger.log(f"Pretrained/ckpt: {args.pretrained}")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if is_dist_avail_and_initialized():
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler if args.amp else None,
            device=device,
            epoch=epoch,
            logger=logger,
            mixup_fn=mixup_fn,
            log_interval=args.log_interval,
        )

        if scheduler is not None:
            scheduler.step(epoch + 1)

        val_loss, val_acc1 = evaluate(model, val_loader, device)

        if is_main_process():
            # print(f"[Eval] Epoch {epoch} "
            #       f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            #       f"top1={val_acc1:.2f}")
            logger.log(f"[Eval] Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} top1={val_acc1:.2f}")

                  

        is_best = val_acc1 > best_acc1
        best_acc1 = max(best_acc1, val_acc1)

        # 更新 best_acc1（先算出当前是否 best）
        is_best = val_acc1 > best_acc1
        best_acc1 = max(best_acc1, val_acc1)

        # 只保存 best + 最近3个epoch (+ 可选 last)
        save_ckpt_limited(
            save_dir=args.save_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if args.amp else None,
            epoch=epoch,
            best_acc=best_acc1,
            val_acc1=val_acc1,
            keep_last_k=3,
            save_last=True,   # 如果你不想要 last.pth，改成 False
        )

    total_time = time.time() - start_time
    print("Training completed.")
    print(args.save_dir)
    print(f"Best Top-1 Accuracy: {best_acc1:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    if is_main_process():
        print(f"Total training time: {total_time_str}")
        logger.log(f"Total training time: {total_time_str}")
        logger.log(f"Best Top-1 Accuracy: {best_acc1:.2f}%")


    ddp_cleanup()


if __name__ == "__main__":
    main()

