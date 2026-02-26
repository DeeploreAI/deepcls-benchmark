#!/usr/bin/env python3
"""Train ResNet on an ImageFolder dataset with training recipe.

Configuration for training custom datasets from scratch:
- Standard ResNet architecture (ResNet50)
- Basic data augmentation (RandomResizedCrop, HFlip, ColorJitter)
- Conservative regularization (weight decay only)
- No Mixup/Cutmix, ModelEMA, or aggressive augmentation

This script uses a recipe suitable for from-scratch training on custom datasets.
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

import timm
from accelerate import Accelerator
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import ModelEmaV2, accuracy

# Model base configuration
MODEL_CONFIG = {
    "model": "convnext_tiny",
    "input_size": 224,
    "batch_size_per_device": 128,
    "epochs": 100,
    "seed": 666,
    "num_workers": 12,
    "pin_mem": True,
    "use_prefetcher": False,  # PrefetchLoader not available in this timm version
    "model_ema": True,  # disabled
    "model_ema_eval": True,  # disabled
    "model_ema_decay": 0.9999,  # disabled
}

# Optimizer configuration
OPTIMIZER_CONFIG = {
    "opt": "adamw",
    "lr": 1e-3,  # 4e-3 * total_batch_size / 1024
    "weight_decay": 0.05,
    "opt_eps": 1e-8,
    "opt_betas": (0.9, 0.999),
    "filter_bias_and_bn": True,  # remove bias and bn from weight decay
}

# Learning rate scheduler configuration
SCHEDULER_CONFIG = {
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "warmup_epochs": 10,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    # crop then resize (random resized crop), could crop at any position.
    # the crop size is based on the target area and ratio.
    "scale": (0.4, 1.0),  # crop ratio of the origin image
    "ratio": (3.0 / 4.0, 4.0 / 3.0),  # width-height ratio fron 3:4 to 4:3
    "hflip": 0.5,  # 50% probability for a horizontal flip
    "vflip": 0.0,  # disabled
    "color_jitter": 0.4,  # brightness, contrast, saturation, hue, jitter from 60% to 140%
    "interpolation": "bicubic",
    "reprob": 0.2,  # random erasing probability
    "remode": "pixel",  # random erasing mode, pixel -> noise, const -> pure color
    "recount": 1,  # random erasing count
    "crop_pct": 224 / 256,  # eval-only, resize short to 256, center crop to 224
}

# Regularization configuration
REGULARIZATION_CONFIG = {
    "drop_path": 0.2,  # stochastic depth, skip some blocks
    "smoothing": 0.1,  # smooth label from [0, 1, 0] to [0.05, 0.9, 0.05]
    "mixup": 0.0,  # disabled
    "cutmix": 0.0,  # disabled
    "mixup_prob": 0.0,  # disabled
    "mixup_switch_prob": 0.5,  # disabled
    "mixup_mode": "batch",  # disabled
}

# Merge all configurations
TRAIN_CONFIG = {
    **MODEL_CONFIG,
    **OPTIMIZER_CONFIG,
    **SCHEDULER_CONFIG,
    **AUGMENTATION_CONFIG,
    **REGULARIZATION_CONFIG,
}


class RemappedValDataset(torch.utils.data.Dataset):
    """Validation dataset that reuses train class_to_idx mapping.

    This avoids failures when val misses some classes that exist in train.
    """

    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.imgs = samples
        self.targets = [s[1] for s in samples]
        self.class_to_idx = class_to_idx
        self.classes = [None] * len(class_to_idx)
        for name, idx in class_to_idx.items():
            self.classes[idx] = name
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt trainer (Accelerate)")
    parser.add_argument("--data-root", type=Path, default=Path("/data/Datasets/DeepDet"))
    parser.add_argument("--expt-name", type=str, default="convnext_small_sp", help="Experiment name for output directory")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--batch-size-per-device",
        type=int,
        default=TRAIN_CONFIG["batch_size_per_device"],
        help="Per-device batch size. Effective global batch is batch_size_per_device * world_size.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Optional override for classifier output dimension. Default: infer from ImageFolder.",
    )
    parser.add_argument("--workers", type=int, default=TRAIN_CONFIG["num_workers"])
    parser.add_argument("--val-interval", type=int, default=5, help="Run validation every N epochs.")
    parser.add_argument("--print-freq", type=int, default=50)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(data_root: Path):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected ImageFolder dirs: {train_dir} and {val_dir}")

    train_tf = create_transform(
        input_size=(3, TRAIN_CONFIG["input_size"], TRAIN_CONFIG["input_size"]),
        is_training=True,
        use_prefetcher=TRAIN_CONFIG["use_prefetcher"],
        no_aug=False,
        scale=TRAIN_CONFIG["scale"],
        ratio=TRAIN_CONFIG["ratio"],
        hflip=TRAIN_CONFIG["hflip"],
        vflip=TRAIN_CONFIG["vflip"],
        color_jitter=TRAIN_CONFIG["color_jitter"],
        auto_augment=None,
        interpolation=TRAIN_CONFIG["interpolation"],
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=TRAIN_CONFIG["reprob"],
        re_mode=TRAIN_CONFIG["remode"],
        re_count=TRAIN_CONFIG["recount"],
    )

    val_tf = create_transform(
        input_size=(3, TRAIN_CONFIG["input_size"], TRAIN_CONFIG["input_size"]),
        is_training=False,
        interpolation=TRAIN_CONFIG["interpolation"],
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        crop_pct=TRAIN_CONFIG["crop_pct"],
    )

    train_set = ImageFolder(train_dir, transform=train_tf)
    val_class_dirs = {p.name for p in val_dir.iterdir() if p.is_dir()}

    # Check if there is extra class in val set than train set
    extra_val_classes = sorted(val_class_dirs - set(train_set.class_to_idx.keys()))
    if extra_val_classes:
        preview = ", ".join(extra_val_classes[:20])
        suffix = " ..." if len(extra_val_classes) > 20 else ""
        raise RuntimeError(f"val has classes not in train: {preview}{suffix}")

    # Check if there are classes in train set but missing in val set
    missing_val_classes = sorted(set(train_set.class_to_idx.keys()) - val_class_dirs)
    if missing_val_classes:
        preview = ", ".join(missing_val_classes[:20])
        suffix = " ..." if len(missing_val_classes) > 20 else ""
        print(f"Warning: train has classes not in val: {preview}{suffix}")

    val_samples = []
    valid_exts = {ext.lower() for ext in IMG_EXTENSIONS}
    for cls_name, cls_idx in train_set.class_to_idx.items():
        cls_dir = val_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for root, _, files in os.walk(cls_dir):
            for fn in files:
                if Path(fn).suffix.lower() in valid_exts:
                    val_samples.append((str(Path(root) / fn), cls_idx))

    if not val_samples:
        raise RuntimeError(f"No validation images found under {val_dir}")

    val_set = RemappedValDataset(
        samples=val_samples,
        class_to_idx=train_set.class_to_idx,
        transform=val_tf,
    )

    return train_set, val_set


def build_loaders(train_set, val_set, workers: int, batch_size_per_device: int):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=workers,
        pin_memory=TRAIN_CONFIG["pin_mem"],
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=workers,
        pin_memory=TRAIN_CONFIG["pin_mem"],
        drop_last=False,
    )

    return train_loader, val_loader


def build_model(num_classes: int):
    model = timm.create_model(
        TRAIN_CONFIG["model"],
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=TRAIN_CONFIG["drop_path"],
    )
    return model


def build_optimizer(model: nn.Module):
    return create_optimizer_v2(
        model,
        opt=TRAIN_CONFIG["opt"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        eps=TRAIN_CONFIG["opt_eps"],
        betas=TRAIN_CONFIG["opt_betas"],
        filter_bias_and_bn=TRAIN_CONFIG["filter_bias_and_bn"],
    )


def build_scheduler(optimizer, updates_per_epoch: int):
    return CosineLRScheduler(
        optimizer,
        t_initial=TRAIN_CONFIG["epochs"] * updates_per_epoch,
        lr_min=TRAIN_CONFIG["min_lr"],
        warmup_t=TRAIN_CONFIG["warmup_epochs"] * updates_per_epoch,
        warmup_lr_init=TRAIN_CONFIG["warmup_lr"],
        warmup_prefix=False,  # Changed from True to False
        cycle_limit=1,
        t_in_epochs=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    num_classes: int,
):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    topk = (1, 3) if num_classes >= 3 else (1,)

    total_loss = 0.0
    total_acc1 = 0.0
    total_acc3 = 0.0
    total_samples = 0

    # Per-class accuracy tracking
    class_correct_top1 = torch.zeros(num_classes, dtype=torch.float32, device=accelerator.device)
    class_correct_top3 = torch.zeros(num_classes, dtype=torch.float32, device=accelerator.device)
    class_total = torch.zeros(num_classes, dtype=torch.float32, device=accelerator.device)

    for images, target in loader:
        with accelerator.autocast():
            output = model(images)
            loss = criterion(output, target)

        # Gather predictions and targets from all processes
        output, target = accelerator.gather_for_metrics((output, target))

        if len(topk) == 2:
            acc1, acc3 = accuracy(output, target, topk=topk)
            total_acc1 += acc1.item() * target.size(0)
            total_acc3 += acc3.item() * target.size(0)
        else:
            (acc1,) = accuracy(output, target, topk=topk)
            total_acc1 += acc1.item() * target.size(0)
            total_acc3 += acc1.item() * target.size(0)

        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)

        # Calculate per-class accuracy (top-1 and top-3) - vectorized
        pred_top1 = output.argmax(dim=1)
        if len(topk) == 2:
            _, pred_top3 = output.topk(3, dim=1, largest=True, sorted=True)
        else:
            pred_top3 = pred_top1.unsqueeze(1)

        # Vectorized per-class accuracy calculation
        for cls_idx in range(num_classes):
            cls_mask = target == cls_idx
            if cls_mask.sum() == 0:
                continue
            class_total[cls_idx] += cls_mask.sum()
            # Top-1 per-class accuracy
            class_correct_top1[cls_idx] += (pred_top1[cls_mask] == cls_idx).sum()
            # Top-3 per-class accuracy
            if len(topk) == 2:
                class_correct_top3[cls_idx] += (pred_top3[cls_mask] == cls_idx).any(dim=1).sum()
            else:
                class_correct_top3[cls_idx] += (pred_top1[cls_mask] == cls_idx).sum()

    # Avoid division by zero
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples
    avg_acc1 = total_acc1 / total_samples
    avg_acc3 = total_acc3 / total_samples

    # Calculate average per-class accuracy (top-1 and top-3)
    valid_classes = class_total > 0
    if valid_classes.sum() > 0:
        class_acc1 = class_correct_top1[valid_classes] / class_total[valid_classes]
        class_acc3 = class_correct_top3[valid_classes] / class_total[valid_classes]
        per_cls_acc1 = (class_acc1.mean() * 100).item()
        per_cls_acc3 = (class_acc3.mean() * 100).item()
    else:
        per_cls_acc1 = 0.0
        per_cls_acc3 = 0.0

    return avg_loss, avg_acc1, avg_acc3, per_cls_acc1, per_cls_acc3


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    model_ema: Optional[ModelEmaV2],
    optimizer,
    scheduler,
    best_acc1: float,
    class_to_idx: dict,
    is_best: bool,
):
    if not accelerator.is_main_process:
        return

    # Create checkpoints subdirectory
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model if wrapped by Accelerate
    unwrapped_model = accelerator.unwrap_model(model)

    state = {
        "epoch": epoch,
        "model": unwrapped_model.state_dict(),
        "model_ema": model_ema.module.state_dict() if model_ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc1": best_acc1,
        "class_to_idx": class_to_idx,
        "official_recipe": TRAIN_CONFIG,
    }
    ckpt_path = ckpt_dir / "last.pth"
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, ckpt_dir / "best.pth")
        torch.save(state, ckpt_dir / f"best_epoch_{epoch + 1:03d}.pth")


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    model_ema: Optional[ModelEmaV2],
    optimizer,
    scheduler,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if model_ema is not None and ckpt.get("model_ema") is not None:
        model_ema.module.load_state_dict(ckpt["model_ema"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_acc1 = float(ckpt.get("best_acc1", 0.0))
    return start_epoch, best_acc1, ckpt


def init_epoch_log(log_path: Path) -> None:
    if log_path.exists():
        return
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc1",
                "val_acc3",
                "val_cls_acc1",
                "val_cls_acc3",
                "best_acc1",
                "validated",
            ]
        )


def append_epoch_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    validated: bool,
    best_acc1: float,
    val_loss: Optional[float] = None,
    val_acc1: Optional[float] = None,
    val_acc3: Optional[float] = None,
    val_cls_acc1: Optional[float] = None,
    val_cls_acc3: Optional[float] = None,
) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_acc1 is None else f"{val_acc1:.4f}",
                "" if val_acc3 is None else f"{val_acc3:.4f}",
                "" if val_cls_acc1 is None else f"{val_cls_acc1:.4f}",
                "" if val_cls_acc3 is None else f"{val_cls_acc3:.4f}",
                f"{best_acc1:.4f}",
                int(validated),
            ]
        )


def main() -> None:
    args = parse_args()
    if args.val_interval < 1:
        raise ValueError("--val-interval must be >= 1")
    if args.batch_size_per_device < 1:
        raise ValueError("--batch-size-per-device must be >= 1")
    if args.num_classes is not None and args.num_classes < 1:
        raise ValueError("--num-classes must be >= 1")

    # Construct output_dir from expt_name and current date
    date_suffix = datetime.now().strftime("%m%d")
    output_dir = Path("./outputs") / args.expt_name / date_suffix

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
    )

    seed_everything(TRAIN_CONFIG["seed"])

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Generate recipe filename based on model name
        model_name = TRAIN_CONFIG["model"]
        recipe_filename = f"{model_name}_recipe.json"
        with open(output_dir / recipe_filename, "w", encoding="utf-8") as f:
            json.dump(TRAIN_CONFIG, f, indent=2)
        init_epoch_log(output_dir / "train_log.csv")
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    else:
        writer = None

    train_set, val_set = build_datasets(args.data_root)
    inferred_num_classes = len(train_set.classes)
    num_classes = inferred_num_classes if args.num_classes is None else args.num_classes
    if args.num_classes is not None and args.num_classes != inferred_num_classes:
        raise RuntimeError(
            f"--num-classes={args.num_classes}, but dataset has {inferred_num_classes} classes "
            "from ImageFolder directories."
        )

    train_loader, val_loader = build_loaders(train_set, val_set, args.workers, args.batch_size_per_device)

    model = build_model(num_classes=num_classes)
    optimizer = build_optimizer(model)
    updates_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, updates_per_epoch)

    # Load checkpoint BEFORE preparing with accelerator
    start_epoch = 0
    best_acc1 = 0.0
    model_ema = None
    ckpt = None
    if args.resume is not None:
        # Create temporary EMA model for loading checkpoint
        if TRAIN_CONFIG["model_ema"]:
            model_ema = ModelEmaV2(model, decay=TRAIN_CONFIG["model_ema_decay"])
        start_epoch, best_acc1, ckpt = load_checkpoint(
            args.resume,
            model,
            model_ema,
            optimizer,
            scheduler,
        )
        # Clear temporary EMA, will recreate after prepare
        model_ema = None

    # Prepare everything with Accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Cache unwrapped model to avoid repeated unwrapping
    unwrapped_model = accelerator.unwrap_model(model)

    # Create EMA model AFTER prepare to ensure correct device placement
    if TRAIN_CONFIG["model_ema"]:
        model_ema = ModelEmaV2(unwrapped_model, decay=TRAIN_CONFIG["model_ema_decay"])
        model_ema.to(accelerator.device)

        # Reload EMA state if resuming (reuse loaded checkpoint)
        if args.resume is not None and ckpt is not None:
            if ckpt.get("model_ema") is not None:
                model_ema.module.load_state_dict(ckpt["model_ema"], strict=True)

    # Only create Mixup if enabled
    if TRAIN_CONFIG["mixup"] > 0 or TRAIN_CONFIG["cutmix"] > 0:
        mixup_fn = Mixup(
            mixup_alpha=TRAIN_CONFIG["mixup"],
            cutmix_alpha=TRAIN_CONFIG["cutmix"],
            cutmix_minmax=None,
            prob=TRAIN_CONFIG["mixup_prob"],
            switch_prob=TRAIN_CONFIG["mixup_switch_prob"],
            mode=TRAIN_CONFIG["mixup_mode"],
            label_smoothing=TRAIN_CONFIG["smoothing"],
            num_classes=num_classes,
        )
        train_criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        train_criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG["smoothing"])

    if accelerator.is_main_process:
        print("Training recipe:")
        print(json.dumps(TRAIN_CONFIG, indent=2))
        print(f"Experiment name: {args.expt_name}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset: {args.data_root}")
        print(f"Classes: {num_classes} (inferred: {inferred_num_classes})")
        per_step_global = args.batch_size_per_device * accelerator.num_processes
        print(
            f"World size: {accelerator.num_processes}, batch_size_per_device: {args.batch_size_per_device}, "
            f"per_step_global_batch: {per_step_global}"
        )
        print(f"Start epoch: {start_epoch}")

    num_updates = start_epoch * updates_per_epoch

    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start = time.time()

        # For tracking 50-step average loss
        step_loss_sum = 0.0
        step_loss_count = 0

        for it, (images, target) in enumerate(train_loader):
            with accelerator.autocast():
                # Apply mixup/cutmix on GPU if enabled
                if mixup_fn is not None:
                    images, target = mixup_fn(images, target)
                output = model(images)
                loss = train_criterion(output, target)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            num_updates += 1
            scheduler.step_update(num_updates)

            if model_ema is not None:
                model_ema.update(unwrapped_model)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Track loss for 50-step average
            step_loss_sum += loss.item()
            step_loss_count += 1

            # Log per-step metrics to TensorBoard (only on main process)
            if accelerator.is_main_process and writer is not None:
                cur_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train_step/loss", loss.item(), num_updates)
                writer.add_scalar("train_step/lr", cur_lr, num_updates)

            if accelerator.is_main_process and ((it + 1) % args.print_freq == 0):
                cur_lr = optimizer.param_groups[0]["lr"]
                avg_loss = total_loss / max(1, total_samples)

                # Log 50-step average loss to TensorBoard
                if writer is not None and step_loss_count > 0:
                    avg_50step_loss = step_loss_sum / step_loss_count
                    writer.add_scalar("train_step/loss_avg_50step", avg_50step_loss, num_updates)
                    # Reset counters
                    step_loss_sum = 0.0
                    step_loss_count = 0

                print(
                    f"Epoch [{epoch + 1}/{TRAIN_CONFIG['epochs']}] "
                    f"Iter [{it + 1}/{len(train_loader)}] "
                    f"Loss {avg_loss:.4f} LR {cur_lr:.6g}"
                )

        # Synchronize training loss across all processes
        loss_tensor = torch.tensor([total_loss, float(total_samples)], device=accelerator.device)
        gathered_tensors = accelerator.gather(loss_tensor)
        if gathered_tensors.dim() > 1:
            gathered_tensors = gathered_tensors.sum(dim=0)
        train_loss = (gathered_tensors[0] / gathered_tensors[1]).item() if gathered_tensors[1] > 0 else 0.0

        # Log training metrics to TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)

        do_validate = ((epoch + 1) % args.val_interval == 0) or (epoch + 1 == TRAIN_CONFIG["epochs"])
        is_best = False
        val_loss = None
        val_acc1 = None
        val_acc3 = None
        val_cls_acc1 = None
        val_cls_acc3 = None

        if do_validate:
            eval_model = model_ema.module if (model_ema is not None and TRAIN_CONFIG["model_ema_eval"]) else unwrapped_model
            val_loss, val_acc1, val_acc3, val_cls_acc1, val_cls_acc3 = evaluate(eval_model, val_loader, accelerator, num_classes)
            is_best = val_acc1 > best_acc1
            best_acc1 = max(best_acc1, val_acc1)

            # Log validation metrics to TensorBoard
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch + 1)
                writer.add_scalar("val/acc1", val_acc1, epoch + 1)
                writer.add_scalar("val/acc3", val_acc3, epoch + 1)
                writer.add_scalar("val/cls_acc1", val_cls_acc1, epoch + 1)
                writer.add_scalar("val/cls_acc3", val_cls_acc3, epoch + 1)
                writer.add_scalar("val/best_acc1", best_acc1, epoch + 1)

        if accelerator.is_main_process:
            elapsed = time.time() - epoch_start
            if do_validate:
                print(
                    f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
                    f"train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | val_acc1 {val_acc1:.2f} | "
                    f"val_acc3 {val_acc3:.2f} | val_cls_acc1 {val_cls_acc1:.2f} | "
                    f"val_cls_acc3 {val_cls_acc3:.2f} | best_acc1 {best_acc1:.2f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
                    f"train_loss {train_loss:.4f} | "
                    f"validation skipped (interval={args.val_interval}) | "
                    f"best_acc1 {best_acc1:.2f}"
                )

            append_epoch_log(
                log_path=output_dir / "train_log.csv",
                epoch=epoch + 1,
                train_loss=train_loss,
                validated=do_validate,
                val_loss=val_loss,
                val_acc1=val_acc1,
                val_acc3=val_acc3,
                val_cls_acc1=val_cls_acc1,
                val_cls_acc3=val_cls_acc3,
                best_acc1=best_acc1,
            )
            save_checkpoint(
                accelerator=accelerator,
                output_dir=output_dir,
                epoch=epoch,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                best_acc1=best_acc1,
                class_to_idx=train_set.class_to_idx,
                is_best=is_best,
            )

    if accelerator.is_main_process:
        print(f"Training completed. Best Acc@1: {best_acc1:.2f}")
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()