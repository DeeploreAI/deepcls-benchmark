#!/usr/bin/env python3
"""Standalone evaluation script for trained models."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm
from accelerate import Accelerator
from timm.utils import accuracy

from train import build_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/Datasets/DeepDet"),
        help="Path to dataset root",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA model if available",
    )
    return parser.parse_args()


def evaluate_with_per_class(
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

    # Create progress bar only on main process
    if accelerator.is_main_process:
        pbar = tqdm(loader, desc="Evaluating", unit="batch")
    else:
        pbar = loader

    for images, target in pbar:
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

        # Calculate per-class accuracy (top-1 and top-3)
        pred_top1 = output.argmax(dim=1)
        if len(topk) == 2:
            _, pred_top3 = output.topk(3, dim=1, largest=True, sorted=True)
        else:
            pred_top3 = pred_top1.unsqueeze(1)

        for cls_idx in range(num_classes):
            cls_mask = target == cls_idx
            class_total[cls_idx] += cls_mask.sum()
            # Top-1 per-class accuracy
            class_correct_top1[cls_idx] += (pred_top1[cls_mask] == target[cls_mask]).sum()
            # Top-3 per-class accuracy
            if len(topk) == 2:
                target_expanded = target[cls_mask].unsqueeze(1).expand_as(pred_top3[cls_mask])
                class_correct_top3[cls_idx] += (pred_top3[cls_mask] == target_expanded).any(dim=1).sum()
            else:
                class_correct_top3[cls_idx] += (pred_top1[cls_mask] == target[cls_mask]).sum()

        # Update progress bar with current metrics
        if accelerator.is_main_process:
            current_acc1 = (total_acc1 / total_samples) if total_samples > 0 else 0.0
            pbar.set_postfix({"acc1": f"{current_acc1:.2f}%", "samples": total_samples})

    # Avoid division by zero
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], []

    avg_loss = total_loss / total_samples
    avg_acc1 = total_acc1 / total_samples
    avg_acc3 = total_acc3 / total_samples

    # Calculate per-class accuracies
    per_class_acc1 = []
    per_class_acc3 = []
    for cls_idx in range(num_classes):
        if class_total[cls_idx] > 0:
            acc1 = (class_correct_top1[cls_idx] / class_total[cls_idx] * 100).item()
            acc3 = (class_correct_top3[cls_idx] / class_total[cls_idx] * 100).item()
        else:
            acc1 = None  # No data for this class
            acc3 = None  # No data for this class
        per_class_acc1.append(acc1)
        per_class_acc3.append(acc3)

    # Calculate average per-class accuracy
    valid_classes = class_total > 0
    if valid_classes.sum() > 0:
        class_acc1 = class_correct_top1[valid_classes] / class_total[valid_classes]
        class_acc3 = class_correct_top3[valid_classes] / class_total[valid_classes]
        avg_per_cls_acc1 = (class_acc1.mean() * 100).item()
        avg_per_cls_acc3 = (class_acc3.mean() * 100).item()
    else:
        avg_per_cls_acc1 = 0.0
        avg_per_cls_acc3 = 0.0

    return avg_loss, avg_acc1, avg_acc3, avg_per_cls_acc1, avg_per_cls_acc3, per_class_acc1, per_class_acc3


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Load checkpoint
    if accelerator.is_main_process:
        print(f"Loading checkpoint from {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Extract config and class mapping
    config = ckpt.get("official_recipe", {})
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)
    epoch = ckpt.get("epoch", -1)  # Get epoch from checkpoint

    # Create idx_to_class mapping
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    if accelerator.is_main_process:
        print(f"Model: {config.get('model', 'unknown')}")
        print(f"Number of classes: {num_classes}")

    # Build dataset
    _, val_set = build_datasets(args.data_root)

    # Verify class mapping matches
    if val_set.class_to_idx != class_to_idx:
        raise RuntimeError("Class mapping mismatch between checkpoint and dataset")

    # Build dataloader
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model
    model = timm.create_model(
        config.get("model", "resnet50"),
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=0.0,  # No dropout during evaluation
    )

    # Load weights
    if args.use_ema and ckpt.get("model_ema") is not None:
        if accelerator.is_main_process:
            print("Using EMA model weights")
        model.load_state_dict(ckpt["model_ema"], strict=True)
    else:
        if accelerator.is_main_process:
            print("Using standard model weights")
        model.load_state_dict(ckpt["model"], strict=True)

    # Prepare with Accelerator
    model, val_loader = accelerator.prepare(model, val_loader)

    # Run evaluation
    if accelerator.is_main_process:
        print("\nRunning evaluation...")

    val_loss, val_acc1, val_acc3, val_cls_acc1, val_cls_acc3, per_class_acc1, per_class_acc3 = evaluate_with_per_class(
        model, val_loader, accelerator, num_classes
    )

    # Print results
    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Loss:              {val_loss:.6f}")
        print(f"Avg Top-1 Acc:     {val_acc1:.2f}%")
        print(f"Avg Top-3 Acc:     {val_acc3:.2f}%")
        print(f"Per-Class Top-1:   {val_cls_acc1:.2f}%")
        print(f"Per-Class Top-3:   {val_cls_acc3:.2f}%")
        print("=" * 50)

        # Determine output directory (parent of checkpoints folder)
        checkpoint_dir = args.checkpoint.parent
        if checkpoint_dir.name == "checkpoints":
            output_dir = checkpoint_dir.parent / "evaluations"
        else:
            output_dir = checkpoint_dir / "evaluations"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on epoch, model type, and evaluation date
        eval_date = datetime.now().strftime("%Y%m%d")
        if args.use_ema:
            csv_filename = f"eval_epoch_{epoch + 1:03d}_ema_{eval_date}.csv"
        else:
            csv_filename = f"eval_epoch_{epoch + 1:03d}_{eval_date}.csv"
        csv_path = output_dir / csv_filename

        # Write results to CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write overall metrics header and data
            writer.writerow(["metric", "value"])
            writer.writerow(["epoch", epoch + 1])
            writer.writerow(["use_ema", args.use_ema])
            writer.writerow(["val_loss", f"{val_loss:.6f}"])
            writer.writerow(["avg_acc1", f"{val_acc1:.4f}"])
            writer.writerow(["avg_acc3", f"{val_acc3:.4f}"])
            writer.writerow(["per_cls_acc1", f"{val_cls_acc1:.4f}"])
            writer.writerow(["per_cls_acc3", f"{val_cls_acc3:.4f}"])

            # Empty row separator
            writer.writerow([])

            # Write per-class results header and data
            writer.writerow(["class_name", "class_idx", "acc1", "acc3"])
            for cls_idx in range(num_classes):
                class_name = idx_to_class[cls_idx]
                acc1_str = f"{per_class_acc1[cls_idx]:.4f}" if per_class_acc1[cls_idx] is not None else ""
                acc3_str = f"{per_class_acc3[cls_idx]:.4f}" if per_class_acc3[cls_idx] is not None else ""
                writer.writerow([
                    class_name,
                    cls_idx,
                    acc1_str,
                    acc3_str,
                ])

        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
