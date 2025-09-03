from typing import List
import yaml
import shutil
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.backends.mkl import verbose

from models.utils import Model
from datasets.dataloader import data_loader
from ai_shared.training_engine import (
    load_resume_ckpt, save_ckpt,
    setup_logging, parse_args, merge_yaml_with_args,
    set_deterministic, train_epoch, validate_epoch
)
from ai_shared.optimizer import optimizer_scheduler
from ai_shared.metrics import TopKAccuracy


MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "vgg19"]


def load_pretrained_ckpt(model, pretrained_ckpt: Path):
    pretrained_state = torch.load(pretrained_ckpt, map_location='cpu')
    model_state = model.state_dict()

    # def strip_prefixes(key: str) -> str:
    #     prefixes = ['backbone_modules.', 'resnet_layer.']
    #     for p in prefixes:
    #         key = key.split(p)[-1]
    #     return key
    #
    # # Stripped key for model and pretrained model.
    # model_state_stripped = {}
    # for k, v in model_state.items():
    #     model_state_stripped[strip_prefixes(k)] = (k, v)
    # pretrained_state_stripped = {}
    # for k, v in pretrained_state.items():
    #     pretrained_state_stripped[strip_prefixes(k)] = (k, v)


    loaded_params = {}
    num_matched = 0
    num_shape_mismatch = 0
    num_unmatched = 0

    for mk, mv in model_state.items():
        # Skip classification heads by common names
        if any(h in mk for h in ['fc.weight', 'fc.bias', 'classifier.']):
            continue

        smk = strip_prefixes(mk)
        if smk in pretrained_by_stripped:
            pk, pv = pretrained_by_stripped[smk]
            if hasattr(pv, 'shape') and hasattr(mv, 'shape') and pv.shape == mv.shape:
                loaded_params[mk] = pv
                num_matched += 1
            else:
                num_shape_mismatch += 1
        else:
            num_unmatched += 1

    # Update and load
    model_state.update(loaded_params)
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    print("=" * 100)
    print(f"Loaded pretrained weights from {pretrained_ckpt}")
    print(f"Matched params: {num_matched}")
    print(f"Shape mismatches: {num_shape_mismatch}")
    print(f"Unmatched model params (by name): {num_unmatched}")
    if missing:
        print(f"Missing keys in loaded state (not filled): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys in checkpoint (ignored): {len(unexpected)}")
    print("=" * 100)

    return model


def get_latest_ckpt(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    
    # Find all checkpoint files.
    ckpt_files = list(ckpt_dir.glob("epoch_*.pth"))
    if not ckpt_files:
        return None
    
    # Sort by epoch number and return the latest.
    ckpt_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    return ckpt_files[-1]



def to_device(nn_module_list: List):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_module_list = []
    for module in nn_module_list:
        device_module_list.append(module.to(device))
    return device_module_list


def main(args) -> None:
    # Setup logging.
    logger, log_dir = setup_logging(args)
    
    # Create model from model config file.
    if args.train.model_name in MODEL_NAMES:
        with open(f"./configs/models/{args.train.model_name}.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
        model = Model(model_cfg)
    else:
        print("Not a supported model.")
        return

    # Load datasets.
    train_loader, val_loader = data_loader(args)
    num_iters = len(train_loader) * args.train.epochs

    # Define optimizer, scheduler and loss.
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = optimizer_scheduler(args, model, num_iters)

    # Training and validating.
    model, criterion = to_device([model, criterion])
    global_iter = 0
    best_val_metric = 0.0

    # Handle resume training.
    if args.train.resume_version is not None:
        resume_ckpts_dir = log_dir.parent / f"version_{args.train.resume_version}" / "ckpts"
        ckpt_path = get_latest_ckpt(resume_ckpts_dir)
        if ckpt_path:
            model, start_epoch, optimizer, scheduler = load_resume_ckpt(ckpt_path, model, optimizer, scheduler)
            global_iter = start_epoch * len(train_loader)  # Resume from the last global_iter
        else:
            print(f"No checkpoint found for resume_version {args.train.resume_version}. Starting from epoch 1.")
            start_epoch = 1
    else:
        # First time training.
        if args.train.pretrained is not None:
            pretrained_ckpt_path = Path("./pretrained") / f"{args.train.model_name}_{args.train.pretrained}.pth"
            model = load_pretrained_ckpt(model, pretrained_ckpt_path)
        start_epoch = 1

    # Start training.
    for epoch in range(start_epoch, args.train.epochs + 1):
        metrics_fn = {
            "Top1_Acc": TopKAccuracy(topk=1),
            "Top3_Acc": TopKAccuracy(topk=3),
            "Top5_Acc": TopKAccuracy(topk=5),
        }
        global_iter = train_epoch(args, train_loader, model, criterion, optimizer, scheduler, epoch, logger,
                                  global_iter, metrics_fn=metrics_fn, verbose_iters=5)

        if (args.train.validate_epoch is not None) and (epoch % args.train.validate_epoch == 0):
            avg_losses, avg_metrics = validate_epoch(args, val_loader, model, criterion, epoch, logger,
                                                     metrics_fn=metrics_fn, verbose=True)

            # Save checkpoint file.
            ckpt_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": avg_metrics,
            }
            ckpt_file = log_dir / "ckpts" / f"epoch_{epoch}.pth"
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)

            # Save best checkpoint.
            main_metric = list(avg_metrics.values())[0]
            if main_metric > best_val_metric:
                is_best = True
                best_val_metric = main_metric
                print(f"Best ckpt at epoch {epoch}, {list(avg_metrics.keys())[0]}: {best_val_metric:.2f}%")
            else:
                is_best = False
            save_ckpt(ckpt_state, ckpt_file, is_best=is_best)
    
    # Close logger
    logger.close()
    print("Training completed. Logs saved to:", logger.log_dir)


def report_best_val_results(args, best_ckpt: Path) -> None:
    # Model.
    with open(f"./configs/models/{args.train.model_name}.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model = Model(model_cfg)

    # Load best checkpoint.
    model, epoch = load_resume_ckpt(best_ckpt, model)
    model = model.cuda()

    # Load validation dataset.
    _, val_loader = data_loader(args)

    # Create per class metrics.
    top1_acc_per_cls, top3_acc_per_cls, top5_acc_per_cls = {}, {}, {}
    idx_to_class_path = Path(args.data.root_path) / "idx_to_class.yaml"
    with open(idx_to_class_path, "r") as f:
        idx_to_class = yaml.safe_load(f)
    num_cls = len(idx_to_class)
    
    # Initialize per-class counters
    for idx, class_name in idx_to_class.items():
        top1_acc_per_cls[class_name] = {"correct": 0, "total": 0}
        top3_acc_per_cls[class_name] = {"correct": 0, "total": 0}
        top5_acc_per_cls[class_name] = {"correct": 0, "total": 0}

    # Validate.
    model.eval()
    for batch_idx, (input, target) in enumerate(val_loader):
        # Forward propagation.
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)

            # Get predictions for top-1, top-3, top-5
            _, pred_top1 = output.topk(1, 1, largest=True, sorted=True)
            _, pred_top3 = output.topk(3, 1, largest=True, sorted=True)
            _, pred_top5 = output.topk(5, 1, largest=True, sorted=True)
            
            # Calculate per-class accuracy
            for i in range(target.size(0)):
                gt_class_idx = target[i].item()
                gt_class_name = idx_to_class[gt_class_idx]
                
                # Update total count for this class
                top1_acc_per_cls[gt_class_name]["total"] += 1
                top3_acc_per_cls[gt_class_name]["total"] += 1
                top5_acc_per_cls[gt_class_name]["total"] += 1
                
                # Check if prediction is correct for top-1
                if gt_class_idx == pred_top1[i, 0].item():
                    top1_acc_per_cls[gt_class_name]["correct"] += 1
                
                # Check if prediction is correct for top-3
                if gt_class_idx in pred_top3[i].tolist():
                    top3_acc_per_cls[gt_class_name]["correct"] += 1
                
                # Check if prediction is correct for top-5
                if gt_class_idx in pred_top5[i].tolist():
                    top5_acc_per_cls[gt_class_name]["correct"] += 1

    # Calculate final per-class accuracy percentages
    print("\n=== Per-Class Validation Results ===")
    print(f"{'Class Name':<25} {'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Top-5 Acc':<12} {'Samples':<10}")
    print("-" * 100)
    
    total_samples = 0
    total_top1_correct = 0
    total_top3_correct = 0
    total_top5_correct = 0
    mean_top1_acc = 0
    mean_top3_acc = 0
    mean_top5_acc = 0
    for class_name in sorted(top1_acc_per_cls.keys()):
        total = top1_acc_per_cls[class_name]["total"]
        if total == 0:
            continue
            
        top1_acc = (top1_acc_per_cls[class_name]["correct"] / total) * 100
        top3_acc = (top3_acc_per_cls[class_name]["correct"] / total) * 100
        top5_acc = (top5_acc_per_cls[class_name]["correct"] / total) * 100
        mean_top1_acc += top1_acc
        mean_top3_acc += top3_acc
        mean_top5_acc += top5_acc
        
        print(f"{class_name:<25} {top1_acc:>8.2f}% {top3_acc:>8.2f}% {top5_acc:>8.2f}% {total:>8}")
        
        total_samples += total
        total_top1_correct += top1_acc_per_cls[class_name]["correct"]
        total_top3_correct += top3_acc_per_cls[class_name]["correct"]
        total_top5_correct += top5_acc_per_cls[class_name]["correct"]
    
    # Calculate overall accuracy
    overall_top1_acc = (total_top1_correct / total_samples) * 100
    overall_top3_acc = (total_top3_correct / total_samples) * 100
    overall_top5_acc = (total_top5_correct / total_samples) * 100

    # Calculate mean class accuracy.
    mean_top1_acc = mean_top1_acc / num_cls
    mean_top3_acc = mean_top3_acc / num_cls
    mean_top5_acc = mean_top5_acc / num_cls
    
    print("-" * 100)
    print(f"{'Class Mean':<25} {mean_top1_acc:>8.2f}% {mean_top3_acc:>8.2f}% {mean_top5_acc:>8.2f}% {num_cls:>8}")
    print(f"{'Overall':<25} {overall_top1_acc:>8.2f}% {overall_top3_acc:>8.2f}% {overall_top5_acc:>8.2f}% {total_samples:>8}")

    
    # Save detailed results to CSV file
    
    # Create DataFrame for per-class results
    results_data = []
    for class_name in sorted(top1_acc_per_cls.keys()):
        total = top1_acc_per_cls[class_name]["total"]
        if total == 0:
            continue
            
        top1_acc = (top1_acc_per_cls[class_name]["correct"] / total) * 100
        top3_acc = (top3_acc_per_cls[class_name]["correct"] / total) * 100
        top5_acc = (top5_acc_per_cls[class_name]["correct"] / total) * 100
        
        results_data.append({
            'Class_Name': class_name,
            'Top1_Accuracy': round(top1_acc, 2),
            'Top3_Accuracy': round(top3_acc, 2),
            'Top5_Accuracy': round(top5_acc, 2),
            'Samples': total
        })
    
    # Add overall results and class mean results.
    results_data.append({
        "Class_Name": "Class Mean",
        "Top1_Accuracy": round(mean_top1_acc, 2),
        "Top3_Accuracy": round(mean_top3_acc, 2),
        "Top5_Accuracy": round(mean_top5_acc, 2),
        "Samples": num_cls
    })
    results_data.append({
        'Class_Name': 'Overall',
        'Top1_Accuracy': round(overall_top1_acc, 2),
        'Top3_Accuracy': round(overall_top3_acc, 2),
        'Top5_Accuracy': round(overall_top5_acc, 2),
        'Samples': total_samples
    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)
    csv_file = best_ckpt.parent.parent / "per_class_results.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\nDetailed results saved to: {csv_file}")


if __name__ == "__main__":
    set_deterministic(666)
    argv = ['--cfg_path', './configs/butterflyfishes.yaml']
    args = parse_args(argv)
    args = merge_yaml_with_args(args.cfg_path, args)
    main(args)

    # Report best validation results.
    # ckpt_path = Path("/home/ziliang/Projects/inference-benchmark/logs/butterflyfishes-resnet34/version_0/ckpts/best.pth")
    # report_best_val_results(args, ckpt_path)