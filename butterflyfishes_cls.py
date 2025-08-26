import argparse
import datetime
from typing import Optional, List
import yaml
import shutil
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

from models.utils import Model
from datasets.dataloader import data_loader
from models.metrics import topk_accuracy


MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "vgg19"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/butterflyfishes.yaml")
    parser.add_argument("--resume_version", type=int, default=None)
    
    # Parse the config file path first.
    args, unknown = parser.parse_known_args(argv)
    
    # Load YAML config and merge with args.
    args = merge_yaml_with_args(args.cfg, args)
    return args


def merge_yaml_with_args(config_path: str, args: argparse.Namespace) -> argparse.Namespace:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"Not a valid config file: {config_path}")
            return args
        
        # Convert args to dict.
        args_dict = vars(args)
        
        # Merge config with args.
        for key, value_dict in config.items():
            if type(value_dict) is dict:
                for k, v in value_dict.items():
                    args_dict[k] = v
        
        # Convert back to Namespace.
        return argparse.Namespace(**args_dict)
        
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default arguments.")
        return args
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {config_path}: {e}. Using default arguments.")
        return args


def save_training_config(args, log_dir: Path):
    if args.resume_version is not None:
        # Resume training, check the resume config, then save current config.
        resume_config_path = log_dir.parent / f"version_{args.resume_version}" / "config.yaml"
        with open(resume_config_path, 'r') as f:
            resume_config = yaml.safe_load(f)
        args_dict = vars(args)
        for k, resume_v in resume_config.items():
            current_v = args_dict[k]
            if resume_v != current_v:
                print(f"Warning: the current and resume value of {k} are not match.")

    # Save current training config.
    config_file = log_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def setup_logging(args):
    if args.expt_name:
        expt_name = args.expt_name
    else:
        expt_name = f"{args.model_name}_{datetime.datetime.now().strftime('%m%d_%H%M')}"

    # Resume.
    if args.resume_version is not None:
        # Resume training.
        log_dir = Path("./logs") / f"{expt_name}" / f"version_{args.resume_version + 1}"
    else:
        # First time training.
        log_dir = Path("./logs") / f"{expt_name}" / "version_0"

    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)

    # Save training config file.
    save_training_config(args, log_dir)

    return logger, log_dir


def optimizer_scheduler(args, model, num_iters):
    # Initialize optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Create warmup scheduler and cosine annealing scheduler.
    warmup_iters_ratio = 0.1
    warmup_iters = int(num_iters * warmup_iters_ratio)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_iters - warmup_iters,
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )
    return optimizer, scheduler


def save_ckpt(model_state, ckpt_path: Path, is_best=False):
    torch.save(model_state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, ckpt_path.parent / "best.pth")


def load_ckpt(ckpt_path: Path, model, optimizer = None, scheduler = None):
    if not ckpt_path.exists():
        print(f"Checkpoint file {ckpt_path} not found!")
        return None, 0

    ckpt_state = torch.load(ckpt_path, map_location='cpu')
    start_epoch = ckpt_state.get('epoch', 0)
    print(f"\nLoad epoch {start_epoch} ckpt_state from {ckpt_path}, val top1 accuracy: {ckpt_state['val_top1_acc']:.2f}%")
    
    # Load model state dict.
    model.load_state_dict(ckpt_state["model_state_dict"])
    
    # Load optimizer state and scheduler state if provided.
    if (optimizer is not None) and (scheduler is not None):
        try:
            optimizer.load_state_dict(ckpt_state['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt_state['scheduler_state_dict'])
        except KeyError:
            print("Error: saved checkpoint state does not include optimizer state or scheduler state.")
        return model, start_epoch, optimizer, scheduler
    return model, start_epoch


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


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch, logger, global_iter, verbose_iters: int = 1):
    # Switch mode.
    model.train()
    
    running_loss = 0.0
    running_top1_acc = 0.0
    running_top3_acc = 0.0
    running_top5_acc = 0.0
    total_samples = 0
    total_batches = len(train_loader)
    for batch_idx, (input, target) in enumerate(train_loader):
        # Forward propagation.
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        # Number of samples.
        num_samples = len(target)
        total_samples += num_samples

        # Metrics.
        topk_acc = topk_accuracy(output, target, topk=(1, 3, 5))
        top1_acc, top3_acc, top5_acc = [acc.item() for acc in topk_acc]

        # Backward propagation, update optimizer and scheduler.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update running metrics
        running_loss += loss.item() * num_samples
        running_top1_acc += top1_acc * num_samples
        running_top3_acc += top3_acc * num_samples
        running_top5_acc += top5_acc * num_samples
        
        # Log metrics for each iteration
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar('Iter/Loss', loss.item(), global_iter)
        logger.add_scalar('Iter/Top1_Accuracy', top1_acc, global_iter)
        logger.add_scalar('Iter/Top3_Accuracy', top3_acc, global_iter)
        logger.add_scalar('Iter/Top5_Accuracy', top5_acc, global_iter)
        logger.add_scalar('Iter/Learning_Rate', current_lr, global_iter)
        
        global_iter += 1
        
        # Print progress every 100 iterations
        if verbose_iters and (batch_idx+1) % verbose_iters == 0:
            print(f'Epoch [{epoch}/{args.epochs}] [{batch_idx+1}/{total_batches}] '
                  f'Loss: {loss.item():.4f} '
                  f'Top1: {top1_acc:.2f}% '
                  f'Top3: {top3_acc:.2f}% '
                  f'Top5: {top5_acc:.2f}% '
                  f'LR: {current_lr:.6f}')
    
    # Log epoch-level metrics
    avg_loss = running_loss / total_samples
    avg_top1_acc = running_top1_acc / total_samples
    avg_top3_acc = running_top3_acc / total_samples
    avg_top5_acc = running_top5_acc / total_samples
    logger.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
    logger.add_scalar('Epoch/Train_Top1_Accuracy', avg_top1_acc, epoch)
    logger.add_scalar('Epoch/Train_Top3_Accuracy', avg_top3_acc, epoch)
    logger.add_scalar('Epoch/Train_Top5_Accuracy', avg_top5_acc, epoch)
    
    return global_iter


def validate(val_loader, model, criterion, epoch, logger = None, verbose: bool = True):
    # Switch mode.
    model.eval()

    running_loss = 0.0
    running_top1_acc = 0.0
    running_top3_acc = 0.0
    running_top5_acc = 0.0
    total_samples = 0

    for batch_idx, (input, target) in enumerate(val_loader):
        # Forward propagation.
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            num_samples = len(target)
            total_samples += num_samples

            # Metrics.
            topk_acc = topk_accuracy(output, target, topk=(1, 3, 5))
            top1_acc, top3_acc, top5_acc = [acc.item() for acc in topk_acc]

            # Update running metrics.
            running_loss += loss.item() * num_samples
            running_top1_acc += top1_acc * num_samples
            running_top3_acc += top3_acc * num_samples
            running_top5_acc += top5_acc * num_samples

    # Log epoch-level metrics
    avg_loss = running_loss / total_samples
    avg_top1_acc = running_top1_acc / total_samples
    avg_top3_acc = running_top3_acc / total_samples
    avg_top5_acc = running_top5_acc / total_samples
    if logger:
        logger.add_scalar('Val/Loss', avg_loss, epoch)
        logger.add_scalar('Val/Top1_Accuracy', avg_top1_acc, epoch)
        logger.add_scalar('Val/Top3_Accuracy', avg_top3_acc, epoch)
        logger.add_scalar('Val/Top5_Accuracy', avg_top5_acc, epoch)
    if verbose:
        print(f'Val Loss: {avg_loss:.4f} '
              f'Val Top1: {avg_top1_acc:.2f}% '
              f'Val Top3: {avg_top3_acc:.2f}% '
              f'Val Top5: {avg_top5_acc:.2f}%')
    return avg_loss, avg_top1_acc, avg_top3_acc, avg_top5_acc


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
    if args.model_name in MODEL_NAMES:
        with open(f"./configs/models/{args.model_name}.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
        model = Model(model_cfg)
    else:
        print("Not a supported model.")
        return

    # Load datasets.
    train_loader, val_loader = data_loader(args)
    num_iters = len(train_loader) * args.epochs

    # Define optimizer, scheduler and loss.
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = optimizer_scheduler(args, model, num_iters)

    # Training and validating.
    model, criterion = to_device([model, criterion])
    global_iter = 0
    best_val_acc = 0.0

    # Handle resume training.
    if args.resume_version is not None:
        resume_ckpts_dir = log_dir.parent / f"version_{args.resume_version}" / "ckpts"
        ckpt_path = get_latest_ckpt(resume_ckpts_dir)
        if ckpt_path:
            model, start_epoch, optimizer, scheduler = load_ckpt(ckpt_path, model, optimizer, scheduler)
            global_iter = start_epoch * len(train_loader)  # Resume from the last global_iter
        else:
            print(f"No checkpoint found for resume_version {args.resume_version}. Starting from epoch 1.")
            start_epoch = 1
    else:
        # First time training.
        if args.pretrained is not None:
            pretrained_ckpt_path = Path("./pretrained") / f"{args.model_name}_{args.pretrained}.pth"
            model = load_pretrained_ckpt(model, pretrained_ckpt_path)
        start_epoch = 1

    # Start training.
    for epoch in range(start_epoch, args.epochs + 1):
        global_iter = train(args, train_loader, model, criterion, optimizer, scheduler, epoch, logger, global_iter, verbose_iters=5)

        if (args.validate_epoch is not None) and (epoch % args.validate_epoch == 0):
            # Validate.
            loss, top1_acc, top3_acc, top5_acc = validate(val_loader, model, criterion, epoch, logger, verbose=True)

            # Save checkpoint file.
            ckpt_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_top1_acc": top1_acc,
                "val_top3_acc": top3_acc,
                "val_top5_acc": top5_acc,
            }
            ckpt_file = log_dir / "ckpts" / f"epoch_{epoch}.pth"
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            if top1_acc > best_val_acc:
                is_best = True
                best_val_acc = top1_acc
                print(f"Best ckpt at epoch {epoch}, top1_acc: {best_val_acc:.2f}%")
            else:
                is_best = False
            save_ckpt(ckpt_state, ckpt_file, is_best=is_best)
    
    # Close logger
    logger.close()
    print("Training completed. Logs saved to:", logger.log_dir)


def report_best_val_results(args, best_ckpt: Path) -> None:
    # Model.
    with open(f"./configs/models/{args.model_name}.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model = Model(model_cfg)

    # Load best checkpoint.
    model, epoch = load_ckpt(best_ckpt, model)
    model = model.cuda()

    # Load validation dataset.
    _, val_loader = data_loader(args)

    # Create per class metrics.
    top1_acc_per_cls, top3_acc_per_cls, top5_acc_per_cls = {}, {}, {}
    idx_to_class_path = Path(args.root) / "idx_to_class.yaml"
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
    args = parse_args()
    main(args)

    # Report best validation results.
    # ckpt_path = Path("/home/ziliang/Projects/inference-benchmark/logs/butterflyfishes-resnet34/version_0/ckpts/best.pth")
    # report_best_val_results(args, ckpt_path)