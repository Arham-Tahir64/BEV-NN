"""Train camera-only semantic occupancy prediction.

First milestone: overfit one training frame with batch size 1.
Use --overfit-samples 0 to train the complete official mini training split.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data import DEFAULT_DATAROOT, NuScenesOccupancyDataset, make_dataloader, load_occupancy_labels
from data.nuscenes_dataset import IMAGE_MEAN, IMAGE_STD
from src.models.occupancy_model import OccupancyModel
from data.depth_targets import DepthTargetDataset, depth_loss


def parse_args() -> argparse.Namespace:
    """Read training configuration from the command line."""
    parser = argparse.ArgumentParser(
        description="Train camera-only semantic occupancy; defaults target a one-frame overfit check."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100).")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4,
                        help="AdamW learning rate (default: 1e-4).")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01).")
    parser.add_argument("--batch-size", type=int, default=1, help="Samples per batch (default: 1).")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers; zero avoids Windows subprocesses (default: 0).")
    parser.add_argument("--seed", type=int, default=0, help="Nonnegative random seed (default: 0).")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto selects CUDA when available, otherwise CPU (default: auto).")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable mixed precision on CUDA; --no-amp disables it (default: enabled).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "checkpoints",
                        help="Directory for checkpoints and training logs.")
    parser.add_argument("--overfit-samples", type=int, default=1,
                        help="Use the first N training samples; 0 uses the full training split (default: 1).")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=None,
                        help="Run held-out validation; defaults off for overfit mode and on for full training.")
    parser.add_argument("--class-balance", choices=("none", "sqrt-inverse"), default="none",
                        help="Optional inverse-square-root class weighting, estimated from selected training labels only.")
    parser.add_argument("--xyz", action=argparse.BooleanOptionalAction, default=False,
                        help="Add normalized XYZ positional features before the unchanged decoder.")
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable supervised depth-conditioned lifting (new runs).")
    parser.add_argument("--depth-bins", type=int, default=64)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=64.1)
    parser.add_argument("--depth-loss-weight", type=float, default=1.0)
    parser.add_argument("--depth-cache", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "depth_cache")
    args = parser.parse_args()
    if (args.depth_bins <= 0 or not math.isfinite(args.depth_min)
            or not math.isfinite(args.depth_max) or not 0 < args.depth_min < args.depth_max):
        parser.error("Depth bins must be positive and bounds finite with 0 < min < max.")
    if not math.isfinite(args.depth_loss_weight) or args.depth_loss_weight <= 0:
        parser.error("--depth-loss-weight must be finite and positive.")

    for name in ("epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("workers", "seed", "overfit_samples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative.")

    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def estimate_class_weights(dataset, num_classes: int = 18) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate inverse-sqrt weights using selected training annotations only.

    Unobserved classes retain neutral weight one. Observed weights have mean
    one; their relative ratios, rather than this common scale, affect the loss.
    Camera images are not decoded while counting labels. Only a direct
    NuScenesOccupancyDataset or its selected Subset is expected here.
    """
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    counts = torch.zeros(num_classes, dtype=torch.int64)
    for index in indices:
        occupancy = load_occupancy_labels(source.records[index]["label_path"])
        labels = occupancy["semantics"]
        valid = occupancy["mask_camera"] & (labels != 255)
        selected = labels[valid].astype(np.int64)
        if np.any((selected < 0) | (selected >= num_classes)):
            raise ValueError("Training labels contain an unsupported semantic class.")
        counts += torch.from_numpy(np.bincount(selected, minlength=num_classes))
    observed = counts > 0
    if not observed.any():
        raise ValueError("Cannot estimate class weights without supervised training voxels.")
    weights = torch.ones(num_classes, dtype=torch.float32)
    weights[observed] = counts[observed].float().rsqrt()
    weights[observed] /= weights[observed].mean()
    return weights, counts


def compute_loss(
    logits: torch.Tensor,
    semantics: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return masked cross-entropy, or None when no supervised voxels exist.

    logits: [B,18,X,Y,Z], raw class scores.
    semantics: [B,X,Y,Z], int64 class IDs; ignore ID is 255.
    valid_mask: [B,X,Y,Z], boolean camera-visible, non-ignore supervision.
    class_weights: optional positive [C] training-class weights.
    """
    if logits.ndim != 5 or any(size <= 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape [B,C,X,Y,Z].")
    target_shape = (logits.shape[0], *logits.shape[2:])
    if semantics.shape != target_shape or valid_mask.shape != target_shape:
        raise ValueError("semantics and valid_mask must match logits' [B,X,Y,Z] dimensions.")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor.")
    if semantics.dtype != torch.int64 or valid_mask.dtype != torch.bool:
        raise TypeError("semantics must be int64 and valid_mask must be boolean.")
    if not (logits.device == semantics.device == valid_mask.device):
        raise ValueError("logits, semantics, and valid_mask must be on the same device.")

    # masked_fill creates a new tensor, leaving the original labels unchanged.
    # All valid semantic classes remain supervised, including 0 and free space 17.
    targets = semantics.masked_fill(~valid_mask, 255)
    if not (targets != 255).any():
        return None

    if class_weights is not None:
        if class_weights.shape != (logits.shape[1],) or not class_weights.is_floating_point():
            raise ValueError("class_weights must be a floating-point vector with one entry per class.")
        if class_weights.device != logits.device:
            raise ValueError("class_weights and logits must be on the same device.")
        if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
            raise ValueError("class_weights must be finite and positive.")
        class_weights = class_weights.to(dtype=logits.dtype)
    return F.cross_entropy(logits, targets, weight=class_weights, ignore_index=255)


def train_one_epoch(
    model: OccupancyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    class_weights: torch.Tensor | None = None,
    depth_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Update weights and average epoch loss by its target-weight denominator.

    The scaler should be enabled only for CUDA mixed precision. Batches without
    supervised voxels are skipped; an entirely unsupervised/empty epoch raises
    RuntimeError rather than reporting a misleading zero loss.
    """
    model.train()  # ImageEncoder.train() keeps its frozen backbone in eval mode.
    use_amp = amp_enabled and device.type == "cuda"
    total_loss = 0.0
    total_voxels = 0
    total_loss_weight = 0.0
    processed_batches = 0
    skipped_batches = 0
    depth_enabled = getattr(model, "depth_head", None) is not None
    depth_loss_sum = 0.0
    depth_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        ego_to_camera = batch["ego_to_camera"].to(device, non_blocking=True)
        semantics = batch["semantics"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        valid_voxels = int((valid_mask & (semantics != 255)).sum().item())
        if valid_voxels == 0:
            skipped_batches += 1
            print(f"Train batch {batch_index}/{len(loader)}: skipped (no valid targets).")
            continue

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if depth_enabled:
                auxiliary = model(images, intrinsics, ego_to_camera, return_aux=True)
                logits = auxiliary["occupancy_logits"]
            else:
                logits = model(images, intrinsics, ego_to_camera)
            loss = compute_loss(logits, semantics, valid_mask, class_weights)

        if loss is None:
            skipped_batches += 1
            continue
        loss_value = loss.detach().item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss in batch {batch_index}: {loss_value}")

        if depth_enabled:
            if "depth_target" not in batch:
                raise ValueError("Depth training requires cached depth_target from DepthTargetDataset.")
            auxiliary_loss, count = depth_loss(auxiliary["depth_logits"],
                batch["depth_target"].to(device), model.depth_head.bin_edges)
            if not torch.isfinite(auxiliary_loss):
                raise FloatingPointError("Non-finite depth loss.")
            depth_loss_sum += auxiliary_loss.detach().item() * count
            depth_count += count
            loss = loss + depth_loss_weight * auxiliary_loss

        # With a disabled scaler, these perform ordinary backward and step.
        # With CUDA AMP, scaling helps prevent FP16 gradient underflow.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Match weighted cross-entropy's sum-of-target-weights denominator.
        loss_weight = float(valid_voxels)
        if class_weights is not None:
            effective_weights = class_weights.to(dtype=logits.dtype)
            target_weights = effective_weights[semantics[valid_mask & (semantics != 255)]]
            sum_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            loss_weight = target_weights.sum(dtype=sum_dtype).item()
        total_loss += loss_value * loss_weight
        total_loss_weight += loss_weight
        total_voxels += valid_voxels
        processed_batches += 1
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
            print(
                f"Train batch {batch_index}/{len(loader)}: loss={loss_value:.4f}, "
                f"mean_loss={total_loss / total_loss_weight:.4f}, valid_voxels={valid_voxels}"
            )

    if total_voxels == 0:
        raise RuntimeError("Training epoch has no supervised voxels; check the data and visibility masks.")
    if depth_enabled and depth_count == 0:
        raise RuntimeError("No trustworthy depth targets in this epoch. Inspect target coverage before training.")
    return {
        "loss": total_loss / total_loss_weight,
        **({"depth_loss": depth_loss_sum / depth_count, "depth_targets": depth_count}
           if depth_enabled else {}),
        "valid_voxels": total_voxels,
        "batches": processed_batches,
        "skipped_batches": skipped_batches,
    }


def validate(
    model: OccupancyModel,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    """Return held-out masked loss and voxel accuracy without updating weights."""
    model.eval()
    use_amp = amp_enabled and device.type == "cuda"
    total_loss = 0.0
    total_voxels = 0
    correct_voxels = 0
    processed_batches = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            intrinsics = batch["intrinsics"].to(device, non_blocking=True)
            ego_to_camera = batch["ego_to_camera"].to(device, non_blocking=True)
            semantics = batch["semantics"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            supervised = valid_mask & (semantics != 255)
            valid_voxels = int(supervised.sum().item())
            if valid_voxels == 0:
                skipped_batches += 1
                continue

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images, intrinsics, ego_to_camera)
                loss = compute_loss(logits, semantics, valid_mask)
            if loss is None:
                skipped_batches += 1
                continue
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite validation loss in batch {batch_index}: {loss_value}")

            total_loss += loss_value * valid_voxels
            total_voxels += valid_voxels
            correct_voxels += int(((logits.argmax(dim=1) == semantics) & supervised).sum().item())
            processed_batches += 1
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"Validation batch {batch_index}/{len(loader)}: "
                    f"mean_loss={total_loss / total_voxels:.4f}"
                )

    if total_voxels == 0:
        raise RuntimeError("Validation has no supervised voxels; check the data and visibility masks.")
    return {
        "loss": total_loss / total_voxels,
        "accuracy": correct_voxels / total_voxels,
        "valid_voxels": total_voxels,
        "batches": processed_batches,
        "skipped_batches": skipped_batches,
    }


def save_checkpoint(
    path: Path,
    model: OccupancyModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: dict,
) -> None:
    """Save model/training state and config, replacing the file after writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or check your installation.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    amp_enabled = args.amp and device.type == "cuda"
    run_validation = args.validate if args.validate is not None else args.overfit_samples == 0

    train_dataset = NuScenesOccupancyDataset(split="train")
    if args.depth:
        train_dataset = DepthTargetDataset(train_dataset, args.depth_cache,
                                           args.depth_min, args.depth_max)
    if args.overfit_samples > len(train_dataset):
        raise ValueError(
            f"--overfit-samples exceeds the {len(train_dataset)} available training samples."
        )
    selected_dataset = (
        Subset(train_dataset, range(args.overfit_samples))
        if args.overfit_samples > 0 else train_dataset
    )
    train_loader = make_dataloader(
        selected_dataset, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=True, seed=args.seed, pin_memory=device.type == "cuda",
    )

    class_weights = None
    class_counts = None
    if args.class_balance == "sqrt-inverse":
        class_weights, class_counts = estimate_class_weights(
            selected_dataset, train_dataset.num_classes
        )
        print(f"Training-only class counts: {class_counts.tolist()}")
        print(f"Inverse-sqrt class weights: {[round(value, 4) for value in class_weights.tolist()]}")
        class_weights = class_weights.to(device)

    val_dataset = None
    val_loader = None
    if run_validation:
        val_dataset = NuScenesOccupancyDataset(split="val")
        if not set(train_dataset.scene_names).isdisjoint(val_dataset.scene_names):
            raise ValueError("Training and validation scenes overlap.")
        val_loader = make_dataloader(
            val_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, seed=args.seed, pin_memory=device.type == "cuda",
        )

    model_config = {
        "feature_channels": 32,
        "hidden_channels": 32,
        "num_classes": train_dataset.num_classes,
        "pretrained": True,
        "grid_shape": train_dataset.grid_shape,
        "voxel_bounds": train_dataset.voxel_bounds,
        "voxel_chunk_size": 65536,
        "use_xyz": args.xyz,
        "use_depth": args.depth,
        "depth_bins": args.depth_bins,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
    }
    print(
        f"Device: {device}; AMP: {amp_enabled}; training samples: {len(selected_dataset)}; "
        f"validation: {run_validation}"
    )
    if args.overfit_samples > 0:
        print("Overfit check: training fit on a small subset; use held-out validation to assess generalization.")
    # Downloads ImageNet weights on first construction if not cached.
    model = OccupancyModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    selection_metric = "validation_loss" if run_validation else "training_loss"
    config = {
        "training": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "model": model_config,
        "loss": {
            "class_balance": args.class_balance,
            "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
            "training_class_counts": class_counts.tolist() if class_counts is not None else None,
            "validation_loss": "Unweighted masked cross-entropy for comparable evaluation.",
            "depth_loss_weight": args.depth_loss_weight if args.depth else 0.0,
            "depth_targets": "Conservative voxel-ray first-surface pseudo-depth; unknown rays masked.",
        },
        "data": {
            "dataroot": str(DEFAULT_DATAROOT),
            "occupancy_root": str(train_dataset.occupancy_root),
            "image_size": train_dataset.image_size,
            "image_mean": IMAGE_MEAN,
            "image_std": IMAGE_STD,
            "camera_order": train_dataset.camera_order,
            "train_scenes": train_dataset.scene_names,
            "validation_scenes": val_dataset.scene_names if val_dataset is not None else (),
            "train_sample_tokens": [
                record["sample_token"] for record in train_dataset.records[:len(selected_dataset)]
            ],
            "ignore_index": train_dataset.ignore_index,
        },
        "device": str(device),
        "amp_enabled": amp_enabled,
        "selection_metric": selection_metric,
        "torch_version": str(torch.__version__),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_loss = math.inf

    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_log:
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            start = time.perf_counter()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, scaler, device, amp_enabled, class_weights,
                depth_loss_weight=args.depth_loss_weight,
            )
            val_stats = (
                validate(model, val_loader, device, amp_enabled)
                if val_loader is not None else None
            )
            selected_loss = val_stats["loss"] if val_stats is not None else train_stats["loss"]
            improved = selected_loss < best_loss
            best_loss = min(best_loss, selected_loss)
            record = {
                "epoch": epoch, "train": train_stats, "validation": val_stats,
                "seconds": time.perf_counter() - start,
                "selection_metric": selection_metric, "best_loss": best_loss,
            }
            metrics_log.write(json.dumps(record, allow_nan=False) + "\n")
            metrics_log.flush()
            checkpoint_config = {**config, "metrics": record}
            save_checkpoint(
                args.output_dir / "latest.pt", model, optimizer, scaler, epoch, checkpoint_config,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt", model, optimizer, scaler, epoch, checkpoint_config,
                )
            summary = f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}"
            if val_stats is not None:
                summary += f", val_loss={val_stats['loss']:.4f}, val_accuracy={val_stats['accuracy']:.3%}"
            print(summary)

    print(f"Finished. Logs and latest/best checkpoints: {args.output_dir}")

if __name__ == "__main__":
    # Required for Windows DataLoader worker processes.
    main()
