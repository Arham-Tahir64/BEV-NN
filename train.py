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
from src.metrics import summarize_confusion, occupancy_boundary_counts, summarize_boundary_counts
from src.losses import occupied_free_loss, boundary_occupied_free_loss
from src.semantic_supervision import semantic_dice_loss, bev_auxiliary_loss
from data.scene_selection import read_scene_list


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
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                        default=(320, 576), help='Camera input size; try 480 864 for finer image features.')
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
    parser.add_argument('--architecture', choices=('baseline', 'bevdepth'), default='baseline',
                        help='bevdepth enables camera-aware DepthNet, frustum refinement, forward pooling, '
                             'a trainable image FPN and a multiscale voxel decoder.')
    parser.add_argument("--depth-bins", type=int, default=64)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=64.1)
    parser.add_argument("--depth-loss-weight", type=float, default=None)
    parser.add_argument("--depth-cache", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "depth_cache")
    parser.add_argument('--depth-target-mode', choices=('ray', 'projected', 'lidar'), default='projected')
    parser.add_argument('--depth-fusion', choices=('legacy', 'residual', 'gated', 'probability', 'bevdepth'),
                        default=None,
                        help='New --depth runs default to probability: joint probability-weighted lifting, '
                             'no raw appearance bypass or clipping. Other modes reproduce older runs.')
    parser.add_argument('--depth-loss-mode', choices=('hard', 'linear', 'bce'), default=None)
    parser.add_argument('--geometry-loss-weight', type=float, default=0.0,
                        help='Weight of unweighted occupied/free BCE on existing logits; 0 disables it.')
    parser.add_argument('--boundary-loss-weight', type=float, default=0.0,
                        help='Extra occupied/free BCE on observed six-neighbour boundary endpoints.')
    parser.add_argument('--semantic-dice-weight', type=float, default=0.0,
                        help='Macro soft Dice on target-present occupied classes; 0 disables it.')
    parser.add_argument('--bev-aux-weight', type=float, default=0.0,
                        help='Training-only BEV semantic CE head (bevdepth only); 0 disables it.')
    parser.add_argument('--dataset-version', choices=('v1.0-mini','v1.0-trainval'), default='v1.0-mini')
    parser.add_argument('--dataroot', type=Path, default=DEFAULT_DATAROOT,
                        help='nuScenes root containing the selected metadata version.')
    parser.add_argument('--occupancy-root', type=Path, help='Occ3D gts directory.')
    parser.add_argument('--image-root', type=Path,
                        help='Optional keyframe images root containing CAM_* folders (e.g. Occ3D imgs).')
    parser.add_argument('--annotations-json', type=Path,
                        help='Official Occ3D annotations.json; supplies camera/pose metadata directly.')
    parser.add_argument('--train-scenes', type=Path, help='JSON scene-name list within the official training split.')
    parser.add_argument('--val-scenes', type=Path, help='JSON scene-name list within the official validation split.')
    args = parser.parse_args()
    if args.depth_fusion is None:
        args.depth_fusion = 'bevdepth' if args.architecture == 'bevdepth' else ('probability' if args.depth else 'legacy')
    if args.depth_loss_mode is None:
        args.depth_loss_mode = 'bce' if args.architecture == 'bevdepth' else 'linear'
    if args.depth_loss_weight is None:
        args.depth_loss_weight = 3.0 if args.architecture == 'bevdepth' else 1.0
    if args.architecture == 'bevdepth' and (not args.depth or args.depth_fusion != 'bevdepth'):
        parser.error('--architecture bevdepth requires --depth and --depth-fusion bevdepth.')
    if args.architecture != 'bevdepth' and args.depth_fusion == 'bevdepth':
        parser.error('--depth-fusion bevdepth requires --architecture bevdepth.')
    if args.depth_fusion in ('gated', 'probability') and not args.depth:
        parser.error(f'--depth-fusion {args.depth_fusion} requires --depth.')
    if (args.depth_bins <= 0 or not math.isfinite(args.depth_min)
            or not math.isfinite(args.depth_max) or not 0 < args.depth_min < args.depth_max):
        parser.error("Depth bins must be positive and bounds finite with 0 < min < max.")
    if not math.isfinite(args.depth_loss_weight) or args.depth_loss_weight <= 0:
        parser.error("--depth-loss-weight must be finite and positive.")
    if not math.isfinite(args.geometry_loss_weight) or args.geometry_loss_weight < 0:
        parser.error('--geometry-loss-weight must be finite and nonnegative.')
    if not math.isfinite(args.boundary_loss_weight) or args.boundary_loss_weight < 0:
        parser.error('--boundary-loss-weight must be finite and nonnegative.')
    if any(size <= 0 for size in args.image_size):
        parser.error('--image-size requires positive height and width.')
    for name in ('semantic_dice_weight', 'bev_aux_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and nonnegative.')
    if args.bev_aux_weight > 0 and args.architecture != 'bevdepth':
        parser.error('--bev-aux-weight requires --architecture bevdepth.')

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
    depth_loss_mode: str = 'hard',
    geometry_loss_weight: float = 0.0,
    boundary_loss_weight: float = 0.0,
    semantic_dice_weight: float = 0.0,
    bev_aux_weight: float = 0.0,
) -> dict[str, float]:
    """Update weights and average epoch loss by its target-weight denominator.

    The scaler should be enabled only for CUDA mixed precision. Batches without
    supervised voxels are skipped; an entirely unsupervised/empty epoch raises
    RuntimeError rather than reporting a misleading zero loss.
    """
    model.train()  # ImageEncoder.train() keeps its frozen backbone in eval mode.
    for name, weight in (('semantic_dice_weight', semantic_dice_weight), ('bev_aux_weight', bev_aux_weight)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'{name} must be finite and nonnegative.')
    if bev_aux_weight > 0 and getattr(model, 'bev_aux_head', None) is None:
        raise ValueError('BEV auxiliary loss requires a model constructed with use_bev_aux=True.')
    if not math.isfinite(geometry_loss_weight) or geometry_loss_weight < 0:
        raise ValueError('geometry_loss_weight must be finite and nonnegative.')
    if not math.isfinite(boundary_loss_weight) or boundary_loss_weight < 0:
        raise ValueError('boundary_loss_weight must be finite and nonnegative.')
    use_amp = amp_enabled and device.type == "cuda"
    total_loss = 0.0
    total_voxels = 0
    total_loss_weight = 0.0
    processed_batches = 0
    skipped_batches = 0
    depth_enabled = getattr(model, "depth_head", None) is not None
    depth_loss_sum = 0.0
    depth_count = 0
    geometry_sum = 0.0
    geometry_count = 0
    boundary_sum = 0.0
    boundary_count = 0
    dice_sum = bev_sum = 0.0
    dice_count = bev_count = bev_possible = 0

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
            if depth_enabled or bev_aux_weight > 0:
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

        if geometry_loss_weight > 0:
            geometry, count = occupied_free_loss(logits, semantics, valid_mask)
            if not torch.isfinite(geometry):
                raise FloatingPointError('Non-finite occupied/free loss.')
            geometry_sum += geometry.detach().item()*count
            geometry_count += count
            loss = loss + geometry_loss_weight*geometry

        if semantic_dice_weight > 0:
            dice, count = semantic_dice_loss(logits, semantics, valid_mask)
            if not torch.isfinite(dice):
                raise FloatingPointError('Non-finite semantic Dice loss.')
            dice_sum += dice.detach().item()*count
            dice_count += count
            loss = loss + semantic_dice_weight*dice

        if bev_aux_weight > 0:
            bev, count = bev_auxiliary_loss(auxiliary['bev_logits'], semantics, valid_mask)
            if not torch.isfinite(bev):
                raise FloatingPointError('Non-finite BEV auxiliary loss.')
            bev_sum += bev.detach().item()*count
            bev_count += count
            bev_possible += semantics.numel() // semantics.shape[-1]
            loss = loss + bev_aux_weight*bev

        if boundary_loss_weight > 0:
            boundary, count = boundary_occupied_free_loss(logits, semantics, valid_mask)
            if not torch.isfinite(boundary):
                raise FloatingPointError('Non-finite occupied/free boundary loss.')
            boundary_sum += boundary.detach().item()*count
            boundary_count += count
            loss = loss + boundary_loss_weight*boundary

        if depth_enabled:
            if "depth_target" not in batch:
                raise ValueError("Depth training requires cached depth_target from DepthTargetDataset.")
            auxiliary_loss, count = depth_loss(auxiliary["depth_logits"],
                batch["depth_target"].to(device), model.depth_head.bin_edges, mode=depth_loss_mode)
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
    if bev_aux_weight > 0 and bev_count == 0:
        raise RuntimeError('No trustworthy BEV targets in this epoch. Inspect visibility/height coverage.')
    return {
        "loss": total_loss / total_loss_weight,
        **({'semantic_dice_loss': dice_sum/dice_count if dice_count else 0.0,
            'semantic_dice_classes': dice_count} if semantic_dice_weight > 0 else {}),
        **({'bev_aux_loss': bev_sum/bev_count, 'bev_aux_cells': bev_count,
            'bev_aux_coverage': bev_count/bev_possible} if bev_aux_weight > 0 else {}),
        **({"depth_loss": depth_loss_sum / depth_count, "depth_targets": depth_count}
           if depth_enabled else {}),
        **({'geometry_loss': geometry_sum/geometry_count, 'geometry_voxels': geometry_count}
           if geometry_count else {}),
        **({'boundary_loss': boundary_sum/boundary_count if boundary_count else 0.0,
            'boundary_voxels': boundary_count} if boundary_loss_weight > 0 else {}),
        "valid_voxels": total_voxels,
        "batches": processed_batches,
        "skipped_batches": skipped_batches,
    }


def validate(
    model: OccupancyModel,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    depth_loss_mode: str = 'hard',
) -> dict[str, float]:
    """Return held-out masked loss and voxel accuracy without updating weights."""
    model.eval()
    use_amp = amp_enabled and device.type == "cuda"
    total_loss = 0.0
    total_voxels = 0
    correct_voxels = 0
    processed_batches = 0
    skipped_batches = 0
    confusion = torch.zeros(18, 18, dtype=torch.int64, device=device)
    boundary_counts = torch.zeros(3, dtype=torch.int64, device=device)
    boundary_loss_sum = 0.0
    boundary_loss_count = 0
    depth_sum = depth_abs = 0.0
    depth_count = depth_possible = 0
    range_counts = [0, 0, 0]
    range_abs = [0.0, 0.0, 0.0]
    geometry_sum = 0.0

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
                measure_depth = getattr(model, 'depth_head', None) is not None and 'depth_target' in batch
                if measure_depth:
                    auxiliary = model(images, intrinsics, ego_to_camera, return_aux=True)
                    logits = auxiliary['occupancy_logits']
                else:
                    logits = model(images, intrinsics, ego_to_camera)
                loss = compute_loss(logits, semantics, valid_mask)
            if loss is None:
                skipped_batches += 1
                continue
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite validation loss in batch {batch_index}: {loss_value}")

            total_loss += loss_value * valid_voxels
            geometry, _ = occupied_free_loss(logits, semantics, valid_mask)
            geometry_sum += geometry.item()*valid_voxels
            total_voxels += valid_voxels
            prediction = logits.argmax(dim=1)
            boundary_counts += occupancy_boundary_counts(prediction, semantics, supervised)
            boundary, boundary_n = boundary_occupied_free_loss(logits, semantics, supervised)
            boundary_loss_sum += boundary.item()*boundary_n
            boundary_loss_count += boundary_n
            correct_voxels += int(((prediction == semantics) & supervised).sum().item())
            confusion += torch.bincount(semantics[supervised]*18 + prediction[supervised],
                                        minlength=324).reshape(18, 18)
            if measure_depth:
                targets = batch['depth_target'].to(device)
                edges = model.depth_head.bin_edges
                auxiliary_loss, count = depth_loss(auxiliary['depth_logits'], targets, edges, mode=depth_loss_mode)
                depth_sum += auxiliary_loss.item()*count
                depth_count += count
                depth_possible += targets.numel()
                known = torch.isfinite(targets) & (targets >= edges[0]) & (targets < edges[-1])
                depths = model.depth_head.bin_centres[auxiliary['depth_logits'].argmax(2)]
                error = (depths-targets).abs()
                depth_abs += error[known].sum().item()
                for ri, (lo, hi) in enumerate(((0, 10), (10, 20), (20, float('inf')))):
                    keep = known & (targets >= lo) & (targets < hi)
                    range_counts[ri] += int(keep.sum().item())
                    range_abs[ri] += error[keep].sum().item()
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
        'geometry_loss': geometry_sum/total_voxels,
        **summarize_confusion(confusion),
        **summarize_boundary_counts(boundary_counts),
        'boundary_loss': boundary_loss_sum/boundary_loss_count if boundary_loss_count else None,
        'boundary_loss_voxels': boundary_loss_count,
        **({'depth_loss': depth_sum/depth_count if depth_count else None,
            'pseudo_depth_mae_m': depth_abs/depth_count if depth_count else None,
            'depth_targets': depth_count, 'depth_target_coverage': depth_count/depth_possible,
            'depth_range_counts_0_10_20_plus': range_counts,
            'depth_range_mae_m': [e/n if n else None for e, n in zip(range_abs, range_counts)]}
           if depth_possible else {}),
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
    # Do not overwrite an interrupted or finished experiment accidentally.
    if any((args.output_dir / name).exists() for name in ('config.json', 'metrics.jsonl', 'latest.pt', 'best.pt')):
        raise FileExistsError('Run directory already exists. Use resume_train.py or a new --output-dir.')
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

    dataset_options = dict(dataroot=args.dataroot, occupancy_root=args.occupancy_root,
                           version=args.dataset_version, image_root=args.image_root,
                           annotations_json=args.annotations_json,
                           image_size=tuple(getattr(args, 'image_size', (320, 576))))
    print(f'Indexing {args.dataset_version} training scenes...',flush=True)
    train_dataset = NuScenesOccupancyDataset(split="train", **dataset_options,
                                            scene_names=read_scene_list(args.train_scenes))
    if args.depth:
        train_dataset = DepthTargetDataset(train_dataset, args.depth_cache,
                                           args.depth_min, args.depth_max, args.depth_target_mode)
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
        print(f'Indexing {args.dataset_version} validation scenes...',flush=True)
        val_dataset = NuScenesOccupancyDataset(split="val", **dataset_options,
                                              scene_names=read_scene_list(args.val_scenes))
        if args.depth:
            val_dataset = DepthTargetDataset(val_dataset, args.depth_cache,
                args.depth_min, args.depth_max, args.depth_target_mode)
        if not set(train_dataset.scene_names).isdisjoint(val_dataset.scene_names):
            raise ValueError("Training and validation scenes overlap.")
        val_loader = make_dataloader(
            val_dataset, batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False, seed=args.seed, pin_memory=device.type == "cuda",
        )

    model_config = {
        "architecture": getattr(args, 'architecture', 'baseline'),
        "feature_channels": 32,
        "hidden_channels": 32,
        "num_classes": train_dataset.num_classes,
        "pretrained": True,
        "grid_shape": train_dataset.grid_shape,
        "voxel_bounds": train_dataset.voxel_bounds,
        "voxel_chunk_size": 65536,
        "use_xyz": args.xyz,
        "use_depth": args.depth,
        "use_bev_aux": getattr(args, 'bev_aux_weight', 0.0) > 0,
        "depth_bins": args.depth_bins,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "depth_fusion": args.depth_fusion,
    }
    print(
        f"Dataset: {args.dataset_version}; training scenes: {len(train_dataset.scene_names)}; "
        f"Device: {device}; AMP: {amp_enabled}; training samples: {len(selected_dataset)}; "
        f"validation: {run_validation}"
    )
    print(f'Image size: {train_dataset.image_size}; geometry BCE weight: {args.geometry_loss_weight}; '
          f'boundary BCE weight: {getattr(args, "boundary_loss_weight", 0.0)}', flush=True)
    if args.overfit_samples > 0:
        print("Overfit check: training fit on a small subset; use held-out validation to assess generalization.")
    # Downloads ImageNet weights on first construction if not cached.
    model = OccupancyModel(**model_config).to(device)
    if model_config['architecture'] == 'bevdepth':
        print(f'BEVDepth occupancy adaptation: {args.depth_target_mode} depth targets, '
              f'{args.depth_loss_mode} depth loss, weight {args.depth_loss_weight}.', flush=True)
        if args.depth_target_mode != 'lidar':
            print('Depth uses occupancy-derived pseudo-labels; this is not the paper\'s measured LiDAR supervision.', flush=True)
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
            'semantic_dice_weight': getattr(args, 'semantic_dice_weight', 0.0),
            'semantic_dice': 'Macro soft Dice over target-present occupied classes; observed free voxels penalize false positives.',
            'bev_aux_weight': getattr(args, 'bev_aux_weight', 0.0),
            'bev_aux': 'Unweighted topmost-surface semantic CE; unknown above surface excluded; free requires entire column known.',
            "class_balance": args.class_balance,
            "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
            "training_class_counts": class_counts.tolist() if class_counts is not None else None,
            "validation_loss": "Unweighted masked cross-entropy for comparable evaluation.",
            "depth_loss_weight": args.depth_loss_weight if args.depth else 0.0,
            "depth_targets": args.depth_target_mode,
            "depth_loss_mode": args.depth_loss_mode,
            'geometry_loss_weight': args.geometry_loss_weight,
            'geometry_loss': 'Unweighted observed-voxel occupied/free BCE; free=17, unknown ignored.',
            'boundary_loss_weight': getattr(args, 'boundary_loss_weight', 0.0),
            'boundary_loss': 'Mean occupied/free BCE over both endpoints of observed six-neighbour '
                             'occupied/free transitions; invisible/unknown neighbours excluded.',
        },
        "data": {
            "dataroot": str(args.dataroot.expanduser().resolve()),
            'version': args.dataset_version,
            'image_root': str(args.image_root.expanduser().resolve()) if args.image_root is not None else None,
            'annotations_json': str(args.annotations_json.expanduser().resolve()) if args.annotations_json is not None else None,
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
    best_semantic_miou = -1.0

    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_log:
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            start = time.perf_counter()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, scaler, device, amp_enabled, class_weights,
                depth_loss_weight=args.depth_loss_weight,
                depth_loss_mode=args.depth_loss_mode,
                geometry_loss_weight=args.geometry_loss_weight,
                boundary_loss_weight=getattr(args, 'boundary_loss_weight', 0.0),
                semantic_dice_weight=getattr(args, 'semantic_dice_weight', 0.0),
                bev_aux_weight=getattr(args, 'bev_aux_weight', 0.0),
            )
            val_stats = (
                validate(model, val_loader, device, amp_enabled, depth_loss_mode=args.depth_loss_mode)
                if val_loader is not None else None
            )
            selected_loss = val_stats["loss"] if val_stats is not None else train_stats["loss"]
            improved = selected_loss < best_loss
            best_loss = min(best_loss, selected_loss)
            semantic_improved = val_stats is not None and val_stats['occupied_class_miou'] > best_semantic_miou
            if semantic_improved:
                best_semantic_miou = val_stats['occupied_class_miou']
            record = {
                "epoch": epoch, "train": train_stats, "validation": val_stats,
                "seconds": time.perf_counter() - start,
                "selection_metric": selection_metric, "best_loss": best_loss,
                "best_semantic_miou": best_semantic_miou,
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
            if semantic_improved:
                save_checkpoint(args.output_dir / 'best_semantic.pt', model, optimizer,
                                scaler, epoch, checkpoint_config)
            summary = f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}"
            if 'geometry_loss' in train_stats:
                summary += f", geometry_BCE={train_stats['geometry_loss']:.4f}"
            if 'boundary_loss' in train_stats:
                summary += f", boundary_BCE={train_stats['boundary_loss']:.4f}"
            if 'semantic_dice_loss' in train_stats:
                summary += f", semantic_Dice={train_stats['semantic_dice_loss']:.4f}"
            if 'bev_aux_loss' in train_stats:
                summary += f", BEV_CE={train_stats['bev_aux_loss']:.4f}, BEV_coverage={train_stats['bev_aux_coverage']:.1%}"
            if val_stats is not None:
                summary += f", val_loss={val_stats['loss']:.4f}, val_accuracy={val_stats['accuracy']:.3%}"
                summary += (f", occ_mIoU={val_stats['occupied_class_miou']:.3%}, "
                            f"manmade_IoU={val_stats['manmade_iou']:.3%}, "
                            f"occupied_P/R={val_stats['occupied_precision']:.3%}/{val_stats['occupied_recall']:.3%}")
                if val_stats['missing_supported_classes']:
                    summary += f", MISSING_CLASS_IDS={val_stats['missing_supported_classes']}"
                if val_stats.get('boundary_f1') is not None:
                    summary += f", boundary_F1={val_stats['boundary_f1']:.3%}"
            print(summary)

    print(f"Finished. Logs and latest/best checkpoints: {args.output_dir}")

if __name__ == "__main__":
    # Required for Windows DataLoader worker processes.
    main()
