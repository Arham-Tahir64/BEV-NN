"""Evaluate a checkpoint on one frame and compare predictions with ground truth."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import torch

from data import NuScenesOccupancyDataset
from data.voxel_projection import (
    CLASS_NAMES, LEGEND, PALETTE, draw_voxel_overlay,
    occupied_voxel_centres, project_centres,
)
from src.models.occupancy_model import OccupancyModel
from train import compute_loss


def occupancy_metrics(prediction, target, valid_mask):
    """Diagnostic metrics on supervised voxels, including occupied class zero.

    Confusion rows are target classes; columns are predicted classes. Classes
    with zero union have undefined IoU and are excluded from macro averages.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("Prediction, target and mask must have matching shapes.")
    keep = valid_mask & (target != 255)
    if not keep.any():
        raise ValueError("No supervised voxels to evaluate.")
    actual = target[keep].astype(np.int64)
    predicted = prediction[keep].astype(np.int64)
    if np.any((actual < 0) | (actual >= 18) | (predicted < 0) | (predicted >= 18)):
        raise ValueError("Expected class IDs 0..17 on evaluated voxels.")
    confusion = np.bincount(actual * 18 + predicted, minlength=18*18).reshape(18,18)
    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    tp = confusion.diagonal()
    union = support + predicted_counts - tp
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        classes.append({
            "class_id": i, "class_name": name,
            "gt_voxels": int(support[i]), "predicted_voxels": int(predicted_counts[i]),
            "true_positive": int(tp[i]),
            "false_positive": int(predicted_counts[i] - tp[i]),
            "false_negative": int(support[i] - tp[i]),
            "iou": float(tp[i] / union[i]) if union[i] else None,
            "precision": float(tp[i] / predicted_counts[i]) if predicted_counts[i] else None,
            "recall": float(tp[i] / support[i]) if support[i] else None,
        })
    occupied_ious = [row["iou"] for row in classes[:17] if row["iou"] is not None]
    all_ious = [row["iou"] for row in classes if row["iou"] is not None]
    occupied_tp = int(confusion[:17,:17].sum())
    occupied_fp = int(confusion[17,:17].sum())
    occupied_fn = int(confusion[:17,17].sum())
    occupied_union = occupied_tp + occupied_fp + occupied_fn
    return {
        "valid_voxels": int(keep.sum()),
        "voxel_accuracy": float(tp.sum() / keep.sum()),
        "free_only_baseline_accuracy": float(support[17] / keep.sum()),
        "occupied_class_miou": float(np.mean(occupied_ious)) if occupied_ious else None,
        "all_class_miou": float(np.mean(all_ious)),
        "free_iou": classes[17]["iou"],
        "binary_occupied_iou": occupied_tp / occupied_union if occupied_union else None,
        "gt_occupied_voxels": int(support[:17].sum()),
        "predicted_occupied_voxels": int(predicted_counts[:17].sum()),
        "macro_average_rule": "Nonzero-union classes only; occupied semantic classes are IDs 0..16.",
        "classes": classes, "confusion_matrix": confusion.tolist(),
    }


def top_down_labels(labels, valid):
    """Highest supervised occupied voxel per XY column; free=17, unknown=18."""
    heights = np.arange(labels.shape[2])[None,None,:]
    occupied = valid & (labels < 17)
    top = np.max(np.where(occupied, heights, -1), axis=2)
    selected = np.take_along_axis(labels, np.maximum(top, 0)[...,None], axis=2)[...,0]
    return np.where(top >= 0, selected, np.where(valid.any(axis=2), 17, 18))


def save_bev_comparison(prediction, target, valid, bounds, destination):
    colours = np.vstack([PALETTE, [18,23,30]]).astype(float) / 255.
    colours[17] = [0.19,0.24,0.29]
    cmap = ListedColormap(colours)
    xmin, ymin, _, xmax, ymax, _ = bounds
    # Horizontal axis is ego Y (left); vertical is ego X (forward).
    extent = (ymin, ymax, xmin, xmax)
    fig, axes = plt.subplots(1,3, figsize=(16,6), constrained_layout=True)
    for ax, labels, title in zip(
        axes[:2], (target, prediction), ("Ground truth", "Prediction")
    ):
        ax.imshow(top_down_labels(labels, valid), origin="lower", extent=extent,
                  interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=18.5)
        ax.set_title(title)
    counts = valid.sum(axis=2)
    accuracy = np.divide(((prediction == target) & valid).sum(axis=2), counts,
                         out=np.zeros_like(counts, dtype=float), where=counts > 0)
    image = axes[2].imshow(np.ma.masked_where(counts == 0, accuracy),
                           origin="lower", extent=extent, vmin=0, vmax=1, cmap="viridis")
    axes[2].set_title("3D voxel accuracy per XY column")
    if not np.any((prediction < 17) & valid):
        axes[1].set_title("Prediction: all supervised voxels are free")
    fig.colorbar(image, ax=axes[2], label="Fraction correct", shrink=0.75)
    for ax in axes:
        ax.set_xlabel("Ego Y: positive = left (m)")
        ax.set_ylabel("Ego X: positive = forward (m)")
        ax.plot(0,0, "w+", markersize=8)
    handles = [Line2D([],[], marker="s", linestyle="", color=PALETTE[c]/255., label=name)
               for name,c in LEGEND]
    handles.extend([
        Line2D([],[], marker="s", linestyle="", color=colours[17], label="Free"),
        Line2D([],[], marker="s", linestyle="", color=colours[18], label="Unsupervised"),
    ])
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Ground truth vs prediction: highest supervised occupied voxel per XY column\n"
                 "Metrics use all supervised 3D voxels, not just this top-down projection")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_iou_plot(metrics, destination):
    values = [row["iou"] * 100 if row["iou"] is not None else 0 for row in metrics["classes"]]
    fig, ax = plt.subplots(figsize=(11,7), constrained_layout=True)
    ax.barh(np.arange(18), values, color=PALETTE / 255.)
    ax.set_yticks(np.arange(18), labels=CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0,105)
    ax.set_xlabel("IoU (%) on supervised voxels")
    for i, row in enumerate(metrics["classes"]):
        value = values[i]
        text = f"{value:.1f}%" if row["iou"] is not None else "N/A (zero union)"
        ax.text(value + 1, i, text, va="center", fontsize=8)
    ax.set_title("Per-class IoU on the evaluated frame\n"
                 "N/A classes are excluded from macro averages")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_camera_comparison(prediction, target, valid, record, camera_order, image_size, destination):
    height, width = image_size
    projections = []
    for labels in (target, prediction):
        masked_labels = np.where(valid, labels, 255).astype(np.uint8)
        projections.append(occupied_voxel_centres({
            "semantics": masked_labels, "mask_camera": valid
        }))
    order = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
    fig, axes = plt.subplots(6,2, figsize=(12,22), facecolor="#171d23")
    fig.subplots_adjust(left=.015, right=.985, top=.95, bottom=.045, hspace=.15, wspace=.025)
    for row, channel in enumerate(order):
        index = camera_order.index(channel)
        with Image.open(record["image_paths"][index]) as original:
            image = original.convert("RGB").resize((width,height), Image.Resampling.BILINEAR)
        for column, (centres, labels) in enumerate(projections):
            projected = project_centres(centres, labels, record["ego_to_camera"][index],
                                        record["intrinsics"][index], (width,height))
            overlay = draw_voxel_overlay(image, projected, radius=1, spacing=16)
            ax = axes[row,column]
            ax.imshow(overlay)
            label = "Ground truth" if column == 0 else "Prediction"
            if column == 1 and not np.any((prediction < 17) & valid):
                label += " (no occupied voxels)"
            ax.set_title(f"{channel} | {label}",
                         loc="left", fontsize=10, color="white")
            ax.axis("off")
    handles = [Line2D([],[], marker="o", linestyle="", color=PALETTE[c]/255., label=name)
               for name,c in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False, labelcolor="white")
    fig.suptitle("Ground truth vs predicted occupied voxel centres\n"
                 "Same supervised region; free/ignore omitted; approximate surface projection",
                 fontsize=14, color="white")
    fig.savefig(destination, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "outputs/overfit/best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train","val"), default="train")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-token")
    selection.add_argument("--sample-index", type=int)
    parser.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    metadata = config["data"]
    dataset = NuScenesOccupancyDataset(
        dataroot=metadata["dataroot"], occupancy_root=metadata["occupancy_root"],
        split=args.split, image_size=tuple(metadata["image_size"]),
    )
    if tuple(config["model"]["grid_shape"]) != dataset.grid_shape:
        raise ValueError("Checkpoint and annotation grid shapes differ.")
    if tuple(config["model"]["voxel_bounds"]) != dataset.voxel_bounds:
        raise ValueError("Checkpoint and annotation grid bounds differ.")
    if tuple(metadata["camera_order"]) != dataset.camera_order:
        raise ValueError("Checkpoint and dataset camera orders differ.")
    token = args.sample_token
    if token is None and args.sample_index is None and args.split == "train":
        token = metadata["train_sample_tokens"][0]
    if token is not None:
        indices = [i for i,r in enumerate(dataset.records) if r["sample_token"] == token]
        if not indices:
            parser.error(f"Sample {token} is not in the selected {args.split} split.")
        index = indices[0]
    else:
        index = args.sample_index if args.sample_index is not None else 0
    if not 0 <= index < len(dataset):
        parser.error("Sample index is outside the selected split.")
    item = dataset[index]
    model = OccupancyModel(**{**config["model"], "pretrained": False}).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"
    ):
        logits = model(item["images"].unsqueeze(0).to(device),
                       item["intrinsics"].unsqueeze(0).to(device),
                       item["ego_to_camera"].unsqueeze(0).to(device))
        loss = compute_loss(logits, item["semantics"].unsqueeze(0).to(device),
                            item["valid_mask"].unsqueeze(0).to(device))
        if loss is None:
            raise ValueError("Selected frame has no supervision.")
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        loss_value = float(loss.item())
    target = item["semantics"].numpy()
    valid = item["valid_mask"].numpy() & (target != 255)
    metrics = occupancy_metrics(prediction, target, valid)
    metrics.update({
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
        "sample_token": item["sample_token"], "scene_name": item["scene_name"],
        "split": args.split,
        "is_training_sample": item["sample_token"] in metadata["train_sample_tokens"],
        "loss": loss_value,
        "evaluation_scope": "One frame; supervised voxels only. Training-frame fit is not held-out performance.",
    })
    output = (args.output_dir or checkpoint_path.parent / "evaluation").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics["classes"][0]))
        writer.writeheader()
        writer.writerows(metrics["classes"])
    np.savez_compressed(output / "predictions.npz", prediction=prediction,
                        semantics=target.astype(np.uint8), valid_mask=valid,
                        sample_token=np.array(item["sample_token"]))
    save_bev_comparison(prediction, target, valid, dataset.voxel_bounds, output / "bev_comparison.png")
    save_iou_plot(metrics, output / "per_class_iou.png")
    save_camera_comparison(prediction, target, valid, dataset.records[index],
                           dataset.camera_order, dataset.image_size, output / "camera_comparison.png")
    for row in metrics["classes"]:
        score = f"{100 * row['iou']:.2f}%" if row["iou"] is not None else "N/A"
        print(f"{row['class_name']:22s} IoU={score:>8s} GT={row['gt_voxels']:6d} predicted={row['predicted_voxels']:6d}")
    summary = {key:metrics[key] for key in (
        "checkpoint_epoch", "sample_token", "valid_voxels", "voxel_accuracy",
        "free_only_baseline_accuracy", "occupied_class_miou", "free_iou",
        "binary_occupied_iou", "gt_occupied_voxels", "predicted_occupied_voxels", "loss"
    )}
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
