"""Compare best/latest checkpoints on complete train and validation splits.

Example: python evaluate_run.py --project-root PATH --run-dir PATH --output-dir PATH
Uses that project's existing dataset, model and plotting helpers. No training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


def summarize(confusion, names):
    import numpy as np
    confusion = np.asarray(confusion, dtype=np.int64)
    support, predicted = confusion.sum(1), confusion.sum(0)
    tp = confusion.diagonal()
    union = support + predicted - tp
    classes = []
    for i, name in enumerate(names):
        classes.append({
            "class_id": i, "class_name": name, "gt_voxels": int(support[i]),
            "predicted_voxels": int(predicted[i]),
            "iou": float(tp[i] / union[i]) if union[i] else None,
            "recall": float(tp[i] / support[i]) if support[i] else None,
            "precision": float(tp[i] / predicted[i]) if predicted[i] else None,
        })
    occupied_tp = int(confusion[:17, :17].sum())
    occupied_gt, occupied_pred = int(support[:17].sum()), int(predicted[:17].sum())
    occupied_union = occupied_gt + occupied_pred - occupied_tp
    ious = [row["iou"] for row in classes[:17] if row["iou"] is not None]
    return {
        "valid_voxels": int(confusion.sum()),
        "voxel_accuracy": float(tp.sum() / confusion.sum()),
        "free_only_baseline_accuracy": float(support[17] / confusion.sum()),
        "occupied_class_miou": float(np.mean(ious)) if ious else None,
        "free_iou": classes[17]["iou"],
        "binary_occupied_iou": occupied_tp / occupied_union if occupied_union else None,
        "binary_occupied_recall": occupied_tp / occupied_gt if occupied_gt else None,
        "binary_occupied_precision": occupied_tp / occupied_pred if occupied_pred else None,
        "gt_occupied_voxels": occupied_gt, "predicted_occupied_voxels": occupied_pred,
        "classes": classes, "confusion_matrix": confusion.tolist(),
        "macro_average_rule": "Global confusion matrix; occupied IDs 0..16 with nonzero union, free=17, ignore=255.",
    }


def plot_validation_comparison(output, best_epoch, last_epoch):
    import numpy as np
    import matplotlib.pyplot as plt
    # A compact, directly comparable view of both validation scenes.
    from evaluate import top_down_labels
    from data.voxel_projection import PALETTE
    from matplotlib.colors import ListedColormap
    colours = np.vstack([PALETTE, [18, 23, 30]]).astype(float) / 255.
    colours[17] = [.19, .24, .29]
    cmap = ListedColormap(colours)
    examples = sorted((output / "best" / "val").glob("*/predictions.npz"))
    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 5*len(examples)), squeeze=False,
                             constrained_layout=True)
    for row, path in enumerate(examples):
        with np.load(path) as archive:
            target, valid = archive["semantics"], archive["valid_mask"]
            best_prediction = archive["prediction"]
        with np.load(output / "latest" / "val" / path.parent.name / "predictions.npz") as archive:
            latest_prediction = archive["prediction"]
        for ax, labels, title in zip(axes[row], (target, best_prediction, latest_prediction),
                                    ("Ground truth", f"Best loss: epoch {best_epoch}", f"Latest: epoch {last_epoch}")):
            ax.imshow(top_down_labels(labels, valid), origin="lower", extent=(-40, 40, -40, 40),
                      interpolation="nearest", cmap=cmap, vmin=-.5, vmax=18.5)
            ax.set_title(title)
            ax.set_xlabel("Ego Y (m)")
            ax.set_ylabel(f"{path.parent.name.split('_')[0]}\nEgo X (m)")
            ax.plot(0, 0, "w+", markersize=6)
    fig.suptitle("Validation: highest supervised occupied voxel per XY column\n"
                 "Gray = free; black = unsupervised. Metrics evaluate the full 3D volume.")
    fig.savefig(output / "validation_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--project-root", type=Path, default=root.parent)
    parser.add_argument("--run-dir", type=Path, default=root.parent / "outputs/train_xyz_full")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/train_xyz_full_evaluation")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root.resolve()))
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.nuscenes_dataset import NuScenesOccupancyDataset, make_dataloader
    from data.voxel_projection import CLASS_NAMES
    from src.models.occupancy_model import OccupancyModel
    from train import compute_loss
    from evaluate import (occupancy_metrics, save_bev_comparison,
                          save_camera_comparison, save_iou_plot)

    # Check aggregation against the existing single-frame evaluator before use.
    target = np.array([0, 17, 17, 4, 255])
    prediction = np.array([0, 17, 4, 17, 0])
    reference = occupancy_metrics(prediction, target, np.ones(5, dtype=bool))
    combined = summarize(reference["confusion_matrix"], CLASS_NAMES)
    for key in ("voxel_accuracy", "free_iou", "occupied_class_miou", "binary_occupied_iou"):
        assert combined[key] == reference[key], key

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    epochs = [r["epoch"] for r in rows]
    axes[0].plot(epochs, [r["train"]["loss"] for r in rows], label="Train: weighted CE")
    axes[0].set_title("Training objective")
    axes[1].plot(epochs, [r["validation"]["loss"] for r in rows], color="tab:orange")
    best_epoch = min(rows, key=lambda r: r["validation"]["loss"])["epoch"]
    axes[1].axvline(best_epoch, linestyle="--", color="gray", label=f"Best loss: epoch {best_epoch}")
    axes[1].set_title("Validation: unweighted CE")
    axes[1].legend()
    axes[2].plot(epochs, [100*r["validation"]["accuracy"] for r in rows], color="tab:green")
    axes[2].set_title("Validation voxel accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
    fig.suptitle("Training and validation use different loss weighting; compare each curve over time")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    config = json.loads((args.run_dir / "config.json").read_text())
    metadata = config["data"]
    datasets = {split: NuScenesOccupancyDataset(
        dataroot=metadata["dataroot"], occupancy_root=metadata["occupancy_root"],
        split=split, image_size=tuple(metadata["image_size"]),
    ) for split in ("train", "val")}
    assert set(datasets["train"].scene_names).isdisjoint(datasets["val"].scene_names)
    for dataset in datasets.values():
        assert tuple(config["model"]["grid_shape"]) == dataset.grid_shape
        assert tuple(config["model"]["voxel_bounds"]) == dataset.voxel_bounds
        assert tuple(metadata["camera_order"]) == dataset.camera_order
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; train={len(datasets['train'])}; validation={len(datasets['val'])}", flush=True)
    reports = {}
    for checkpoint_name in ("best", "latest"):
        checkpoint = torch.load(args.run_dir / f"{checkpoint_name}.pt", map_location="cpu", weights_only=True)
        model = OccupancyModel(**{**checkpoint["config"]["model"], "pretrained": False}).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        epoch = checkpoint["epoch"]
        print(f"Evaluating {checkpoint_name}, epoch {epoch}", flush=True)
        for split, dataset in datasets.items():
            destination = output / checkpoint_name / split
            destination.mkdir(parents=True, exist_ok=True)
            loader = make_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False)
            confusion = np.zeros((18, 18), dtype=np.int64)
            scene_confusion = {name: np.zeros((18, 18), dtype=np.int64) for name in dataset.scene_names}
            total_loss = total_voxels = 0
            first_scenes = set()
            per_frame = []
            for index, batch in enumerate(loader):
                scene = batch["scene_name"][0]
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits = model(*(batch[key].to(device) for key in ("images", "intrinsics", "ego_to_camera")))
                    loss = compute_loss(logits, batch["semantics"].to(device), batch["valid_mask"].to(device))
                    prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                target = batch["semantics"][0].numpy()
                valid = batch["valid_mask"][0].numpy() & (target != 255)
                if not valid.any():
                    continue
                frame = occupancy_metrics(prediction, target, valid)
                matrix = np.asarray(frame["confusion_matrix"], dtype=np.int64)
                confusion += matrix
                scene_confusion[scene] += matrix
                total_voxels += int(valid.sum())
                total_loss += float(loss.item()) * int(valid.sum())
                per_frame.append({"sample_token": batch["sample_token"][0], "scene_name": scene,
                                  **{key: frame[key] for key in ("occupied_class_miou", "binary_occupied_iou",
                                     "gt_occupied_voxels", "predicted_occupied_voxels", "voxel_accuracy")}})
                # Deterministic examples: first train frame and first frame in each val scene.
                visualize = index == 0 if split == "train" else scene not in first_scenes
                if visualize:
                    sample_dir = destination / f"{scene}_{index:03d}"
                    sample_dir.mkdir(exist_ok=True)
                    frame.update({"sample_token": batch["sample_token"][0], "scene_name": scene, "epoch": epoch})
                    (sample_dir / "metrics.json").write_text(json.dumps(frame, indent=2))
                    np.savez_compressed(sample_dir / "predictions.npz", prediction=prediction,
                                        semantics=target.astype(np.uint8), valid_mask=valid)
                    save_bev_comparison(prediction, target, valid, dataset.voxel_bounds, sample_dir / "bev_comparison.png")
                    save_camera_comparison(prediction, target, valid, dataset.records[index],
                                           dataset.camera_order, dataset.image_size, sample_dir / "camera_comparison.png")
                first_scenes.add(scene)
                if (index + 1) % 40 == 0 or index + 1 == len(dataset):
                    print(f"{checkpoint_name}/{split}: {index+1}/{len(dataset)} frames", flush=True)
            summary = summarize(confusion, CLASS_NAMES)
            summary.update({"checkpoint_epoch": epoch, "frames": len(per_frame), "split": split,
                            "unweighted_loss": total_loss / total_voxels,
                            "scenes": {scene: summarize(cm, CLASS_NAMES) for scene, cm in scene_confusion.items()}})
            (destination / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
            (destination / "per_frame.json").write_text(json.dumps(per_frame, indent=2))
            with (destination / "per_class.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary["classes"][0]))
                writer.writeheader()
                writer.writerows(summary["classes"])
            save_iou_plot(summary, destination / "per_class_iou.png")
            reports[f"{checkpoint_name}/{split}"] = {key: value for key, value in summary.items() if key not in ("scenes", "classes", "confusion_matrix")}
            print(json.dumps(reports[f"{checkpoint_name}/{split}"]), flush=True)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output / "comparison.json").write_text(json.dumps(reports, indent=2))
    plot_validation_comparison(output, best_epoch, epochs[-1])
    (output / "README.md").write_text(
        "Complete mini-split evaluation: 323 train / 81 validation frames.\n\n"
        "best.pt is selected by validation loss; latest.pt is epoch 100.\n"
        "All metrics use camera-visible, non-ignore targets; occupied IDs 0..16, free=17.\n"
        "mIoU uses a pooled confusion matrix and excludes zero-union classes.\n"
        "Visual examples are first train frame and first frame of each validation scene.\n"
        "Validation was also used for checkpoint selection; no independent test split was evaluated.\n"
        "The eight training scenes and two validation scenes are disjoint.\n"
        "Camera overlays use approximate surface visibility. No model weights were modified.\n"
    )
    print(f"Saved evaluation: {output}", flush=True)


if __name__ == "__main__":
    main()
