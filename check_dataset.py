"""Verify a real training batch and save resized camera projection checks."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch

from data.nuscenes_dataset import (
    NuScenesOccupancyDataset, make_dataloader, load_nuscenes, IMAGE_MEAN, IMAGE_STD,
)
from data.occupancy_labels import load_sample_occupancy
from data.voxel_projection import occupied_voxel_centres, camera_geometry, project_centres
from load_nuscenes import show_camera_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "training_projection.png")
    args = parser.parse_args()
    nusc = load_nuscenes(verbose=False)
    train = NuScenesOccupancyDataset(nusc=nusc, split="train")
    val = NuScenesOccupancyDataset(nusc=nusc, split="val")
    assert set(train.scene_names).isdisjoint(val.scene_names)
    assert len(train) + len(val) == len(nusc.sample)
    batch = next(iter(make_dataloader(train, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)))
    size = batch["images"].shape[0]
    assert batch["images"].shape == (size, 6, 3, 320, 576)
    assert batch["semantics"].shape == (size, 200, 200, 16)
    assert batch["semantics"].dtype == torch.int64
    assert batch["valid_mask"].dtype == torch.bool
    assert torch.equal(batch["valid_mask"], batch["mask_camera"] & (batch["semantics"] != 255))
    assert torch.isfinite(batch["images"]).all()
    assert (batch["valid_mask"] & (batch["semantics"] == 17)).any()
    assert (batch["valid_mask"] & (batch["semantics"] < 17)).any()
    sample = nusc.get("sample", batch["sample_token"][0])
    occupancy = load_sample_occupancy(nusc, sample["token"])
    points, labels = occupied_voxel_centres(occupancy)
    original_geometry = camera_geometry(nusc, sample)
    images, projections, max_errors = {}, {}, []
    for i, channel in enumerate(train.camera_order):
        rgb = batch["images"][0, i].permute(1, 2, 0).numpy()
        rgb = rgb * np.array(IMAGE_STD) + np.array(IMAGE_MEAN)
        images[channel] = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
        geometry = original_geometry[channel]
        native_camera = points @ geometry["ego_to_camera"][:3, :3].T + geometry["ego_to_camera"][:3, 3]
        keep = native_camera[:, 2] > 0.1
        native_uv = native_camera[keep] @ geometry["intrinsics"].T
        native_uv = native_uv[:, :2] / native_uv[:, 2:3]
        scale = np.array([576, 320]) / np.array(geometry["image_size"])
        expected_uv = (native_uv + 0.5) * scale - 0.5
        transform = batch["ego_to_camera"][0, i].numpy()
        intrinsics = batch["intrinsics"][0, i].numpy()
        camera = points[keep] @ transform[:3, :3].T + transform[:3, 3]
        uv = camera @ intrinsics.T
        uv = uv[:, :2] / uv[:, 2:3]
        inside = (expected_uv[:, 0] >= 0) & (expected_uv[:, 0] < 576) & (expected_uv[:, 1] >= 0) & (expected_uv[:, 1] < 320)
        error = float(np.max(np.abs(uv[inside] - expected_uv[inside]))) if inside.any() else 0.
        assert error < 0.01, (channel, error)
        max_errors.append(error)
        projections[channel] = project_centres(points, labels, transform, intrinsics, (576, 320))
    item = {"images": images, "voxel_projections": projections, "sample": sample,
            "scene": nusc.get("scene", sample["scene_token"])}
    show_camera_frames(item, save_path=args.output, show=False)
    gpu = None
    if torch.cuda.is_available():
        device_images = batch["images"].to("cuda")
        assert torch.isfinite(device_images).all().item()
        gpu = torch.cuda.get_device_name(0)
        del device_images
    report = {
        "train_samples": len(train), "val_samples": len(val),
        "train_scenes": train.scene_names, "val_scenes": val.scene_names,
        "batch_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
        "max_resize_projection_error_pixels": max(max_errors),
        "valid_voxels": int(batch["valid_mask"].sum()),
        "supervised_free_voxels": int((batch["valid_mask"] & (batch["semantics"] == 17)).sum()),
        "workers": args.workers, "torch": torch.__version__, "gpu": gpu,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
