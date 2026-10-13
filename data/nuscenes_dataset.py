"""Load nuScenes metadata, six-camera keyframes, and optional occupancy labels."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes

DEFAULT_DATAROOT = Path(__file__).resolve().parent / "nuscenes"
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def load_nuscenes(
    dataroot: str | Path = DEFAULT_DATAROOT,
    version: str = "v1.0-mini",
    verbose: bool = True,
) -> NuScenes:
    """Load the official devkit database; dataroot contains samples and metadata."""
    root = Path(dataroot).expanduser().resolve()
    if not (root / version / "scene.json").is_file():
        raise FileNotFoundError(
            f"Missing nuScenes metadata: {root / version / 'scene.json'}. "
            "Extract the dataset into dataroot, so it contains the version, "
            "samples, sweeps, and maps folders."
        )
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=str(root), verbose=verbose)


def load_sample(
    nusc: NuScenes,
    sample_index: int = 0,
    camera: str = "CAM_FRONT",
    occupancy_root: str | Path | None = None,
) -> dict:
    """Read one keyframe's six RGB images and 3D annotations.

    images maps each camera name to a uint8 (height, width, 3) RGB array.
    image retains the selected camera for single-camera callers.
    Pass occupancy_root (the Occ3D gts folder) to attach voxel labels and masks.
    Annotations remain in the nuScenes global frame; no coordinate
    transformation or projection is applied.
    """
    if not 0 <= sample_index < len(nusc.sample):
        raise IndexError(f"sample_index must be between 0 and {len(nusc.sample) - 1}.")
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}; choose one of {CAMERAS}.")

    import numpy as np
    from PIL import Image

    sample = nusc.sample[sample_index]
    image_paths = {
        channel: Path(nusc.get_sample_data_path(sample["data"][channel]))
        for channel in CAMERAS
    }
    images = {}
    for channel, path in image_paths.items():
        with Image.open(path) as image:
            images[channel] = np.array(image.convert("RGB"))
    image_path = image_paths[camera]
    rgb = images[camera]
    annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]

    item = {
        "sample": sample,
        "scene": nusc.get("scene", sample["scene_token"]),
        "image": rgb,
        "images": images,
        "image_paths": image_paths,
        "annotations": annotations,
        "image_path": image_path,
    }
    if occupancy_root is not None:
        from .occupancy_labels import load_sample_occupancy
        item["occupancy"] = load_sample_occupancy(nusc, sample["token"], occupancy_root)
    return item
