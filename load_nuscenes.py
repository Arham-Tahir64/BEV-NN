"""Load nuScenes metadata and read a camera image and LiDAR point cloud.

Run with the project environment: python load_nuscenes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes

DEFAULT_DATAROOT = Path(__file__).resolve().parent / "data" / "nuscenes"
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
) -> dict:
    """Read one keyframe's RGB image, raw LiDAR points, and 3D annotations.

    image is a uint8 array with shape (height, width, 3).
    lidar_points is a float32 array with shape (N, 4): x, y, z, intensity.
    LiDAR coordinates and annotations remain in their original sensor/global
    frames respectively; no projection or coordinate transformation is applied.
    """
    if not 0 <= sample_index < len(nusc.sample):
        raise IndexError(f"sample_index must be between 0 and {len(nusc.sample) - 1}.")
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}; choose one of {CAMERAS}.")

    import numpy as np
    from PIL import Image
    from nuscenes.utils.data_classes import LidarPointCloud

    sample = nusc.sample[sample_index]
    image_path = Path(nusc.get_sample_data_path(sample["data"][camera]))
    lidar_path = Path(nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"]))
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    lidar = LidarPointCloud.from_file(str(lidar_path)).points.T.copy()
    annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]

    return {
        "sample": sample,
        "scene": nusc.get("scene", sample["scene_token"]),
        "image": rgb,
        "lidar_points": lidar,
        "annotations": annotations,
        "image_path": image_path,
        "lidar_path": lidar_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera", choices=CAMERAS, default="CAM_FRONT")
    args = parser.parse_args()

    try:
        nusc = load_nuscenes(args.dataroot, args.version)
        item = load_sample(nusc, args.sample_index, args.camera)
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Run: python -m pip install -r requirements.txt\n")
    except (FileNotFoundError, IndexError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print(f"\nLoaded {nusc.version}: {len(nusc.scene)} scenes, {len(nusc.sample)} samples")
    print(f"Scene: {item['scene']['name']}")
    print(f"Sample token: {item['sample']['token']}")
    print(f"Camera image: {item['image'].shape} ({item['image'].dtype})")
    print(f"LiDAR points: {item['lidar_points'].shape} ({item['lidar_points'].dtype})")
    print(f"Annotations: {len(item['annotations'])}")
    print(f"Image file: {item['image_path']}")
    print(f"LiDAR file: {item['lidar_path']}")


if __name__ == "__main__":
    main()

