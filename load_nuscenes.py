"""Load nuScenes data and stream all six cameras through the entire dataset.

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
    """Read one keyframe's six RGB images, raw LiDAR points, and 3D annotations.

    images maps each camera name to a uint8 (height, width, 3) RGB array.
    image retains the selected camera for single-camera callers.
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
    lidar_path = Path(nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"]))
    lidar = LidarPointCloud.from_file(str(lidar_path)).points.T.copy()
    annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]

    return {
        "sample": sample,
        "scene": nusc.get("scene", sample["scene_token"]),
        "image": rgb,
        "images": images,
        "image_paths": image_paths,
        "lidar_points": lidar,
        "annotations": annotations,
        "image_path": image_path,
        "lidar_path": lidar_path,
    }


def show_camera_frames(item: dict, save_path: str | Path | None = None, show: bool = True):
    """Display the six camera images in a labeled 2-by-3 grid."""
    import matplotlib.pyplot as plt

    order = (
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), layout="constrained")
    for ax, channel in zip(axes.flat, order):
        ax.imshow(item["images"][channel])
        ax.set_title(channel)
        ax.axis("off")
    fig.suptitle(f"{item['scene']['name']} | Sample {item['sample']['token']}")
    if save_path is not None:
        destination = Path(save_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
        print(f"Camera grid saved to: {destination}")
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera", choices=CAMERAS, default="CAM_FRONT")
    parser.add_argument("--no-show", action="store_true", help="Skip the camera window.")
    parser.add_argument("--save-camera-grid", type=Path, help="Save all six views as an image.")
    parser.add_argument("--single-frame", action="store_true", help="Display one sample instead of video.")
    parser.add_argument("--scene", help="Stream only this scene name, e.g. scene-0061.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1).")
    parser.add_argument("--loop", action="store_true", help="Repeat video after the final scene.")
    args = parser.parse_args()

    try:
        nusc = load_nuscenes(args.dataroot, args.version)
        if not args.single_frame and not args.no_show and args.save_camera_grid is None:
            from camera_stream import stream_camera_frames
            stream_camera_frames(nusc, scene_name=args.scene, speed=args.speed, loop=args.loop)
            return
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
    print(f"Camera views loaded: {', '.join(item['images'])}")
    if not args.no_show or args.save_camera_grid:
        show_camera_frames(item, save_path=args.save_camera_grid, show=not args.no_show)


if __name__ == "__main__":
    main()

