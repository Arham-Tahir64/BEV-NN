"""Load nuScenes data and stream all six cameras through the entire dataset.

Run with the project environment: python load_nuscenes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from data.nuscenes_dataset import CAMERAS, DEFAULT_DATAROOT, load_nuscenes, load_sample


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
    parser.add_argument("--occupancy-root", type=Path,
                        help="Occ3D gts folder; attach labels in sample mode (--no-show or --single-frame).")
    args = parser.parse_args()
    if args.occupancy_root is not None and not (args.single_frame or args.no_show or args.save_camera_grid):
        parser.error("--occupancy-root requires sample mode: --single-frame or --no-show.")

    try:
        nusc = load_nuscenes(args.dataroot, args.version)
        if not args.single_frame and not args.no_show and args.save_camera_grid is None:
            from camera_stream import stream_camera_frames
            stream_camera_frames(nusc, scene_name=args.scene, speed=args.speed, loop=args.loop)
            return
        item = load_sample(nusc, args.sample_index, args.camera, occupancy_root=args.occupancy_root)
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Run: python -m pip install -r requirements.txt\n")
    except (FileNotFoundError, IndexError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print(f"\nLoaded {nusc.version}: {len(nusc.scene)} scenes, {len(nusc.sample)} samples")
    print(f"Scene: {item['scene']['name']}")
    print(f"Sample token: {item['sample']['token']}")
    print(f"Camera image: {item['image'].shape} ({item['image'].dtype})")
    print(f"Annotations: {len(item['annotations'])}")
    print(f"Image file: {item['image_path']}")
    print(f"Camera views loaded: {', '.join(item['images'])}")
    if "occupancy" in item:
        occupancy = item["occupancy"]
        print(f"Voxel labels: {occupancy['semantics'].shape} ({occupancy['semantics'].dtype})")
        print(f"Camera-visible voxels: {occupancy['mask_camera'].sum()}")
    if not args.no_show or args.save_camera_grid:
        show_camera_frames(item, save_path=args.save_camera_grid, show=not args.no_show)


if __name__ == "__main__":
    main()

