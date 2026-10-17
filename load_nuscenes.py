"""Load nuScenes data and stream all six cameras through the entire dataset.

Run with the project environment: python load_nuscenes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from data.nuscenes_dataset import CAMERAS, DEFAULT_DATAROOT, load_nuscenes, load_sample


def show_camera_frames(item: dict, save_path: str | Path | None = None, show: bool = True):
    """Display all six images, with ground-truth semantic dots when attached."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from data.voxel_projection import LEGEND, PALETTE, draw_voxel_overlay

    order = (
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    )
    overlays = item.get("voxel_projections")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor="#171d23")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88,
                        bottom=0.13 if overlays is not None else 0.04, wspace=0.02, hspace=0.12)
    for ax, channel in zip(axes.flat, order):
        image = Image.fromarray(item["images"][channel])
        if overlays is not None:
            image = draw_voxel_overlay(image, overlays[channel], radius=1, spacing=28)
        ax.imshow(image)
        ax.set_title(channel, color="#dce1e8", fontsize=11, loc="left")
        ax.axis("off")
    heading = "Ground-truth voxel projection" if overlays is not None else "Six-camera views"
    fig.suptitle(f"{heading}\n{item['scene']['name']} | Sample {item['sample']['token']}",
                 color="white", fontsize=16)
    if overlays is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], linestyle="", marker="o", markersize=7,
                          color=PALETTE[class_id] / 255, label=name) for name, class_id in LEGEND]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                   labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 0.015))
    if save_path is not None:
        destination = Path(save_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, facecolor=fig.get_facecolor())
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
                        help="Occ3D gts folder (default: data/occupancy/gts).")
    parser.add_argument("--project-voxels", action=argparse.BooleanOptionalAction, default=None,
                        help="Overlay GT voxel centres (automatic when labels are installed).")
    args = parser.parse_args()
    from data.occupancy_labels import DEFAULT_OCCUPANCY_ROOT
    occupancy_root = args.occupancy_root or DEFAULT_OCCUPANCY_ROOT
    project_voxels = args.project_voxels
    if project_voxels is None:
        project_voxels = args.occupancy_root is not None or next(occupancy_root.glob("scene-*/*/labels.npz"), None) is not None

    try:
        nusc = load_nuscenes(args.dataroot, args.version)
        if not args.single_frame and not args.no_show and args.save_camera_grid is None:
            from camera_stream import stream_camera_frames
            stream_camera_frames(nusc, scene_name=args.scene, speed=args.speed, loop=args.loop,
                                 occupancy_root=occupancy_root if project_voxels else None)
            return
        item = load_sample(nusc, args.sample_index, args.camera,
                           occupancy_root=occupancy_root if project_voxels else args.occupancy_root)
        if project_voxels:
            from data.voxel_projection import project_sample_voxels
            item["voxel_projections"] = project_sample_voxels(nusc, item["sample"]["token"], item["occupancy"])
            for channel, projection in item["voxel_projections"].items():
                print(f"{channel}: {len(projection['uv'])} voxel centres in view")
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

