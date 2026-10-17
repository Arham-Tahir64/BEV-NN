"""Project Occ3D ego-frame voxel centres onto calibrated nuScenes cameras.

Occ3D grid/class conventions: https://github.com/Tsinghua-MARS-Lab/Occ3D
Transforms use sample-time ego -> global -> camera-time ego -> camera.
Only pose/calibration metadata and occupancy labels are used, never raw clouds.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .nuscenes_dataset import CAMERAS
from .occupancy_labels import DEFAULT_OCCUPANCY_ROOT, load_sample_occupancy

GRID_SHAPE = (200, 200, 16)
GRID_MIN = np.array([-40.0, -40.0, -1.0])
VOXEL_SIZE = np.array([0.4, 0.4, 0.4])
CLASS_NAMES = (
    "Others", "Barrier", "Bicycle", "Bus", "Car", "Construction vehicle",
    "Motorcycle", "Pedestrian", "Traffic cone", "Trailer", "Truck",
    "Driveable surface", "Other flat", "Sidewalk", "Terrain", "Manmade",
    "Vegetation", "Free",
)
# Display colours, grouped for a readable legend; original semantic IDs survive.
PALETTE = np.array([
    [180, 180, 180], [255, 220, 75], [255, 158, 30], [255, 158, 30],
    [255, 158, 30], [255, 158, 30], [255, 158, 30], [255, 75, 100],
    [255, 220, 75], [255, 158, 30], [255, 158, 30], [70, 215, 240],
    [170, 150, 105], [245, 151, 200], [150, 195, 110], [185, 115, 240],
    [120, 225, 75], [0, 0, 0],
], dtype=np.uint8)
LEGEND = (
    ("Road", 11), ("Vehicles / cycles", 4), ("Manmade", 15),
    ("Vegetation", 16), ("Sidewalk", 13), ("Pedestrian", 7),
    ("Barrier / cone", 1), ("Terrain", 14), ("Other flat", 12), ("Others", 0),
)


def occupied_voxel_centres(occupancy: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return visible occupied *surface* centres and unchanged semantic IDs.

    Occ3D-nuScenes arrays are [X,Y,Z], 0.4 m, ego x-forward/y-left/z-up.
    Free=17, ignore=255. Unsupported shapes/IDs fail instead of silently
    projecting a differently configured grid. Interior voxels are discarded.
    """
    labels = occupancy["semantics"]
    visible = occupancy["mask_camera"]
    if labels.shape != GRID_SHAPE or visible.shape != GRID_SHAPE:
        raise ValueError(f"Projection requires the Occ3D-nuScenes {GRID_SHAPE} grid; got {labels.shape}.")
    if not np.all((labels <= 17) | (labels == 255)) or np.any(labels < 0):
        raise ValueError("Projection requires Occ3D semantic IDs 0..17 (or ignore=255).")
    occupied = labels < 17
    padded = np.pad(occupied, 1, constant_values=False)
    interior = occupied.copy()
    for axis in range(3):
        for offset in (-1, 1):
            slices = [slice(1, -1)] * 3
            slices[axis] = slice(0, -2) if offset == -1 else slice(2, None)
            interior &= padded[tuple(slices)]
    keep = occupied & ~interior & visible.astype(bool)
    indices = np.argwhere(keep)
    centres = GRID_MIN + (indices + 0.5) * VOXEL_SIZE
    return centres, labels[keep]


def pose_matrix(record: dict) -> np.ndarray:
    """nuScenes quaternion order is [w,x,y,z]; returns local -> parent."""
    from pyquaternion import Quaternion

    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    matrix[:3, 3] = record["translation"]
    return matrix


def camera_geometry(nusc, sample: dict) -> dict:
    """Get reference ego -> each camera and native image intrinsics.

    The occupancy ego pose is at the sample timestamp, whereas individual
    cameras have their own acquisition poses. Read the exact reference pose
    from a matching sample_data metadata record; no sensor file is opened.
    """
    reference_pose = None
    for token in sample["data"].values():
        record = nusc.get("sample_data", token)
        if record["timestamp"] == sample["timestamp"]:
            reference_pose = nusc.get("ego_pose", record["ego_pose_token"])
            break
    if reference_pose is None:
        raise ValueError("No exact sample-time ego pose found; cannot align occupancy safely.")
    ego_to_global = pose_matrix(reference_pose)
    geometry = {}
    for channel in CAMERAS:
        record = nusc.get("sample_data", sample["data"][channel])
        calibration = nusc.get("calibrated_sensor", record["calibrated_sensor_token"])
        camera_pose = nusc.get("ego_pose", record["ego_pose_token"])
        ego_to_camera = (
            np.linalg.inv(pose_matrix(calibration))
            @ np.linalg.inv(pose_matrix(camera_pose))
            @ ego_to_global
        )
        geometry[channel] = {
            "ego_to_camera": ego_to_camera,
            "intrinsics": np.array(calibration["camera_intrinsic"], dtype=float),
            "image_size": (record["width"], record["height"]),
        }
    return geometry


def project_centres(points, labels, ego_to_camera, intrinsics, image_size, min_depth=0.1):
    """Project pinhole centres; reject points behind camera or outside image."""
    camera = points @ ego_to_camera[:3, :3].T + ego_to_camera[:3, 3]
    depth = camera[:, 2]
    keep = np.isfinite(camera).all(axis=1) & (depth > min_depth)
    camera, depth, labels = camera[keep], depth[keep], labels[keep]
    projected = camera @ intrinsics.T
    uv = projected[:, :2] / projected[:, 2:3]
    width, height = image_size
    keep = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return {"uv": uv[keep], "depth": depth[keep], "labels": labels[keep], "image_size": image_size}


def project_sample_voxels(nusc, sample_token: str, occupancy: dict) -> dict:
    points, labels = occupied_voxel_centres(occupancy)
    geometry = camera_geometry(nusc, nusc.get("sample", sample_token))
    return {
        channel: project_centres(points, labels, **parameters)
        for channel, parameters in geometry.items()
    }


def visible_projection(projection: dict, image_size, spacing: int = 14) -> dict:
    """Scale to display pixels and keep the nearest centre per image cell.

    This is an approximate centre-based z-buffer for overdraw/occlusion, not
    mesh rendering. It cannot guarantee perfect visibility at object edges.
    Pixel-centre scaling matches PIL resizing. Filtering precedes drawing.
    """
    if spacing < 1:
        raise ValueError("Point spacing must be at least one pixel.")
    scale = np.array(image_size) / np.array(projection["image_size"])
    uv = (projection["uv"] + 0.5) * scale - 0.5
    # Clamp subpixel border coordinates to their edge pixel after resize.
    uv = np.clip(uv, 0, np.array(image_size) - 1)
    cells = np.floor(uv / spacing).astype(np.int64)
    columns = (image_size[0] + spacing - 1) // spacing
    ids = cells[:, 1] * columns + cells[:, 0]
    by_depth = np.argsort(projection["depth"], kind="stable")
    _, first = np.unique(ids[by_depth], return_index=True)
    selected = by_depth[first]
    return {"uv": uv[selected], "depth": projection["depth"][selected],
            "labels": projection["labels"][selected]}


def draw_voxel_overlay(image, projection: dict, radius: int = 1, spacing: int = 14):
    """Draw coloured semantic dots on an RGB PIL image; leave input unchanged."""
    from PIL import ImageDraw

    if radius < 1:
        raise ValueError("Point radius must be at least one pixel.")
    image = image.convert("RGB").copy()
    selected = visible_projection(projection, image.size, spacing)
    draw = ImageDraw.Draw(image)
    for (x, y), label in zip(selected["uv"], selected["labels"]):
        colour = tuple(int(value) for value in PALETTE[int(label)])
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=colour)
    return image


def occupancy_label_path(root, scene_name: str, sample_token: str) -> Path:
    return Path(root).expanduser().resolve() / scene_name / sample_token / "labels.npz"


def build_occupancy_timeline(nusc, scene_name=None, occupancy_root=DEFAULT_OCCUPANCY_ROOT):
    """Use exact keyframe camera tokens for installed GT; never reuse sweep GT."""
    scenes = [scene for scene in nusc.scene if scene_name is None or scene["name"] == scene_name]
    if not scenes:
        raise ValueError(f"Scene {scene_name!r} was not found.")
    scenes.sort(key=lambda scene: nusc.get("sample", scene["first_sample_token"])["timestamp"])
    timeline, missing = [], 0
    for scene in scenes:
        samples = []
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            samples.append(sample)
            token = sample["next"]
        for i, sample in enumerate(samples):
            if not occupancy_label_path(occupancy_root, scene["name"], sample["token"]).is_file():
                missing += 1
                continue
            timeline.append({
                "scene": scene["name"], "timestamp": sample["timestamp"],
                "sample_token": sample["token"], "scene_start": samples[0]["timestamp"],
                "scene_frame": i + 1, "scene_frames": len(samples),
                "camera_tokens": {channel: sample["data"][channel] for channel in CAMERAS},
            })
    if not timeline:
        raise FileNotFoundError(
            f"No matching occupancy keyframes under {Path(occupancy_root).resolve()}. "
            "Expected <scene_name>/<sample_token>/labels.npz."
        )
    if missing:
        print(f"Skipped {missing} keyframes without occupancy ground truth.")
    return timeline


def overlay_camera_images(nusc, frame: dict, images: dict, occupancy_root):
    occupancy = load_sample_occupancy(nusc, frame["sample_token"], occupancy_root)
    projections = project_sample_voxels(nusc, frame["sample_token"], occupancy)
    return {channel: draw_voxel_overlay(image, projections[channel]) for channel, image in images.items()}
