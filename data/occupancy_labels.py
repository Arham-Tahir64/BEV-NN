"""Load dense Occ3D-nuScenes voxel semantics and camera visibility masks.

Format reference: https://github.com/Tsinghua-MARS-Lab/Occ3D
Arrays retain their saved axis order and resolution. Labels are not remapped;
unobserved voxels are not converted to free space.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes

DEFAULT_OCCUPANCY_ROOT = Path(__file__).resolve().parent / "occupancy" / "gts"
REQUIRED_KEYS = ("semantics", "mask_camera")


def load_occupancy_labels(label_path: str | Path) -> dict[str, np.ndarray]:
    """Read labels.npz and return semantics plus a boolean camera visibility mask.

    All arrays must share a nonempty 3D shape (X, Y, Z). Semantics must contain
    integer class IDs. The camera mask must contain only 0/1 or booleans;
    True means observed. Semantic IDs, including ignore IDs, remain unchanged.
    Other archive arrays are ignored and are not decoded.
    No resampling is performed; grid bounds and class mapping belong to the
    annotation dataset and must match the model configuration.
    """
    path = Path(label_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing occupancy labels: {path}. Dense voxel labels are separate "
            "annotations and are not included in the standard nuScenes mini download. "
            "Provide Occ3D-nuScenes labels.npz files."
        )
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an Occ3D .npz archive, got: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {', '.join(sorted(missing))}.")
        arrays = {key: archive[key].copy() for key in REQUIRED_KEYS}

    semantics = arrays["semantics"]
    if semantics.ndim != 3 or any(size == 0 for size in semantics.shape):
        raise ValueError(f"{path}: semantics must be a nonempty 3D grid, got {semantics.shape}.")
    if semantics.dtype.kind not in "iu":
        raise ValueError(f"{path}: semantics must contain integer class IDs, got {semantics.dtype}.")
    mask = arrays["mask_camera"]
    if mask.shape != semantics.shape:
        raise ValueError(f"{path}: mask_camera shape {mask.shape} does not match {semantics.shape}.")
    if mask.dtype.kind not in "biu" or not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{path}: mask_camera must contain only boolean or integer 0/1 values.")
    arrays["mask_camera"] = mask.astype(np.bool_, copy=False)
    return arrays


def load_sample_occupancy(
    nusc: NuScenes,
    sample_token: str,
    occupancy_root: str | Path = DEFAULT_OCCUPANCY_ROOT,
) -> dict[str, np.ndarray]:
    """Resolve gts/<scene_name>/<sample_token>/labels.npz for a keyframe.

    occupancy_root must point directly to gts, not its parent directory.
    Occupancy annotations are keyframe labels; camera sweeps are not assigned
    fabricated labels or automatically reused keyframe ground truth.
    """
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    path = Path(occupancy_root).expanduser().resolve() / scene["name"] / sample_token / "labels.npz"
    return load_occupancy_labels(path)
