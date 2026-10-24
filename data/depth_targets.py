"""Conservative, approximate first-surface depth labels from voxel annotations.

Trace feature-centre rays through exact voxel boundaries (3D DDA). Stop at
unknown/unobserved space; never look through it to manufacture a depth label.
The occupied voxel entry is an approximate surface, not sensor-measured depth.
Camera origins outside the annotated grid are deliberately left unsupervised.
"""
from pathlib import Path
import hashlib
import json
import os
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset


def first_surface_depth(semantics, valid, intrinsics, transforms, image_size, bounds,
                        min_depth=0.1, max_depth=64.1):
    """Return [N,ceil(H/16),ceil(W/16)] camera-Z depths; NaN means unknown.

    Rays start at the optical centre, so an occupied/unknown near region blocks
    supervision farther away even when it lies below min_depth.
    """
    shape = np.array(semantics.shape)
    low, high = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    size = (high-low) / shape
    h, w = ((v+15)//16 for v in image_size)
    yy, xx = np.meshgrid(np.arange(h)*16, np.arange(w)*16, indexing="ij")
    pixels = np.stack((xx.ravel(), yy.ravel(), np.ones(h*w)), axis=1)
    result = np.full((len(intrinsics), h*w), np.nan, dtype=np.float32)
    for cam, (k, transform) in enumerate(zip(intrinsics, transforms)):
        camera_rays = pixels @ np.linalg.inv(k).T
        camera_rays /= camera_rays[:, 2:3]  # Parameter t equals camera Z.
        inverse = np.linalg.inv(transform)
        origin = inverse[:3, 3]
        direction = camera_rays @ inverse[:3, :3].T
        if np.any(origin < low) or np.any(origin >= high):
            continue
        index = np.broadcast_to(np.floor((origin-low)/size).astype(int), (h*w, 3)).copy()
        step = np.sign(direction).astype(int)
        boundary = low + (index + (step > 0)) * size
        with np.errstate(divide="ignore", invalid="ignore"):
            t_next = np.where(direction != 0, (boundary-origin)/direction, np.inf)
            t_delta = np.where(direction != 0, size/np.abs(direction), np.inf)
        entry = np.zeros(h*w)
        active = np.ones(h*w, dtype=bool)
        # Each iteration crosses at least one grid plane.
        for _ in range(int(shape.sum())+3):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            inside = ((index[ids] >= 0) & (index[ids] < shape)).all(1) & (entry[ids] < max_depth)
            active[ids[~inside]] = False
            ids = ids[inside]
            coords = tuple(index[ids].T)
            labels = semantics[coords]
            known = valid[coords] & (labels != 255)
            occupied = known & (labels >= 0) & (labels < 17)
            hits = ids[occupied & (entry[ids] >= min_depth)]
            result[cam, hits] = entry[hits]
            active[ids[~known | occupied]] = False
            ids = ids[known & ~occupied]
            crossing = t_next[ids].min(1)
            axes = np.isclose(t_next[ids], crossing[:, None], rtol=0, atol=1e-9)
            index[ids] += axes * step[ids]
            t_next[ids] += np.where(axes, t_delta[ids], 0)
            entry[ids] = crossing
    return result.reshape(len(intrinsics), h, w)


class DepthTargetDataset(Dataset):
    """Wrap the occupancy dataset with content-keyed, atomic depth caches.

    Targets are in metres, independent of bin count; - no target is encoded as
    NaN. Source labels, masks, calibration and settings all enter the cache key.
    """
    def __init__(self, dataset, cache_dir, min_depth=0.1, max_depth=64.1):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_depth, self.max_depth = min_depth, max_depth

    def __getattr__(self, name):
        # Avoid recursive attribute lookup when DataLoader unpickles workers.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        arrays = [item[key].numpy() for key in
                  ("semantics", "valid_mask", "intrinsics", "ego_to_camera")]
        settings = ("dda-v1", self.image_size, self.voxel_bounds, self.min_depth, self.max_depth)
        digest = hashlib.sha256(json.dumps(settings).encode())
        for array in arrays:
            digest.update(array.tobytes())
        path = self.cache_dir / (digest.hexdigest()+".npy")
        if path.exists():
            depth = np.load(path, allow_pickle=False)
        else:
            depth = first_surface_depth(*arrays, self.image_size, self.voxel_bounds,
                                        self.min_depth, self.max_depth)
            temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, depth, allow_pickle=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return {**item, "depth_target": torch.from_numpy(depth.copy())}


def depth_loss(logits, target, edges):
    """Masked bin CE and valid count; no-target batches return differentiable zero."""
    if logits.ndim != 5 or target.shape != (logits.shape[0], logits.shape[1], *logits.shape[-2:]):
        raise ValueError("Depth logits/targets must be [B,N,D,h,w] / [B,N,h,w].")
    valid = torch.isfinite(target) & (target >= edges[0]) & (target < edges[-1])
    count = int(valid.sum().item())
    if not count:
        return logits.float().sum()*0, 0
    # Internal boundaries belong to the following bin; max edge is excluded.
    labels = torch.bucketize(target[valid].contiguous(), edges[1:-1], right=True)
    selected = logits.permute(0, 1, 3, 4, 2)[valid].float()
    return torch.nn.functional.cross_entropy(selected, labels), count
