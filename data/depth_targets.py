"""Approximate depth supervision from occupancy annotations, not measured depth.

Legacy ray mode stops at unknown space using exact voxel traversal. Projected
mode selects the nearest observed occupied voxel centre in each feature cell;
it does not require an observed camera origin. Unknown gaps can conceal nearer
surfaces, so projected targets are first annotated surfaces, not guaranteed
physical first hits. Neither mode uses labels as model inputs at inference.
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


def projected_surface_depth(semantics, observed, camera_visible, intrinsics,
                            transforms, image_size, bounds, min_depth=0.1,
                            max_depth=64.1):
    """Approximate foreground camera-Z depth per stride-16 feature cell.

    All observed occupied centres occlude, including non-camera-visible ones.
    Accept a target only when its nearest centre is camera-visible. Feature
    cells are centred at pixel i*16. Unknown voxels never become free targets.
    Missing annotations can still hide an occluder: these are weak pseudo-labels,
    not measured depth. Foreground-min pooling biases mixed cells toward near
    surfaces. Camera calibration and feature stride match ViewTransformer.
    """
    if semantics.shape != observed.shape or semantics.shape != camera_visible.shape:
        raise ValueError('Depth annotation shapes must match.')
    h, w = ((v+15)//16 for v in image_size)
    result = np.full((len(intrinsics), h*w), np.nan, dtype=np.float32)
    indices = np.argwhere(observed & (semantics >= 0) & (semantics < 17))
    if not len(indices):
        return result.reshape(len(intrinsics), h, w)
    low, high = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    points = low + (indices + .5) * ((high-low)/np.asarray(semantics.shape))
    visible = camera_visible[tuple(indices.T)]
    for c, (k, transform) in enumerate(zip(intrinsics, transforms)):
        xyz = points @ transform[:3, :3].T + transform[:3, 3]
        z = xyz[:, 2]
        uvw = xyz @ k.T
        uv = uvw[:, :2] / np.maximum(z[:, None], 1e-6)
        cells = np.floor(uv/16 + .5).astype(np.int64)
        # Include near surfaces as occluders even below the target range.
        inside = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < image_size[1])
                  & (uv[:, 1] >= 0) & (uv[:, 1] < image_size[0])
                  & (cells[:, 0] >= 0) & (cells[:, 0] < w)
                  & (cells[:, 1] >= 0) & (cells[:, 1] < h))
        ids = np.flatnonzero(inside)
        ids = ids[np.argsort(z[ids], kind='stable')]
        flat = cells[ids, 1]*w + cells[ids, 0]
        _, first = np.unique(flat, return_index=True)
        nearest = ids[first]
        valid = visible[nearest] & (z[nearest] >= min_depth) & (z[nearest] < max_depth)
        selected = nearest[valid]
        result[c, cells[selected, 1]*w + cells[selected, 0]] = z[selected]
    return result.reshape(len(intrinsics), h, w)


class DepthTargetDataset(Dataset):
    """Wrap the occupancy dataset with content-keyed, atomic depth caches.

    Targets are in metres, independent of bin count; missing targets are NaN.
    Source labels, masks, calibration and settings all enter the cache key.
    """
    def __init__(self, dataset, cache_dir, min_depth=0.1, max_depth=64.1,
                 target_mode='ray'):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_depth, self.max_depth = min_depth, max_depth
        if target_mode not in ('ray', 'projected'):
            raise ValueError('Unknown depth target mode.')
        self.target_mode = target_mode

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
        observed = None
        if self.target_mode == 'projected':
            # Annotation mask only: no sensor point clouds are loaded.
            with np.load(self.dataset.records[index]['label_path'], allow_pickle=False) as archive:
                if 'mask_lidar' not in archive:
                    raise ValueError('Projected targets require annotation mask_lidar.')
                observed = archive['mask_lidar']
            if observed.shape != arrays[0].shape or not np.isin(observed, [0, 1]).all():
                raise ValueError('Invalid observed-space annotation mask.')
            observed = observed.astype(bool)
        settings = ("dda-v1" if self.target_mode == 'ray' else 'projected-v2',
                    self.image_size, self.voxel_bounds, self.min_depth, self.max_depth)
        digest = hashlib.sha256(json.dumps(settings).encode())
        for array in arrays:
            digest.update(array.tobytes())
        if observed is not None:
            digest.update(observed.tobytes())
        path = self.cache_dir / (digest.hexdigest()+".npy")
        if path.exists():
            depth = np.load(path, allow_pickle=False)
        else:
            if self.target_mode == 'ray':
                depth = first_surface_depth(*arrays, self.image_size, self.voxel_bounds,
                                            self.min_depth, self.max_depth)
            else:
                depth = projected_surface_depth(arrays[0], observed, arrays[1], arrays[2], arrays[3],
                    self.image_size, self.voxel_bounds, self.min_depth, self.max_depth)
            temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, depth, allow_pickle=False)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return {**item, "depth_target": torch.from_numpy(depth.copy())}


def depth_loss(logits, target, edges, mode='hard'):
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
    if mode == 'hard':
        return torch.nn.functional.cross_entropy(selected, labels), count
    if mode != 'linear':
        raise ValueError('Unknown depth loss mode.')
    # Split supervision across neighbouring bin centres instead of making a
    # noisy voxel surface abruptly switch one-hot labels at a bin boundary.
    coordinate = ((target[valid]-edges[0])/(edges[1]-edges[0])-.5).clamp(0, logits.shape[2]-1)
    left = coordinate.floor().long()
    right = (left+1).clamp_max(logits.shape[2]-1)
    fraction = coordinate-left
    logp = selected.log_softmax(-1)
    loss = -(1-fraction)*logp.gather(1, left[:, None]).squeeze(1)
    loss -= fraction*logp.gather(1, right[:, None]).squeeze(1)
    return loss.mean(), count
