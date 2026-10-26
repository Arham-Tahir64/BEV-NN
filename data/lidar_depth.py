"""Measured nuScenes LiDAR depth targets in the occupancy reference frame."""
from pathlib import Path
import numpy as np


def lidar_record(nusc, sample):
    """Prepare sensor -> sample-ego transform; point files are not read here."""
    if 'LIDAR_TOP' not in sample['data']:
        return {}
    from .voxel_projection import pose_matrix
    reference = None
    for token in sample['data'].values():
        record = nusc.get('sample_data', token)
        if record['timestamp'] == sample['timestamp']:
            reference = pose_matrix(nusc.get('ego_pose', record['ego_pose_token']))
            break
    if reference is None:
        raise ValueError('Measured depth needs an exact sample-time ego pose.')
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sensor_to_global = (pose_matrix(nusc.get('ego_pose', lidar['ego_pose_token'])) @
                        pose_matrix(nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])))
    return {'lidar_path': Path(nusc.get_sample_data_path(lidar['token'])),
            'lidar_to_ego': np.linalg.inv(reference) @ sensor_to_global}


def project_lidar_depth(points, lidar_to_ego, intrinsics, ego_to_camera, image_size,
                        min_depth=.1, max_depth=64.1):
    """Nearest measured camera-Z per feature cell; NaN denotes no supervision.

    Feature cells are centred at 16*i, matching this project's encoder and
    frustum. This differs from the official implementation's floor-block min
    pooling so targets and feature-ray coordinates remain aligned here.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('LiDAR points must be [P,3+] sensor-frame coordinates.')
    points = points[:, :3]
    points = points[np.isfinite(points).all(1)]
    ego = points @ lidar_to_ego[:3, :3].T + lidar_to_ego[:3, 3]
    h, w = ((s+15)//16 for s in image_size)
    result = np.full((len(intrinsics), h*w), np.inf, dtype=np.float32)
    for camera, (k, transform) in enumerate(zip(intrinsics, ego_to_camera)):
        xyz = ego @ transform[:3, :3].T + transform[:3, 3]
        z = xyz[:, 2]
        projection = xyz @ k.T
        uv = projection[:, :2] / np.maximum(z[:, None], 1e-6)
        cell = np.floor(uv/16 + .5).astype(np.int64)
        keep = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < image_size[1]) &
                (uv[:, 1] >= 0) & (uv[:, 1] < image_size[0]) &
                (cell[:, 0] >= 0) & (cell[:, 0] < w) & (cell[:, 1] >= 0) & (cell[:, 1] < h))
        np.minimum.at(result[camera], cell[keep, 1]*w+cell[keep, 0], z[keep])
    result[(result < min_depth) | (result >= max_depth) | ~np.isfinite(result)] = np.nan
    return result.reshape(len(intrinsics), h, w)
