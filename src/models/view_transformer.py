"""Scaffold for geometric lifting of camera features into a 3D voxel grid."""
from __future__ import annotations

import torch
from torch import nn

import math


class ViewTransformer(nn.Module):
    """Project voxel centres into cameras, sample features, and fuse valid views.

    Inputs:
        features: [B, 6, C, h, w] from ImageEncoder.
        intrinsics: [B, 6, 3, 3], calibrated for the resized input images.
        ego_to_camera: [B, 6, 4, 4], mapping sample-time ego coordinates
            (the occupancy frame) into each camera's acquisition frame.
        image_size: (H, W) of those resized images, e.g. (320, 576).

    Output:
        Voxel features [B, C, X, Y, Z], in the labels' [X, Y, Z] axis order.

    Default Occ3D grid:
        Shape: (200, 200, 16).
        Bounds: (-40, -40, -1) to (40, 40, 5.4) metres.
        Voxel size: (0.4, 0.4, 0.4) metres.

    Baseline fusion uses masked averaging. Camera validity means positive
    depth and a projection inside the image; it does not prove lack of occlusion.
    Ground-truth visibility masks belong to supervision and are not inputs here.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (200, 200, 16),
        voxel_bounds: tuple[float, float, float, float, float, float] = (
            -40.0, -40.0, -1.0, 40.0, 40.0, 5.4
        ),
        min_depth: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.voxel_bounds = voxel_bounds
        self.min_depth = min_depth

        # TODO: Validate grid dimensions, ordered bounds, and min_depth.
        if len(grid_shape) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in grid_shape
        ):
            raise ValueError("grid_shape must contain three positive integers.")

        if len(voxel_bounds) != 6:
            raise ValueError("voxel_bounds must contain (xmin, ymin, zmin, xmax, ymax, zmax).")

        if not all(math.isfinite(value) for value in voxel_bounds):
            raise ValueError("voxel_bounds must contain finite values.")

        minimum = voxel_bounds[:3]
        maximum = voxel_bounds[3:]

        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError("Each minimum bound must be smaller than its maximum.")

        if not math.isfinite(min_depth) or min_depth <= 0:
            raise ValueError("min_depth must be finite and positive.")

        self.voxel_centres = self.make_voxel_centres()
        self.register_buffer("voxel_centres", self.voxel_centres)

def make_voxel_centres(self) -> torch.Tensor:
    """Return metric voxel centres [V, 3], where V = X * Y * Z."""
    minimum = self.voxel_bounds[:3]
    maximum = self.voxel_bounds[3:]
    grid_shape = self.grid_shape

    x_size = (maximum[0] - minimum[0]) / grid_shape[0]
    y_size = (maximum[1] - minimum[1]) / grid_shape[1]
    z_size = (maximum[2] - minimum[2]) / grid_shape[2]

    x_centres = minimum[0] + (
        torch.arange(grid_shape[0], dtype=torch.float32) + 0.5
    ) * x_size
    y_centres = minimum[1] + (
        torch.arange(grid_shape[1], dtype=torch.float32) + 0.5
    ) * y_size
    z_centres = minimum[2] + (
        torch.arange(grid_shape[2], dtype=torch.float32) + 0.5
    ) * z_size

    x_grid, y_grid, z_grid = torch.meshgrid(
        x_centres, y_centres, z_centres,
        indexing="ij",
    )

    return torch.stack(
        [x_grid, y_grid, z_grid], dim=-1
    ).reshape(-1, 3)

    def project_voxels(
        self,
        centres: torch.Tensor,
        intrinsics: torch.Tensor,
        ego_to_camera: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image pixels [B,6,V,2] and valid-camera mask [B,6,V]."""
        # TODO: Transform ego-frame centres into each camera frame.
        # TODO: Apply intrinsics and divide by safe positive depth.
        # TODO: Mask points behind/too close to cameras or outside the image.
        # TODO: Keep projected pixel coordinates in the resized image frame.
        raise NotImplementedError("Implement voxel-to-camera projection.")

    def sample_features(
        self,
        features: torch.Tensor,
        pixels: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return sampled feature vectors [B,6,C,V]."""
        # TODO: Convert image pixels into feature-map sampling coordinates.
        # Account for the backbone stride and pixel-centre convention.
        # TODO: Convert to grid_sample coordinates with an explicit,
        # consistent align_corners convention.
        # TODO: Use bilinear grid_sample; preserve gradients to image features.
        raise NotImplementedError("Implement image-feature sampling.")

    def fuse_features(
        self,
        sampled: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked mean features [B,C,V]; unseen voxels receive zeros."""
        # TODO: Mask invalid camera contributions before summing.
        # TODO: Divide by valid-camera counts, avoiding division by zero.
        raise NotImplementedError("Implement masked camera averaging.")

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        ego_to_camera: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Lift per-camera features into [B,C,X,Y,Z] voxel features."""
        # TODO: Validate shapes, camera order, devices, and floating-point dtypes.
        # TODO: Project voxel centres into all cameras.
        # TODO: Sample features and fuse valid contributions.
        # Consider processing voxel chunks to limit peak memory.
        # TODO: Restore [X, Y, Z] dimensions without changing voxel order.
        raise NotImplementedError("Implement the view-transformer forward pass.")
