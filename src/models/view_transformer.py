"""Geometric lifting of camera features into a 3D voxel grid."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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
        Residual depth fusion returns 2*C+1 channels (raw, gated, support).
        Gated depth fusion returns C+1 channels (gated, support).
        Probability fusion returns C+1 channels sampled jointly from image
        features multiplied by D*p(depth), plus D*p(depth) support. It has no
        raw appearance branch or probability clipping. A fixed global D
        preserves uniform-depth feature scale without cancelling the gate.

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
        voxel_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.voxel_bounds = voxel_bounds
        self.min_depth = min_depth

        # Validate the fixed grid and projection settings.
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

        if (
            isinstance(voxel_chunk_size, bool)
            or not isinstance(voxel_chunk_size, int)
            or voxel_chunk_size <= 0
        ):
            raise ValueError("voxel_chunk_size must be a positive integer.")
        self.voxel_chunk_size = voxel_chunk_size

        self.register_buffer("voxel_centres", self.make_voxel_centres())

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
        # Transform ego-frame centres into each camera frame.
        centres = centres.to(
            device=ego_to_camera.device,
            dtype=ego_to_camera.dtype,
        )

        rotation = ego_to_camera[..., :3, :3]    # [B, 6, 3, 3]
        translation = ego_to_camera[..., :3, 3]  # [B, 6, 3]

        camera_points = torch.einsum(
            "bnij,vj->bnvi", rotation, centres
        ) + translation.unsqueeze(-2)           # [B, 6, V, 3]
        # Apply intrinsics and divide by safe positive depth.
        depth = camera_points[..., 2]  # [B, 6, V]
        valid_depth = depth > self.min_depth

        # Apply each camera's intrinsic matrix.
        projected = torch.einsum(
            "bnij,bnvj->bnvi", intrinsics, camera_points
        )  # [B, 6, V, 3]

        # Avoid division by zero or negative depth.
        safe_depth = depth.clamp_min(self.min_depth)

        pixels = projected[..., :2] / safe_depth.unsqueeze(-1)
        # [B, 6, V, 2], holding [u, v]
        # Mask points behind/too close to cameras or outside the image.
        H, W = image_size

        u = pixels[..., 0]
        v = pixels[..., 1]

        inside_image = (
            (u >= 0)
            & (u < W)
            & (v >= 0)
            & (v < H)
        )

        valid = valid_depth & inside_image

        return pixels, valid

    def sample_features(
        self,
        features: torch.Tensor,
        pixels: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return sampled feature vectors [B,6,C,V]."""
        # ResNet-18 through layer3 has stride 16 and feature index 0
        # is centred on input pixel 0. Preserve fractional coordinates.
        feature_stride = 16
        feature_pixels = pixels / feature_stride  # [B, 6, V, 2], [x, y]

        # With align_corners=False, -1 and +1 are the feature-map edges;
        # feature-pixel centres use 2 * (coordinate + 0.5) / size - 1.
        align_corners = False
        feature_height, feature_width = features.shape[-2:]
        feature_size = feature_pixels.new_tensor([feature_width, feature_height])
        sampling_grid = 2.0 * (feature_pixels + 0.5) / feature_size - 1.0
        # sampling_grid: [B, 6, V, 2], ordered [x, y]. Do not clamp;
        # invalid camera contributions are excluded by the projection mask.
        # Treat each camera as a batch item, with one output location per voxel.
        batch_size, cameras, channels, _, _ = features.shape
        voxel_count = pixels.shape[-2]
        flat_features = features.reshape(
            batch_size * cameras, channels, feature_height, feature_width
        )
        grid = sampling_grid.to(device=features.device, dtype=features.dtype)
        grid = grid.reshape(batch_size * cameras, voxel_count, 1, 2)

        # No detach/no_grad: gradients flow through sampling to image features.
        sampled = F.grid_sample(
            flat_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )  # [B * 6, C, V, 1]
        return sampled.reshape(batch_size, cameras, channels, voxel_count)

    def fuse_features(
        self,
        sampled: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked mean features [B,C,V]; unseen voxels receive zeros."""
        # Broadcast one validity value over every feature channel.
        camera_mask = valid.unsqueeze(2)  # [B, 6, 1, V]
        masked = sampled.masked_fill(~camera_mask, 0.0)
        feature_sum = masked.sum(dim=1)  # [B, C, V]

        # Average only valid views. With no valid views the sum is zero;
        # clamping the denominator to one keeps that output zero and finite.
        counts = camera_mask.sum(dim=1).clamp_min(1)  # [B, 1, V]
        return feature_sum / counts.to(dtype=sampled.dtype)

    def sample_depth_support(self, probabilities, pixels, depths, edges):
        """Trilinear sampling returns [B,N,V], never [B,N,D,V].

        Uniform bins; border padding extends end-bin centres to their edges.
        Explicit masks exclude out-of-range depth and image positions. FP32
        sampling supports CPU and CUDA AMP with gradients to the depth head.
        """
        b, n, d, h, w = probabilities.shape
        xy = 2 * (pixels.float() / 16 + 0.5) / pixels.new_tensor([w, h]).float() - 1
        z = 2 * (depths.float() - edges[0]) / (edges[-1] - edges[0]) - 1
        grid = torch.cat((xy, z.unsqueeze(-1)), dim=-1).reshape(b*n, 1, -1, 1, 3)
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            support = F.grid_sample(probabilities.float().reshape(b*n, 1, d, h, w),
                                    grid, align_corners=False, padding_mode="border")
        support = support.reshape(b, n, -1)
        return support.masked_fill((depths < edges[0]) | (depths >= edges[-1]), 0)

    def make_probability_volume(self, features, probabilities):
        """Weight context BEFORE spatial interpolation; return [B*N,C+1,D,h,w].

        Interpolating context and probability separately creates cross terms
        between adjacent image rays: a foreground feature can borrow a
        background pixel's depth. Joint interpolation retains their pairing.
        D is a fixed global scale, not a per-voxel support normalization.
        The small image frustum is materialized once, never [B,N,C,D,V].
        """
        b, n, c, h, w = features.shape
        d = probabilities.shape[2]
        # CUDA's trilinear grid sampler requires FP32 under mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            support = probabilities.float().unsqueeze(2)
            weighted = features.float().unsqueeze(3) * support
            joint = torch.cat((weighted, support), dim=2) * d
        return joint.reshape(b*n, c+1, d, h, w)

    def sample_probability_volume(self, volume, pixels, depths, edges):
        """Sample paired appearance/depth; preserve end bins and image bounds."""
        b, n, v = depths.shape
        _, channels, d, h, w = volume.shape
        xy = 2 * (pixels.float() / 16 + .5) / pixels.new_tensor([w, h]).float() - 1
        z = 2 * (depths.float() - edges[0]) / (edges[-1] - edges[0]) - 1
        # Extend the first/last depth centres only to their own bin edges.
        # Spatial padding stays zero; border padding would copy edge pixels.
        z = z.clamp(min=-1 + 1/d, max=1 - 1/d)
        grid = torch.cat((xy, z.unsqueeze(-1)), dim=-1).reshape(b*n, 1, v, 1, 3)
        with torch.autocast(device_type=volume.device.type, enabled=False):
            sampled = F.grid_sample(volume, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(b, n, channels, v)
        outside_depth = (depths < edges[0]) | (depths >= edges[-1])
        return sampled.masked_fill(outside_depth.unsqueeze(2), 0)

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        ego_to_camera: torch.Tensor,
        image_size: tuple[int, int],
        *, depth_probs: torch.Tensor | None = None,
        depth_edges: torch.Tensor | None = None,
        depth_fusion: str = 'legacy',
    ) -> torch.Tensor:
        """Lift per-camera features into [B,C,X,Y,Z] voxel features.

        Process voxel chunks to limit temporary sampling tensors. Camera order
        must match the dataset's calibration order. Geometry uses at least FP32,
        including when the caller enables mixed-precision autocasting.
        """
        if features.ndim != 5:
            raise ValueError("features must have shape [B,6,C,h,w].")
        batch_size, cameras, channels, feature_height, feature_width = features.shape
        if cameras != 6 or min(batch_size, channels, feature_height, feature_width) <= 0:
            raise ValueError("Expected six cameras and positive batch/channel/spatial dimensions.")
        if (
            len(image_size) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in image_size)
        ):
            raise ValueError("image_size must be two positive integers (height, width).")
        expected_size = tuple((size + 15) // 16 for size in image_size)
        if (feature_height, feature_width) != expected_size:
            raise ValueError(f"Expected stride-16 feature size {expected_size}, got {features.shape[-2:]}.")
        if intrinsics.shape != (batch_size, cameras, 3, 3):
            raise ValueError("intrinsics must have shape [B,6,3,3] matching features.")
        if ego_to_camera.shape != (batch_size, cameras, 4, 4):
            raise ValueError("ego_to_camera must have shape [B,6,4,4] matching features.")
        if not all(tensor.is_floating_point() for tensor in (features, intrinsics, ego_to_camera)):
            raise TypeError("Features and camera calibration must be floating-point tensors.")
        if not (features.device == intrinsics.device == ego_to_camera.device):
            raise ValueError("Features and camera calibration must be on the same device.")

        if depth_fusion not in ('legacy', 'residual', 'gated', 'probability'):
            raise ValueError('Unknown depth fusion mode.')
        if depth_fusion in ('residual', 'gated', 'probability') and depth_probs is None:
            raise ValueError('Depth fusion requires depth probabilities.')

        if depth_probs is not None:
            if (depth_probs.ndim != 5 or depth_probs.shape[:2] != (batch_size, cameras)
                    or depth_probs.shape[2] <= 0
                    or depth_probs.shape[-2:] != (feature_height, feature_width)):
                raise ValueError("depth_probs must be [B,6,D,h,w] matching image features.")
            if depth_edges is None or depth_edges.shape != (depth_probs.shape[2] + 1,):
                raise ValueError("Provide D+1 uniform depth_edges.")
            if depth_probs.device != features.device or depth_edges.device != features.device:
                raise ValueError("Depth inputs must be on the features device.")
            widths = depth_edges[1:] - depth_edges[:-1]
            if (not torch.isfinite(depth_edges).all() or not (widths > 0).all()
                    or not torch.allclose(widths, widths[:1].expand_as(widths), atol=1e-5)):
                raise ValueError("Depth edges must be finite, increasing and uniformly spaced.")

        geometry_dtype = (
            torch.float64
            if torch.float64 in (intrinsics.dtype, ego_to_camera.dtype)
            else torch.float32
        )
        intrinsics = intrinsics.to(dtype=geometry_dtype)
        ego_to_camera = ego_to_camera.to(dtype=geometry_dtype)

        probability_volume = (self.make_probability_volume(features, depth_probs)
                              if depth_fusion == 'probability' else None)
        fused_chunks = []
        for centres in self.voxel_centres.split(self.voxel_chunk_size):
            # Avoid low-precision matrix products in the metric projection.
            with torch.autocast(device_type=features.device.type, enabled=False):
                pixels, valid = self.project_voxels(
                    centres, intrinsics, ego_to_camera, image_size
                )
            if probability_volume is not None:
                with torch.autocast(device_type=features.device.type, enabled=False):
                    depths = torch.einsum("bnj,vj->bnv", ego_to_camera[..., 2, :3],
                                          centres.to(geometry_dtype)) + ego_to_camera[..., 2, 3, None]
                    sampled = self.sample_probability_volume(
                        probability_volume, pixels, depths, depth_edges)
                # Average by geometric camera counts, never by depth support.
                # Keep the large voxel output in the context's original dtype.
                fused_chunks.append(self.fuse_features(sampled, valid).to(features.dtype))
                continue
            sampled = self.sample_features(features, pixels, image_size)
            context = self.fuse_features(sampled, valid) if depth_fusion == 'residual' else None
            if depth_probs is not None:
                with torch.autocast(device_type=features.device.type, enabled=False):
                    depths = torch.einsum("bnj,vj->bnv", ego_to_camera[..., 2, :3],
                                          centres.to(geometry_dtype)) + ego_to_camera[..., 2, 3, None]
                support = self.sample_depth_support(depth_probs, pixels, depths, depth_edges)
                if depth_fusion in ('residual', 'gated'):
                    # Uniform depth is neutral (D*p=1); limit amplification by
                    # overconfident pseudo-depth. Keep raw context in a separate
                    # branch so unsupported surfaces retain visual evidence.
                    support = (support * depth_probs.shape[2]).clamp(max=4.0)
                sampled = sampled * support.unsqueeze(2).to(sampled.dtype)
                # Retain the geometric-view-count denominator. Normalizing by
                # support would cancel the gate for a single contributing view.
            fused = self.fuse_features(sampled, valid)
            if depth_fusion == 'residual':
                evidence = self.fuse_features(support.unsqueeze(2).to(sampled.dtype), valid)
                fused = torch.cat((context, fused, evidence), dim=1)
            elif depth_fusion == 'gated':
                # Keep depth evidence explicit without any ungated appearance
                # path. Geometric view counts retain the gate in single views.
                evidence = self.fuse_features(support.unsqueeze(2).to(sampled.dtype), valid)
                fused = torch.cat((fused, evidence), dim=1)
            fused_chunks.append(fused)

        # Concatenation preserves gradients and the original [X,Y,Z] order.
        fused = torch.cat(fused_chunks, dim=-1)  # [B,C,V]
        output_channels = {'legacy': channels, 'residual': 2*channels+1,
                           'gated': channels+1, 'probability': channels+1}[depth_fusion]
        return fused.reshape(batch_size, output_channels, *self.grid_shape)
