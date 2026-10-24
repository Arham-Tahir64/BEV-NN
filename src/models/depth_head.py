"""Shared image-space depth head for a future depth-conditioned voxel lift.

This module is not yet connected to OccupancyModel. Feed it ResNet layer3
features BEFORE the encoder's 256-to-32 context projection:

    backbone_features: [B, 6, 256, h, w]
    depth_logits = head(backbone_features)       # [B, 6, D, h, w]
    depth_probabilities = depth_logits.float().softmax(dim=2)

Depth is camera-space Z in metres, not distance from the ego origin. The
default uniform bin edges cover [0.1, 64.1] metres; these are provisional
experiment settings to validate against training depths.

Integration still required:
  * Expose backbone features from ImageEncoder and instantiate this branch.
  * Generate/cache trustworthy first-surface targets; mask unknown rays and
    out-of-range depths rather than clamping them into the end bins.
  * Train logits with a masked depth loss alongside the occupancy objective.
  * Sample probabilities at (u, v, camera_Z) in ViewTransformer, using the
    actual bin centres and chunked sampling rather than expanding D per voxel.
  * Preserve depth-support magnitude in fusion; dividing by summed support
    would cancel the gate in regions seen by only one camera.

This is a lightweight BEVDepth-inspired building block, not the full BEVDepth
architecture. No depth targets or ground-truth masks are inference inputs.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class DepthHead(nn.Module):
    """Predict raw depth-bin logits with weights shared across camera views.

    Input:  [B, N, in_channels, h, w], normally N=6 and h,w=20,36.
    Output: [B, N, num_bins, h, w]. Softmax belongs along dimension 2.

    GroupNorm avoids dependence on batch statistics for small training batches.
    The caller owns masking, loss reduction, and depth-to-bin conversion.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        num_bins: int = 64,
        min_depth: float = 0.1,
        max_depth: float = 64.1,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("hidden_channels", hidden_channels),
            ("num_bins", num_bins),
            ("norm_groups", norm_groups),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if hidden_channels % norm_groups:
            raise ValueError("hidden_channels must be divisible by norm_groups.")
        if not (math.isfinite(min_depth) and math.isfinite(max_depth)
                and 0 < min_depth < max_depth):
            raise ValueError("Depth bounds must be finite and satisfy 0 < min < max.")

        self.in_channels = in_channels
        self.num_bins = num_bins
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        edges = torch.linspace(min_depth, max_depth, num_bins + 1)
        self.register_buffer("bin_edges", edges)
        self.register_buffer("bin_centres", (edges[:-1] + edges[1:]) / 2)

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_bins, kernel_size=1),
        )

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """Return logits without detaching the shared backbone's gradients."""
        if backbone_features.ndim != 5:
            raise ValueError("Expected backbone_features shaped [B,N,C,h,w].")
        batch, cameras, channels, height, width = backbone_features.shape
        if min(backbone_features.shape) <= 0:
            raise ValueError("All feature dimensions must be positive.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        if not backbone_features.is_floating_point():
            raise TypeError("backbone_features must be floating point.")

        features = backbone_features.reshape(batch * cameras, channels, height, width)
        logits = self.network(features)
        return logits.reshape(batch, cameras, self.num_bins, height, width)
