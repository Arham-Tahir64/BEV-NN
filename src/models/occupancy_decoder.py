"""Predict semantic occupancy from fused 3D voxel features."""
from __future__ import annotations

import torch
from torch import nn


class OccupancyDecoder(nn.Module):
    """Convert voxel features into per-voxel semantic class logits.

    Input:
        voxel_features: [B, in_channels, X, Y, Z].
        Default example: [B, 32, 200, 200, 16].

    Output:
        Raw logits: [B, num_classes, X, Y, Z].
        Default example: [B, 18, 200, 200, 16].

    Preserve the dataset's [X, Y, Z] axis order and spatial resolution.
    Use padding=1 for stride-1, 3x3x3 convolutions.
    Class IDs are 0..17, including free space at 17. Ignore ID 255 is
    handled by the loss, rather than added as an output class.
    Do not apply softmax before cross-entropy loss.
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 32,
        num_classes: int = 18,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        for name, value in (
            ("in_channels", in_channels),
            ("hidden_channels", hidden_channels),
            ("num_classes", num_classes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        # Combine neighbouring voxel features without changing grid resolution.
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Produce raw class scores independently at each voxel.
        self.classifier = nn.Conv3d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        """Return raw class logits while preserving the [X,Y,Z] dimensions."""
        if voxel_features.ndim != 5:
            raise ValueError("voxel_features must have shape [B,in_channels,X,Y,Z].")
        if voxel_features.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} feature channels, got {voxel_features.shape[1]}."
            )
        if any(size <= 0 for size in voxel_features.shape):
            raise ValueError("Batch size and voxel dimensions must be positive.")
        if not voxel_features.is_floating_point():
            raise TypeError("voxel_features must be a floating-point tensor.")

        features = self.decoder(voxel_features)
        return self.classifier(features)  # Raw logits [B,num_classes,X,Y,Z].
