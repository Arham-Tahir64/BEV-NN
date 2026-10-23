"""Shared ResNet-18 image encoder scaffold."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    """Extract spatial features from six normalized RGB camera images.

    Input:  [B, 6, 3, H, W].
    Output: [B, 6, out_channels, ceil(H/32), ceil(W/32)].

    For 320x576 images and 64 output channels: [B, 6, 64, 10, 18].
    Inputs are already resized and ImageNet-normalized by the dataset.
    Use one shared backbone for all six cameras.
    """

    output_stride = 32
    num_cameras = 6

    def __init__(self, out_channels: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        self.out_channels = out_channels

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.projection = nn.Conv2d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-camera feature maps while preserving camera order."""
        if images.ndim != 5:
            raise ValueError("Expected images shaped [B, 6, 3, H, W].")

        B, N, C, H, W = images.shape

        if N != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {N}.")

        if C != 3:
            raise ValueError(f"Expected 3 RGB channels, got {C}.")

        if B < 1 or H < 1 or W < 1:
            raise ValueError("Batch size, height, and width must be positive.")

        # Flatten batch and camera dimensions for the shared backbone.
        images = images.view(B * N, C, H, W)

        # Extract spatial features and project to out_channels.
        features = self.backbone(images)
        features = self.projection(features)

        # Restore the batch and camera dimensions.
        _, channels, h, w = features.shape
        features = features.reshape(B, N, channels, h, w)
        return features
