"""Connect the image encoder, view transformer, and 3D decoder."""
from __future__ import annotations

import torch
from torch import nn

from .image_encoder import ImageEncoder
from .view_transformer import ViewTransformer
from .occupancy_decoder import OccupancyDecoder


class OccupancyModel(nn.Module):
    """Predict semantic occupancy from six normalized RGB camera images.

    Inputs:
        images: [B,6,3,H,W], already resized and ImageNet-normalized.
        intrinsics: [B,6,3,3], adjusted for the images' current resolution.
        ego_to_camera: [B,6,4,4], from sample ego frame to camera frame.
        Camera order must match across all three inputs.

    Output:
        Raw class logits [B,num_classes,X,Y,Z].
        Default: [B,18,200,200,16].

    Pipeline:
        ImageEncoder -> ViewTransformer -> OccupancyDecoder.

    Labels and supervision masks are used by the training loss, not passed
    into the model. Do not apply softmax before cross-entropy loss.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        hidden_channels: int = 32,
        num_classes: int = 18,
        pretrained: bool = True,
        grid_shape: tuple[int, int, int] = (200, 200, 16),
        voxel_bounds: tuple[float, float, float, float, float, float] = (
            -40.0, -40.0, -1.0, 40.0, 40.0, 5.4
        ),
        voxel_chunk_size: int = 65536,
    ) -> None:
        super().__init__()

        # The encoder handles freezing its ResNet backbone internally.
        self.image_encoder = ImageEncoder(
            out_channels=feature_channels,
            pretrained=pretrained,
        )
        self.view_transformer = ViewTransformer(
            grid_shape=grid_shape,
            voxel_bounds=voxel_bounds,
            voxel_chunk_size=voxel_chunk_size,
        )
        self.occupancy_decoder = OccupancyDecoder(
            in_channels=feature_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        ego_to_camera: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-voxel class logits while preserving [X,Y,Z] order."""
        if images.ndim != 5:
            raise ValueError("images must have shape [B,6,3,H,W].")
        batch_size = images.shape[0]
        if intrinsics.shape != (batch_size, 6, 3, 3):
            raise ValueError("intrinsics must have shape [B,6,3,3] matching images.")
        if ego_to_camera.shape != (batch_size, 6, 4, 4):
            raise ValueError("ego_to_camera must have shape [B,6,4,4] matching images.")
        if not all(tensor.is_floating_point() for tensor in (images, intrinsics, ego_to_camera)):
            raise TypeError("Images and camera calibration must be floating-point tensors.")
        if not (images.device == intrinsics.device == ego_to_camera.device):
            raise ValueError("Images and camera calibration must be on the same device.")

        image_size = tuple(images.shape[-2:])  # (H, W) of the resized input images.
        image_features = self.image_encoder(images)  # [B,6,C,h,w]
        voxel_features = self.view_transformer(
            image_features, intrinsics, ego_to_camera, image_size
        )  # [B,C,X,Y,Z]
        return self.occupancy_decoder(voxel_features)  # Raw logits [B,K,X,Y,Z].
