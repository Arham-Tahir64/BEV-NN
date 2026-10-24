"""Connect the image encoder, view transformer, and 3D decoder."""
from __future__ import annotations

import torch
from torch import nn

from .image_encoder import ImageEncoder
from .view_transformer import ViewTransformer
from .occupancy_decoder import OccupancyDecoder
from .depth_head import DepthHead


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

    Optional normalized XYZ coordinates are embedded into the existing feature
    channels before decoding; the decoder architecture stays unchanged.

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
        use_xyz: bool = False,
        use_depth: bool = False,
        depth_bins: int = 64,
        depth_min: float = 0.1,
        depth_max: float = 64.1,
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

        self.use_xyz = use_xyz
        if use_xyz:
            # Match [X,Y,Z] voxel order and normalize metric bounds to [-1,1].
            centres = self.view_transformer.voxel_centres
            minimum = centres.new_tensor(voxel_bounds[:3])
            maximum = centres.new_tensor(voxel_bounds[3:])
            normalized = 2.0 * (centres - minimum) / (maximum - minimum) - 1.0
            coordinates = normalized.T.reshape(1, 3, *grid_shape).contiguous()
            self.register_buffer("voxel_coordinates", coordinates, persistent=False)

            # Add position information without changing the decoder's input
            # channel count, convolutional layers, or hidden channels.
            # Define this AFTER the decoder to preserve its seeded initialization.
            self.xyz_projection = nn.Conv3d(3, feature_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.xyz_projection.weight)
        else:
            self.register_buffer("voxel_coordinates", None, persistent=False)
            self.xyz_projection = None

        # Opt-in: old configurations retain identical parameters and outputs.
        self.depth_head = DepthHead(num_bins=depth_bins, min_depth=depth_min,
                                    max_depth=depth_max) if use_depth else None


    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        ego_to_camera: torch.Tensor,
        *, return_aux: bool = False,
    ):
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
        depth_logits = None
        if self.depth_head is not None:
            image_features, backbone = self.image_encoder(images, return_backbone=True)
            depth_logits = self.depth_head(backbone)
        else:
            image_features = self.image_encoder(images)
        depth_args = {} if depth_logits is None else {
            "depth_probs": depth_logits.float().softmax(dim=2),
            "depth_edges": self.depth_head.bin_edges,
        }
        voxel_features = self.view_transformer(
            image_features, intrinsics, ego_to_camera, image_size, **depth_args
        )  # [B,C,X,Y,Z]
        if self.xyz_projection is not None:
            # The same ego-frame coordinate grid applies to every batch item.
            # Zero initialization starts from the image-only baseline; training
            # learns how much each coordinate contributes to each channel.
            voxel_features = voxel_features + self.xyz_projection(self.voxel_coordinates)
        logits = self.occupancy_decoder(voxel_features)
        if return_aux:
            return {"occupancy_logits": logits, "depth_logits": depth_logits}
        return logits
