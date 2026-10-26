"""BEVDepth's depth/context/refinement structure adapted to dense occupancy.

Architecture reference: https://arxiv.org/abs/2206.10092, section 4, and
Megvii-BaseDetection/BEVDepth's base_lss_fpn.py. This native PyTorch adaptation
uses a narrower ResNet-18 FPN, GroupNorm, voxel pooling that retains height,
and a multiscale occupancy decoder. See BEVDEPTH_STRUCTURE.md for differences.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import DeformConv2d


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


def conv2(cin, cout, kernel=3, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=(kernel//2)*dilation,
                                   dilation=dilation, bias=False), norm(cout), nn.ReLU())


class Residual2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(conv2(channels, channels),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False), norm(channels))

    def forward(self, x):
        return F.relu(x + self.body(x))


class BEVDepthImageEncoder(nn.Module):
    """Trainable four-scale ResNet-18 features aligned to stride-16 centres."""
    def __init__(self, pretrained=True, out_channels=128):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.stages = nn.ModuleList([network.layer1, network.layer2, network.layer3, network.layer4])
        for parameter in self.stem.parameters():
            parameter.requires_grad_(False)
        self.lateral = nn.ModuleList([conv2(c, 32, kernel=1) for c in (64, 128, 256, 512)])
        self.fusion = conv2(128, out_channels)
        self.out_channels = out_channels
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # Six correlated cameras are a small batch. Keep ImageNet BN statistics
        # fixed while training residual-stage weights and BN affine parameters.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        self.stem.eval()
        return self

    def forward(self, images):
        b, n, _, height, width = images.shape
        if n != 6 or images.shape[2] != 3:
            raise ValueError('Expected six RGB cameras.')
        h, w = (height+15)//16, (width+15)//16
        x = self.stem(images.flatten(0, 1))
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        aligned = []
        for stage, lateral, stride in zip(self.stages, self.lateral, (4, 8, 16, 32)):
            x = stage(x)
            value = lateral(x)
            # Preserve the stride*i feature-centre convention of calibration
            # and depth targets, including odd-sized feature maps.
            grid = torch.stack((2*(xx.float()*16/stride+.5)/value.shape[-1]-1,
                                2*(yy.float()*16/stride+.5)/value.shape[-2]-1), -1)
            aligned.append(F.grid_sample(value, grid[None].expand(b*n, -1, -1, -1).to(value.dtype),
                                         align_corners=False, padding_mode='border'))
        return self.fusion(torch.cat(aligned, 1)).reshape(b, n, self.out_channels, h, w)


class CameraGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(27, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                  nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, image_features, camera):
        return image_features * self.gate(self.mlp(camera))[:, :, None, None]


class MultiScaleDepthContext(nn.Module):
    """ASPP: local/dilated image context and global image context."""
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([conv2(channels, channels, kernel=1)] +
                                     [conv2(channels, channels, dilation=d) for d in (6, 12, 18)])
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv2(channels, channels, kernel=1))
        self.merge = nn.Sequential(conv2(5*channels, channels, kernel=1), nn.Dropout(.5))

    def forward(self, x):
        values = [branch(x) for branch in self.branches]
        values.append(self.global_branch(x).expand(-1, -1, *x.shape[-2:]))
        return self.merge(torch.cat(values, 1))


class CameraAwareDepthNet(nn.Module):
    """Camera-conditioned context and depth branches with residual/ASPP/DCN."""
    def __init__(self, context_channels=32, num_bins=64, min_depth=.1, max_depth=64.1,
                 in_channels=128, hidden_channels=128):
        super().__init__()
        if num_bins < 2 or not (math.isfinite(min_depth) and math.isfinite(max_depth)
                                and 0 < min_depth < max_depth):
            raise ValueError('BEVDepth requires at least two bins and finite positive depth bounds.')
        self.num_bins = num_bins
        self.register_buffer('bin_edges', torch.linspace(min_depth, max_depth, num_bins+1))
        self.register_buffer('bin_centres', (self.bin_edges[:-1]+self.bin_edges[1:])/2)
        self.camera_norm = nn.LayerNorm(27)
        self.reduce = conv2(in_channels, hidden_channels)
        self.context_gate = CameraGate(hidden_channels)
        self.depth_gate = CameraGate(hidden_channels)
        self.context = nn.Conv2d(hidden_channels, context_channels, 1)
        self.depth_features = nn.Sequential(*[Residual2D(hidden_channels) for _ in range(3)],
                                           MultiScaleDepthContext(hidden_channels))
        self.offsets = nn.Conv2d(hidden_channels, 2*3*3*4, 3, padding=1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.deform = DeformConv2d(hidden_channels, hidden_channels, 3, padding=1, groups=4)
        self.classifier = nn.Conv2d(hidden_channels, num_bins, 1)

    def camera_vector(self, intrinsics, ego_to_camera, image_size):
        h, w = image_size
        with torch.autocast(device_type=intrinsics.device.type, enabled=False):
            k = intrinsics.float()
            camera_to_ego = torch.linalg.inv(ego_to_camera.float())[..., :3, :].clone()
            camera_to_ego[..., :3, 3] /= 50.
            focal = torch.stack((k[..., 0, 0]/w, k[..., 1, 1]/h,
                                 k[..., 0, 2]/w, k[..., 1, 2]/h), -1)
            # Resizing is already composed into K; there is no independent
            # image/BEV augmentation matrix in this dataset pipeline.
            identity_aug = focal.new_tensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1])
            identity_aug = identity_aug.expand(*focal.shape[:-1], 11)
            vector = torch.cat((focal, identity_aug, camera_to_ego.flatten(-2)), -1)
        return self.camera_norm(vector.flatten(0, 1))

    def forward(self, features, intrinsics, ego_to_camera, image_size):
        b, n, _, h, w = features.shape
        camera = self.camera_vector(intrinsics, ego_to_camera, image_size)
        reduced = self.reduce(features.flatten(0, 1))
        context = self.context(self.context_gate(reduced, camera))
        depth = self.depth_features(self.depth_gate(reduced, camera))
        # Native torchvision DCN has no MMCV dependency; FP32 keeps its CUDA
        # forward/backward portable across mixed-precision implementations.
        with torch.autocast(device_type=depth.device.type, enabled=False):
            depth = self.deform(depth.float(), self.offsets(depth.float()))
        logits = self.classifier(depth)
        return logits.reshape(b, n, self.num_bins, h, w), context.reshape(b, n, -1, h, w)


class FrustumRefinement(nn.Module):
    """Convolve over depth/width for each image row before voxel pooling."""
    def __init__(self, channels, row_chunk=16):
        super().__init__()
        self.reduce = conv2(channels, channels)
        self.residual = nn.Sequential(conv2(channels, channels), conv2(channels, channels))
        self.output = nn.Conv2d(channels, channels, 3, padding=1)
        self.row_chunk = row_chunk

    def refine_rows(self, rows):
        with torch.autocast(device_type=rows.device.type, enabled=False):
            reduced = self.reduce(rows.float())
            return self.output(reduced + self.residual(reduced))

    def forward(self, volume):
        b, n, c, d, h, w = volume.shape
        rows = volume.permute(0, 1, 4, 2, 3, 5).reshape(b*n*h, c, d, w)
        values = []
        for chunk in rows.split(self.row_chunk):
            if self.training and torch.is_grad_enabled() and chunk.requires_grad:
                values.append(checkpoint(self.refine_rows, chunk, use_reentrant=False))
            else:
                values.append(self.refine_rows(chunk))
        return torch.cat(values).reshape(b, n, h, c, d, w).permute(0, 1, 3, 4, 2, 5)


class FrustumVoxelPool(nn.Module):
    """Forward unprojection and sum pooling; all occupancy height cells survive."""
    def __init__(self, grid_shape, voxel_bounds, channels=32, refine=True):
        super().__init__()
        from .view_transformer import ViewTransformer
        geometry = ViewTransformer(grid_shape=grid_shape, voxel_bounds=voxel_bounds)
        self.grid_shape = tuple(grid_shape)
        self.voxel_bounds = tuple(voxel_bounds)
        self.register_buffer('voxel_centres', geometry.voxel_centres)
        self.refinement = FrustumRefinement(channels) if refine else nn.Identity()

    def forward(self, context, probabilities, depths, intrinsics, ego_to_camera):
        b, n, c, h, w = context.shape
        if probabilities.shape != (b, n, depths.numel(), h, w):
            raise ValueError('Depth probabilities must match context and depth centres.')
        with torch.autocast(device_type=context.device.type, enabled=False):
            # BEVDepth's outer product. No D multiplier, clipping, raw branch,
            # division by depth support, or inverse projection of every voxel.
            volume = context.float().unsqueeze(3) * probabilities.float().unsqueeze(2)
        volume = self.refinement(volume)
        with torch.no_grad(), torch.autocast(device_type=context.device.type, enabled=False):
            yy, xx = torch.meshgrid(torch.arange(h, device=context.device),
                                    torch.arange(w, device=context.device), indexing='ij')
            pixels = torch.stack((xx*16, yy*16, torch.ones_like(xx)), -1).float().reshape(-1, 3)
            rays = torch.einsum('bnij,pj->bnpi', torch.linalg.inv(intrinsics.float()), pixels)
            rays = rays / rays[..., 2:3]
            camera_points = rays[:, :, None] * depths.float()[None, None, :, None, None]
            inverse = torch.linalg.inv(ego_to_camera.float())
            points = torch.einsum('bnij,bndpj->bndpi', inverse[..., :3, :3], camera_points)
            points = points + inverse[..., :3, 3][:, :, None, None, :]
            low = points.new_tensor(self.voxel_bounds[:3])
            high = points.new_tensor(self.voxel_bounds[3:])
            size = (high-low)/points.new_tensor(self.grid_shape)
            valid = torch.isfinite(points).all(-1) & ((points >= low) & (points < high)).all(-1)
            indices = torch.floor((points-low)/size).long()
            # Float32 subtraction/division can round an interior point next
            # to the upper boundary to grid_shape (e.g. 200). The metric
            # bounds above decide inclusion; constrain eligible indices to
            # the edge cell before flattening to prevent overflow/axis wrap.
            indices = indices.clamp_min(0)
            indices = torch.minimum(indices, indices.new_tensor(self.grid_shape)-1)
            flat = (indices[..., 0]*self.grid_shape[1]+indices[..., 1])*self.grid_shape[2]+indices[..., 2]
        values = volume.permute(0, 2, 1, 3, 4, 5).reshape(b, c, -1).float()
        cubes = []
        for batch in range(b):
            keep = valid[batch].flatten()
            cube = values.new_zeros(c, math.prod(self.grid_shape))
            cube.index_add_(1, flat[batch].flatten()[keep], values[batch, :, keep])
            cubes.append(cube)
        return torch.stack(cubes).reshape(b, c, *self.grid_shape).to(context.dtype)


class Residual3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1, bias=False), norm(channels),
                                  nn.ReLU(), nn.Conv3d(channels, channels, 3, padding=1, bias=False), norm(channels))

    def forward(self, x):
        return F.relu(x+self.body(x))


class MultiScaleOccupancyDecoder(nn.Module):
    """BEV encoder/neck role adapted to 3D, retaining 16 height layers."""
    def __init__(self, channels=32, num_classes=18):
        super().__init__()
        self.full = Residual3D(channels)
        self.half_stage = nn.Sequential(nn.Conv3d(channels, 2*channels, 3, stride=(2, 2, 1), padding=1),
                                  norm(2*channels), nn.ReLU(), Residual3D(2*channels))
        self.quarter = nn.Sequential(nn.Conv3d(2*channels, 4*channels, 3, stride=(2, 2, 1), padding=1),
                                     norm(4*channels), nn.ReLU(), Residual3D(4*channels))
        self.up_half = nn.Conv3d(4*channels, 2*channels, 1)
        self.up_full = nn.Conv3d(2*channels, channels, 1)
        self.refine = Residual3D(channels)
        self.classifier = nn.Conv3d(channels, num_classes, 1)

    def forward(self, x):
        full = self.full(x)
        half = self.half_stage(full)
        quarter = self.quarter(half)
        merged = half + F.interpolate(self.up_half(quarter), size=half.shape[-3:], mode='trilinear', align_corners=False)
        merged = full + F.interpolate(self.up_full(merged), size=full.shape[-3:], mode='trilinear', align_corners=False)
        return self.classifier(self.refine(merged))

