"""Compact calibrated BEV queries -> full XYZ semantic occupancy.

The occupancy representation is inspired by Foundation4D section 3.6.
The encoder alignment follows the existing project's validated stride*i convention.
This implementation contains no NVIDIA Foundation4D model code or checkpoints.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def block(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel//2, bias=False),
                         nn.GroupNorm(math.gcd(8, cout), cout), nn.ReLU())


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(block(channels, channels), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return F.relu(x+self.net(x))


class ImageEncoder(nn.Module):
    def __init__(self, channels, pretrained):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
        self.lateral = nn.ModuleList([block(c, 16, 1) for c in (64,128,256,512)])
        self.fusion = block(64, channels)
        for p in self.stem.parameters(): p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        self.stem.eval()
        return self

    def forward(self, images, return_fine=False):
        b,n,_,height,width = images.shape
        h,w = (height+15)//16,(width+15)//16
        x = self.stem(images.flatten(0,1)); aligned = []
        yy,xx = torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
        for stage,lateral,stride in zip(self.stages,self.lateral,(4,8,16,32)):
            x=stage(x);value=lateral(x)
            if stride==8:fine=value.reshape(b,n,16,*value.shape[-2:])
            grid=torch.stack((2*(xx.float()*16/stride+.5)/value.shape[-1]-1,
                              2*(yy.float()*16/stride+.5)/value.shape[-2]-1),-1)
            aligned.append(F.grid_sample(value,grid[None].expand(b*n,-1,-1,-1).to(value.dtype),align_corners=False))
        coarse=self.fusion(torch.cat(aligned,1)).reshape(b,n,-1,h,w)
        return (coarse,fine) if return_fine else coarse


def project_points(points, intrinsics, ego_to_camera, image_hw):
    """[Q,H,3] -> [B,N,Q,H,2] resized pixel coordinates and validity."""
    with torch.autocast(device_type=intrinsics.device.type,enabled=False):
        camera=torch.einsum('bnij,qhj->bnqhi',ego_to_camera.float()[...,:3,:3],points.float())
        camera=camera+ego_to_camera.float()[...,:3,3][:,:,None,None]
        projected=torch.einsum('bnij,bnqhj->bnqhi',intrinsics.float(),camera)
        uv=projected[...,:2]/camera[...,2:].clamp_min(1e-6)
        h,w=image_hw
        valid=(camera[...,2]>.1)&torch.isfinite(uv).all(-1)&(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)
        uv=torch.nan_to_num(uv,nan=-1e6,posinf=1e6,neginf=-1e6)
        return uv,valid


class PillarProjector(nn.Module):
    def __init__(self, channels, bev_shape, bounds, heights=4, samples=4, chunk=1024, learned_offsets=False):
        super().__init__()
        if samples not in (1,4): raise ValueError('Use one or four samples per height.')
        self.bev_shape=tuple(bev_shape);self.heights=heights;self.samples=samples;self.chunk=chunk
        axes=[torch.linspace(bounds[i],bounds[i+3],bev_shape[i]+1) for i in range(2)]
        x=(axes[0][:-1]+axes[0][1:])/2;y=(axes[1][:-1]+axes[1][1:])/2
        z=torch.arange(heights).float().add(.5)*(bounds[5]-bounds[2])/heights+bounds[2]
        xx,yy,zz=torch.meshgrid(x,y,z,indexing='ij')
        points=torch.stack((xx,yy,zz),-1).reshape(-1,heights,3)
        self.xy_scale=max(abs(float(points[...,0].max())),abs(float(points[...,1].max())),1.)
        self.register_buffer('points',points)
        self.register_buffer('sample_offsets',torch.tensor([[0.,0.]]) if samples==1 else
                             torch.tensor([[-.25,-.25],[.25,-.25],[-.25,.25],[.25,.25]]))
        self.position=nn.Sequential(nn.Linear(2,channels),nn.ReLU(),nn.Linear(channels,channels))
        self.height_embedding=nn.Parameter(torch.zeros(heights,channels))
        self.score=nn.Linear(channels,1)
        self.merge=nn.Linear(heights*channels,channels,bias=False)
        self.offset=nn.Linear(channels,heights*samples*2) if learned_offsets else None
        if self.offset is not None:
            nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)

    def aggregate(self, features, uv, visibility, query):
        """Camera-commutative valid-normalized fusion, height retained until merge."""
        b,n,c,fh,fw=features.shape;q=uv.shape[2];h=self.heights;k=self.samples
        offset=self.sample_offsets[None,None,None].expand(b,q,h,-1,-1)
        if self.offset is not None:
            offset=offset+self.offset(query).reshape(b,q,h,k,2).float().tanh()
        # Cameras are independent grid_sample batches. Fuse them in one call per
        # query chunk instead of six small launches, retaining all camera masks.
        sample_uv=uv[:,:,:,:,None,:]+16*offset[:,None]
        hh,ww=self.image_hw
        valid=(visibility[:,:,:,:,None] & (sample_uv[...,0]>=0)&(sample_uv[...,0]<ww)
               &(sample_uv[...,1]>=0)&(sample_uv[...,1]<hh))
        grid=torch.stack((2*(sample_uv[...,0]/16+.5)/fw-1,2*(sample_uv[...,1]/16+.5)/fh-1),-1)
        grid=torch.where(valid[...,None],grid,grid.new_full((),2.))
        sampled=F.grid_sample(features.reshape(b*n,c,fh,fw),grid.reshape(b*n,q,h*k,2).to(features.dtype),
                              mode='bilinear',padding_mode='zeros',align_corners=False)
        sampled=sampled.reshape(b,n,c,q,h,k).permute(0,1,3,4,5,2)
        sampled_float=sampled.float()
        logits=self.score(sampled).squeeze(-1).float()
        logits=logits+(sampled_float*query[:,None,:,None,None,:].float()).sum(-1)/math.sqrt(c)
        weights=(8*torch.tanh(logits/8)).exp()*valid.float()
        numerator=(sampled_float*weights[...,None]).sum(4).sum(1)
        denominator=weights.sum(4).sum(1).unsqueeze(-1)
        values=numerator/denominator.clamp_min(1e-12)
        present=denominator>0
        values=values+self.height_embedding[None,None].float()*present
        return self.merge(values.flatten(-2).to(features.dtype))

    def forward(self, features, intrinsics, transforms, image_hw, previous=None):
        self.image_hw=image_hw
        b=features.shape[0];total=self.points.shape[0]
        chunks=[]
        for start in range(0,total,self.chunk):
            points=self.points[start:start+self.chunk]
            xy=points[:,0,:2]/self.xy_scale
            query=self.position(xy)[None].expand(b,-1,-1)
            if previous is not None: query=query+previous[:,start:start+self.chunk]
            uv,valid=project_points(points,intrinsics,transforms,image_hw)
            chunks.append(self.aggregate(features,uv,valid,query))
        return torch.cat(chunks,1).transpose(1,2).reshape(b,-1,*self.bev_shape)


class Foundation4DLite(nn.Module):
    """Default forward returns 18 decision scores for existing argmax viewers.

    These are NOT joint semantic probability logits. Training must use
    forward_heads and the separate binary/occupied-semantic objectives.
    """
    def __init__(self, channels=64, bev_shape=(100,100), grid_shape=(200,200,16),
                 voxel_bounds=(-40.,-40.,-1.,40.,40.,5.4), pillar_heights=4,
                 samples=4, query_chunk=1024, layers=2, learned_offsets=False,
                 use_depth=False, depth_bins=64, threshold=.5, pretrained=True, geometry_aware=False):
        super().__init__()
        if (len(grid_shape)!=3 or len(bev_shape)!=2 or any(v<=0 for v in (*grid_shape,*bev_shape))
                or channels<8 or channels%8 or not 0<threshold<1 or layers<1 or query_chunk<1 or pillar_heights<1):
            raise ValueError('Invalid model shape, channels, threshold or projector settings.')
        if len(voxel_bounds)!=6 or any(not math.isfinite(v) for v in voxel_bounds) or any(voxel_bounds[i]>=voxel_bounds[i+3] for i in range(3)):
            raise ValueError('Voxel bounds must be finite increasing XYZ limits.')
        self.grid_shape=tuple(grid_shape);self.voxel_bounds=tuple(voxel_bounds);self.threshold=threshold
        self.geometry_aware=geometry_aware
        if geometry_aware:
            from .geometry import GeometryProjector
            use_depth=True
            pillar_heights=max(8,pillar_heights)
        self.encoder=ImageEncoder(channels,pretrained)
        projector_type=GeometryProjector if geometry_aware else PillarProjector
        self.projectors=nn.ModuleList([projector_type(channels,bev_shape,voxel_bounds,pillar_heights,samples,
                                                     query_chunk,learned_offsets) for _ in range(layers)])
        self.refiners=nn.ModuleList([Residual(channels) for _ in range(layers)])
        self.decoder=nn.Sequential(block(channels,channels),Residual(channels))
        self.binary_head=nn.Conv2d(channels,grid_shape[2],1)
        self.semantic_head=nn.Conv2d(channels,17*grid_shape[2],1)
        self.depth_head=nn.Conv2d(channels,depth_bins,1) if use_depth else None
        self.register_buffer('depth_edges',torch.linspace(.1,64.1,depth_bins+1))

    def forward_heads(self, images, intrinsics, ego_to_camera, camera_mask=None):
        if images.ndim!=5 or images.shape[1:3]!=(6,3): raise ValueError('Expected [B,6,3,H,W] images.')
        b,n=images.shape[:2]
        if intrinsics.shape!=(b,n,3,3) or ego_to_camera.shape!=(b,n,4,4): raise ValueError('Invalid camera calibration shapes.')
        if not (images.device==intrinsics.device==ego_to_camera.device): raise ValueError('Camera inputs must share a device.')
        if not all(value.is_floating_point() for value in (images,intrinsics,ego_to_camera)):
            raise TypeError('Images and camera calibration must be floating point.')
        if camera_mask is not None:
            if camera_mask.shape!=(b,n) or camera_mask.dtype!=torch.bool: raise ValueError('Camera mask must be boolean [B,6].')
            images=torch.where(camera_mask[:,:,None,None,None],images,torch.zeros_like(images))
            # Move masked cameras behind every point, so they carry no sampling support.
            ego_to_camera=ego_to_camera.clone()
            ego_to_camera[...,:3,:3]=torch.where(camera_mask[:,:,None,None],ego_to_camera[...,:3,:3],torch.zeros_like(ego_to_camera[...,:3,:3]))
            ego_to_camera[...,2,3]=torch.where(camera_mask,ego_to_camera[...,2,3],ego_to_camera.new_full((),-1e6))
        if self.geometry_aware:
            features,fine=self.encoder(images,return_fine=True)
            depth=self.depth_head(features.flatten(0,1)).reshape(b,n,-1,*features.shape[-2:])
        else:features=self.encoder(images);depth=None
        bev=None
        for projector,refine in zip(self.projectors,self.refiners):
            previous=None if bev is None else bev.flatten(2).transpose(1,2)
            if self.geometry_aware:
                projected=projector(features,intrinsics,ego_to_camera,images.shape[-2:],previous,
                                    depth_logits=depth,fine=fine,edges=self.depth_edges)
            else:projected=projector(features,intrinsics,ego_to_camera,images.shape[-2:],previous)
            bev=refine(projected if bev is None else bev+projected)
        decoded=self.decoder(F.interpolate(bev,size=self.grid_shape[:2],mode='bilinear',align_corners=False))
        binary=self.binary_head(decoded).permute(0,2,3,1)
        semantic=self.semantic_head(decoded).reshape(b,17,self.grid_shape[2],*self.grid_shape[:2]).permute(0,1,3,4,2)
        if not self.geometry_aware:
            depth=None if self.depth_head is None else self.depth_head(features.flatten(0,1)).reshape(b,n,-1,*features.shape[-2:])
        return {'binary_logits':binary,'semantic_logits':semantic,'depth_logits':depth}

    @staticmethod
    def predict(heads, threshold=.5):
        occupied=heads['binary_logits'].float() >= math.log(threshold/(1-threshold))
        return torch.where(occupied,heads['semantic_logits'].argmax(1),17)

    def forward(self, images, intrinsics, ego_to_camera, *, return_aux=False):
        heads=self.forward_heads(images,intrinsics,ego_to_camera)
        sem=heads['semantic_logits'].float()
        scores=sem-sem.amax(1,keepdim=True)+heads['binary_logits'][:,None].float()-math.log(self.threshold/(1-self.threshold))
        scores=torch.cat((scores,torch.zeros_like(scores[:,:1])),1)
        return {**heads,'occupancy_logits':scores} if return_aux else scores
