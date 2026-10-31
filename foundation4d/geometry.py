"""Depth-aware evidence fusion. Labels never enter this inference module."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import PillarProjector,project_points


def supported_mean(values,weights,support):
    # Appearance weights sum to one across observations. A unit null mass keeps
    # weak depth evidence weak even with one camera: suppression cannot cancel.
    evidence=weights*support
    mass=evidence.sum((1,4))
    mean=(values*evidence[...,None]).sum((1,4))/(1+mass[...,None])
    second=(values.square()*evidence[...,None]).sum((1,4))/(1+mass[...,None])
    variance=(second-mean.square()).clamp_min(0).mean(-1)
    return mean,mass,variance


class GeometryProjector(PillarProjector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.height_embedding.shape[1]
        self.fine_projection=nn.Linear(16,c,bias=False)
        self.evidence_projection=nn.Linear(6,c,bias=False)
        self.vertical=nn.Conv1d(c,c,3,padding=1,groups=c,bias=False)

    def forward(self,features,intrinsics,transforms,image_hw,previous=None,*,depth_logits,fine,edges):
        b,n,c,fh,fw=features.shape;h=self.heights;k=self.samples
        probabilities=depth_logits.float().softmax(2)
        # Local probability mass over three depth bins, not a hard estimated hit.
        d=probabilities.shape[2]
        surface=F.avg_pool3d(probabilities.reshape(b*n,1,d,fh,fw),(3,1,1),stride=1,padding=(1,0,0))*3
        cumulative=probabilities.cumsum(2).reshape(b*n,1,d,fh,fw)
        entropy=-(probabilities*probabilities.clamp_min(1e-8).log()).sum(2)/math.log(d)
        outputs=[]
        for start in range(0,len(self.points),self.chunk):
            points=self.points[start:start+self.chunk];q=len(points)
            query=self.position(points[:,0,:2]/self.xy_scale)[None].expand(b,-1,-1)
            if previous is not None:query=query+previous[:,start:start+q]
            uv,visible=project_points(points,intrinsics,transforms,image_hw)
            with torch.autocast(device_type=features.device.type,enabled=False):
                camera=torch.einsum('bnij,qhj->bnqhi',transforms.float()[...,:3,:3],points.float())+transforms.float()[...,:3,3][:,:,None,None]
                z=camera[...,2]
            offset=self.sample_offsets[None,None,None].expand(b,q,h,-1,-1)
            if self.offset is not None:offset=offset+self.offset(query).reshape(b,q,h,k,2).float().tanh()
            pixel=uv[...,None,:]+16*offset[:,None]
            valid=visible[...,None]&(pixel[...,0]>=0)&(pixel[...,0]<image_hw[1])&(pixel[...,1]>=0)&(pixel[...,1]<image_hw[0])
            grid=torch.stack((2*(pixel[...,0]/16+.5)/fw-1,2*(pixel[...,1]/16+.5)/fh-1),-1)
            grid=torch.where(valid[...,None],grid,grid.new_full((),2))
            def sample2(value,stride):
                hh,ww=value.shape[-2:]
                g=torch.stack((2*(pixel[...,0]/stride+.5)/ww-1,2*(pixel[...,1]/stride+.5)/hh-1),-1)
                g=torch.where(valid[...,None],g,g.new_full((),2))
                v=F.grid_sample(value.flatten(0,1),g.reshape(b*n,q,h*k,2).to(value.dtype),align_corners=False)
                return v.reshape(b,n,-1,q,h,k).permute(0,1,3,4,5,2)
            values=sample2(features,16)+self.fine_projection(sample2(fine,8))
            depth_grid=2*(z-edges[0])/(edges[-1]-edges[0])-1
            volume_grid=torch.cat((grid,depth_grid[...,None,None].expand(b,n,q,h,k,1)),-1).reshape(b*n,q,h,k,3)
            def sample3(volume):
                return F.grid_sample(volume,volume_grid.float(),align_corners=False).reshape(b,n,q,h,k)
            in_range=((z>=edges[0])&(z<edges[-1]))[...,None]
            support=sample3(surface)*valid*in_range
            behind=sample3(cumulative)*valid*in_range
            ent=sample2(entropy[:,:,None],16)[...,0]
            score=self.score(values).squeeze(-1).float()+(values.float()*query[:,None,:,None,None].float()).sum(-1)/math.sqrt(c)
            raw=(8*torch.tanh(score/8)).exp()*valid
            weights=raw/raw.sum((1,4),keepdim=True).clamp_min(1e-8)
            mean,mass,variance=supported_mean(values.float(),weights,support)
            diagnostics=torch.stack((mass,variance,valid.any(-1).float().mean(1),
                                     (weights*ent).sum((1,4)),(weights*behind).sum((1,4)),
                                     (weights*in_range).sum((1,4))),-1)
            present=valid.any(1).any(-1)
            retained=mean+self.evidence_projection(diagnostics.to(features.dtype)).float()+self.height_embedding[None,None]*present[...,None]
            retained=retained*present[...,None]
            mixed=retained+self.vertical(retained.reshape(b*q,h,c).transpose(1,2).to(features.dtype)).transpose(1,2).reshape(b,q,h,c)
            outputs.append(self.merge(mixed.flatten(-2).to(features.dtype)))
        return torch.cat(outputs,1).transpose(1,2).reshape(b,c,*self.bev_shape)
