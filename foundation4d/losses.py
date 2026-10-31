"""Separate binary and occupied-only semantic supervision; unknown is never free."""
import torch
from torch.nn import functional as F


def height_weights(counts, cap=1.5):
    counts=torch.as_tensor(counts,dtype=torch.float32)
    if counts.ndim!=2 or counts.shape[1]!=2 or not torch.isfinite(counts).all() or (counts<0).any() or not 1<=cap<float('inf'):
        raise ValueError('Expected nonnegative [Z,2] counts and finite cap >=1.')
    positive,negative=counts[:,0],counts[:,1]
    ratio=(negative/positive.clamp_min(1)).clamp(min=1,max=cap)
    norm=(positive*ratio+negative)/(positive+negative).clamp_min(1)
    norm=torch.where(counts.sum(1)>0,norm,torch.ones_like(norm))
    return ratio,norm


def balanced_binary(logits,labels,valid,ratio,norm):
    known=valid&(labels!=255);positive=labels!=17
    if not known.any():return logits.sum()*0
    weight=torch.where(positive,ratio[None,None,None,:],torch.ones_like(ratio)[None,None,None,:])/norm[None,None,None,:]
    loss=F.binary_cross_entropy_with_logits(logits.float(),positive.float(),reduction='none')
    return (loss*weight)[known].mean()


def occupancy_losses(heads, labels, valid_mask, class_weights=None):
    binary=heads['binary_logits'];semantic=heads['semantic_logits']
    if binary.shape!=labels.shape or semantic.shape!=(labels.shape[0],17,*labels.shape[1:]) or valid_mask.shape!=labels.shape:
        raise ValueError('Expected binary [B,X,Y,Z], semantics [B,17,X,Y,Z] and matching labels/mask.')
    if labels.dtype!=torch.long or valid_mask.dtype!=torch.bool: raise TypeError('Expected int64 labels and boolean mask.')
    known=valid_mask&(labels!=255)
    if ((labels[known]<0)|(labels[known]>17)).any(): raise ValueError('Observed labels must be 0..17.')
    occupied=known&(labels!=17)
    count=int(known.sum());occ_count=int(occupied.sum())
    with torch.autocast(device_type=binary.device.type,enabled=False):
        b=binary[known].float()
        binary_loss=F.binary_cross_entropy_with_logits(b,(labels[known]!=17).float()) if count else b.sum()
        s=semantic.movedim(1,-1)[occupied].float()
        if class_weights is not None:
            if class_weights.shape!=(17,) or not torch.isfinite(class_weights).all() or (class_weights<=0).any():
                raise ValueError('Semantic weights must be 17 finite positive values.')
        semantic_loss=F.cross_entropy(s,labels[occupied],weight=class_weights) if occ_count else s.sum()
    denominator=float(occ_count) if class_weights is None else float(class_weights[labels[occupied]].sum())
    return {'binary':binary_loss,'semantic':semantic_loss,'known':count,'occupied':occ_count,'semantic_denominator':denominator}


def observed_occupancy_boundary(labels, valid):
    """Same exact six-neighbor occupied/free endpoint definition as the project."""
    if labels.ndim!=4 or labels.shape!=valid.shape: raise ValueError('Expected [B,X,Y,Z].')
    known=valid&(labels>=0)&(labels<=17);occupied=labels!=17
    boundary=torch.zeros_like(known)
    for dim in (1,2,3):
        a=[slice(None)]*4;b=a.copy();a[dim]=slice(None,-1);b[dim]=slice(1,None)
        a=tuple(a);b=tuple(b)
        edge=known[a]&known[b]&(occupied[a]!=occupied[b])
        boundary[a]|=edge;boundary[b]|=edge
    return boundary


def boundary_binary_loss(logits,labels,valid):
    boundary=observed_occupancy_boundary(labels,valid)
    if not boundary.any():return logits.sum()*0
    return F.binary_cross_entropy_with_logits(logits[boundary].float(),(labels[boundary]!=17).float())


def observed_ray_loss(logits,labels,valid,intrinsics,transforms,bounds,image_hw,max_voxels=8192):
    """Reweight observed ray samples only; never manufacture free/occluded labels.

    Within each stride-16 image cell select nearest sampled annotated surfaces
    and explicitly labelled free samples before them. These are annotation-based
    groups, not physical first-hit or visibility claims. No behind-hit carving.
    """
    from .model import project_points
    terms=[]
    for b in range(labels.shape[0]):
        known=valid[b]&(labels[b]!=255)
        index=known.nonzero()
        if not len(index):continue
        if len(index)>max_voxels:
            index=index[torch.linspace(0,len(index)-1,max_voxels,device=index.device).long()]
        truth=labels[b][tuple(index.T)];occupied=truth!=17
        scale=logits.new_tensor([(bounds[i+3]-bounds[i])/labels.shape[i+1] for i in range(3)])
        points=(index.float()+.5)*scale+logits.new_tensor(bounds[:3])
        uv,visible=project_points(points[:,None],intrinsics[b:b+1],transforms[b:b+1],image_hw)
        camera=torch.einsum('nij,qj->nqi',transforms[b,...,:3,:3].float(),points.float())+transforms[b,...,:3,3].float()[:,None]
        fh,fw=(image_hw[0]+15)//16,(image_hw[1]+15)//16
        selection=torch.zeros(len(index),device=index.device,dtype=torch.bool)
        for n in range(intrinsics.shape[1]):
            good=visible[0,n,:,0];pixel=uv[0,n,:,0]
            ids=((pixel[:,1]/16).floor().long()*fw+(pixel[:,0]/16).floor().long()).clamp(0,fh*fw-1)
            z=camera[n,:,2]
            nearest=z.new_full((fh*fw,),float('inf'))
            hits=good&occupied
            nearest.scatter_reduce_(0,ids[hits],z[hits],reduce='amin',include_self=True)
            depth=nearest[ids];has_hit=torch.isfinite(depth)
            selection|=good&has_hit&((occupied&((z-depth).abs()<=.4))|(~occupied&(z<depth-.4)))
        if selection.any():
            values=logits[b][tuple(index[selection].T)].float();target=occupied[selection].float()
            # Equal group contribution when both known surfaces and free samples exist.
            for positive in (True,False):
                keep=target.bool()==positive
                if keep.any():terms.append(F.binary_cross_entropy_with_logits(values[keep],target[keep]))
    return torch.stack(terms).mean() if terms else logits.sum()*0
