"""Separate binary and occupied-only semantic supervision; unknown is never free."""
import torch
from torch.nn import functional as F


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
