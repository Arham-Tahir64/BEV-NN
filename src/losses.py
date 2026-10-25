"""Auxiliary geometry supervision from the existing semantic logits."""
import torch
from torch.nn import functional as F


def occupied_free_loss(logits, semantics, valid_mask):
    """Unweighted binary CE over observed voxels; free=17, ignore=255.

    logsumexp(occupied logits) - free logit is the binary occupied log-odds.
    BCEWithLogits avoids subtracting a saturated softmax probability from one.
    Every supervised voxel has equal weight, independent of semantic weights.
    Returns (mean_loss, count); unknown/invisible voxels have zero gradient.
    No parameters or inference computations are added.
    """
    if logits.ndim != 5 or logits.shape[1] != 18:
        raise ValueError('Occupancy logits must be [B,18,X,Y,Z].')
    expected = (logits.shape[0], *logits.shape[2:])
    if semantics.shape != expected or valid_mask.shape != expected:
        raise ValueError('Labels and mask must match logits spatial dimensions.')
    if not logits.is_floating_point() or semantics.dtype != torch.int64 or valid_mask.dtype != torch.bool:
        raise TypeError('Expected floating logits, int64 labels, and boolean mask.')
    if not (logits.device == semantics.device == valid_mask.device):
        raise ValueError('Loss inputs must share a device.')
    keep = valid_mask & (semantics != 255)
    count = int(keep.sum().item())
    if not count:
        return logits.float().sum()*0, 0
    labels = semantics[keep]
    if ((labels < 0) | (labels > 17)).any():
        raise ValueError('Supervised occupancy IDs must be 0 through 17.')
    selected = logits.permute(0, 2, 3, 4, 1)[keep].float()
    # Keep the reduction stable under CUDA AMP, including extreme confidence.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        occupied_log_odds = selected[:, :17].logsumexp(-1) - selected[:, 17]
        loss = F.binary_cross_entropy_with_logits(occupied_log_odds, (labels != 17).float())
    return loss, count
