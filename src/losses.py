"""Auxiliary geometry supervision from the existing semantic logits."""
import torch
from torch.nn import functional as F


def observed_occupancy_boundary(semantics, valid_mask):
    """Both ends of six-neighbour occupied/free transitions, [B,X,Y,Z].

    Both neighbours must be visible and labelled. Unknown/invisible space and
    the exterior of the grid are never treated as free. Different occupied
    semantic classes do not form a binary geometry boundary.
    """
    if semantics.ndim != 4 or semantics.shape != valid_mask.shape:
        raise ValueError('Boundary labels/mask must be matching [B,X,Y,Z].')
    if semantics.dtype != torch.int64 or valid_mask.dtype != torch.bool:
        raise TypeError('Boundary inputs require int64 labels and boolean mask.')
    if semantics.device != valid_mask.device:
        raise ValueError('Boundary inputs must share a device.')
    known = valid_mask & (semantics >= 0) & (semantics <= 17)
    occupied = semantics != 17
    boundary = torch.zeros_like(known)
    for axis in (1, 2, 3):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        transition = known[left] & known[right] & (occupied[left] != occupied[right])
        boundary[left] |= transition
        boundary[right] |= transition
    return boundary


def boundary_occupied_free_loss(logits, semantics, valid_mask):
    """Additional mean binary CE on observed boundary endpoints only.

    Reuses semantic logits: no extra head, dilation or class balancing. The
    caller adds its own modest coefficient independently of global geometry
    BCE. Count-zero batches have differentiable zero loss.
    """
    return occupied_free_loss(logits, semantics,
                              observed_occupancy_boundary(semantics, valid_mask))


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
