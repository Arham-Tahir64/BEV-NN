"""Optional semantic Dice and conservative top-surface BEV supervision.

Free=17, occupied=0..16, ignore=255. All losses exclude unobserved voxels.
Targets are consumed by training only, never by the prediction model.
"""
import torch
from torch.nn import functional as F


def _known_targets(semantics, valid_mask):
    if semantics.shape != valid_mask.shape or semantics.numel() == 0:
        raise ValueError('Labels and mask must have matching, nonempty shapes.')
    if semantics.dtype != torch.int64 or valid_mask.dtype != torch.bool:
        raise TypeError('Expected int64 labels and boolean visibility mask.')
    if semantics.device != valid_mask.device:
        raise ValueError('Labels and mask must be on the same device.')
    known = valid_mask & (semantics != 255)
    if ((semantics[known] < 0) | (semantics[known] > 17)).any():
        raise ValueError('Observed labels must be 0..17 or ignore=255.')
    return known


def semantic_dice_loss(logits, semantics, valid_mask):
    """Macro soft Dice over target-present occupied classes, in FP32.

Free voxels remain in probability denominators, penalizing false occupancy.
Absent occupied classes are omitted from Dice; semantic CE still trains them.
Returns (loss, present_class_count). An all-free/unknown batch returns a
connected zero. Epoch logging averages batch-class contributions, not global
dataset Dice. Avoids a dense one-hot target tensor.
"""
    if logits.ndim != semantics.ndim + 1 or logits.shape[1] != 18:
        raise ValueError('Expected 18-class logits with a leading batch dimension.')
    if (logits.shape[0], *logits.shape[2:]) != tuple(semantics.shape):
        raise ValueError('Logits spatial dimensions must match labels.')
    if not logits.is_floating_point() or logits.device != semantics.device:
        raise ValueError('Floating logits must share the target device.')
    known = _known_targets(semantics, valid_mask)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        selected = logits.movedim(1, -1)[known].float()
        labels = semantics[known]
        counts = torch.bincount(labels, minlength=18)
        present = counts[:17] > 0
        class_count = int(present.sum().item())
        if class_count == 0:
            return selected.reshape(-1)[:0].sum(), 0
        probabilities = selected.softmax(-1)
        matching = probabilities.gather(1, labels[:, None]).squeeze(1)
        intersection = matching.new_zeros(18).scatter_add(0, labels, matching)
        denominator = probabilities.sum(0) + counts.float()
        dice = (2 * intersection[:17] + 1e-6) / (denominator[:17] + 1e-6)
        return (1 - dice[present]).mean(), class_count


def make_bev_targets(semantics, valid_mask):
    """Topmost occupied semantic label per XY column, or certified free.

An occupied column is trusted only when every voxel above its highest known
occupied voxel is known. Unknown voxels below that surface do not affect the
topmost label. A free column requires every height voxel to be known/free.
Ambiguous columns receive 255 and are excluded; labels/masks are not mutated.
"""
    if semantics.ndim != 4:
        raise ValueError('Expected [B,X,Y,Z] semantics.')
    known = _known_targets(semantics, valid_mask)
    z = torch.arange(semantics.shape[-1], device=semantics.device)
    occupied = known & (semantics != 17)
    top = torch.where(occupied, z, -1).amax(-1)
    has_surface = top >= 0
    unknown_above = ((~known) & (z > top.unsqueeze(-1))).any(-1)
    surface_valid = has_surface & ~unknown_above
    free_valid = ~has_surface & known.all(-1)
    selected = semantics.gather(-1, top.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    targets = torch.full_like(top, 255)
    targets = torch.where(surface_valid, selected, targets)
    targets = torch.where(free_valid, 17, targets)
    return targets, surface_valid | free_valid


def bev_auxiliary_loss(logits, semantics, valid_mask):
    """Unweighted BEV semantic CE; returns (mean_loss, trusted_column_count)."""
    targets, valid = make_bev_targets(semantics, valid_mask)
    if logits.shape != (semantics.shape[0], 18, *semantics.shape[1:3]):
        raise ValueError('Expected BEV logits [B,18,X,Y].')
    if not logits.is_floating_point() or logits.device != semantics.device:
        raise ValueError('Floating BEV logits must share the target device.')
    count = int(valid.sum().item())
    with torch.autocast(device_type=logits.device.type, enabled=False):
        selected = logits.movedim(1, -1)[valid].float()
        if count == 0:
            return selected.sum(), 0
        return F.cross_entropy(selected, targets[valid]), count
