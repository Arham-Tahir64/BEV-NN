"""Global masked occupancy metrics; rows=true, columns=predicted."""
import torch
from .losses import observed_occupancy_boundary


def occupancy_boundary_counts(prediction, semantics, valid_mask):
    """Exact six-neighbour binary boundary overlap within observed space.

    Counts matched, predicted and GT boundary voxels; sum across batches before
    computing rates. Semantic instance identity is not evaluated.
    """
    known = valid_mask & (semantics >= 0) & (semantics <= 17)
    truth = observed_occupancy_boundary(semantics, known)
    predicted = observed_occupancy_boundary(prediction, known)
    return torch.stack(((truth & predicted).sum(), predicted.sum(), truth.sum()))


def summarize_boundary_counts(counts):
    matched, predicted, truth = (int(v) for v in counts.detach().cpu().tolist())
    return {'boundary_precision': matched / predicted if predicted else None,
            'boundary_recall': matched / truth if truth else None,
            'boundary_f1': 2*matched/(predicted+truth) if predicted+truth else None,
            'boundary_matched_voxels': matched, 'boundary_predicted_voxels': predicted,
            'boundary_gt_voxels': truth,
            'boundary_metric_rule': 'Exact occupied/free boundary endpoint overlap; '
                'six-neighbour adjacency; both neighbours visible and labelled; '
                'no tolerance or exterior padding. Undefined rates are null.'}


def summarize_confusion(matrix):
    cm = matrix.detach().cpu().to(torch.float64)
    gt, pred = cm.sum(1), cm.sum(0)
    tp = cm.diag()
    union = gt+pred-tp
    iou = tp / union.clamp_min(1)
    occ_tp = cm[:17, :17].sum().item()
    occ_gt, occ_pred = gt[:17].sum().item(), pred[:17].sum().item()
    return {
        # Fixed denominator makes epoch comparisons independent of whether
        # a model happens to predict a class absent from this tiny val split.
        'occupied_class_miou': iou[:17].mean().item(),
        'miou_rule': 'Fixed occupied IDs 0..16; zero-union classes contribute zero.',
        'binary_occupied_iou': occ_tp / max(occ_gt+occ_pred-occ_tp, 1),
        'occupied_precision': occ_tp / max(occ_pred, 1),
        'occupied_recall': occ_tp / max(occ_gt, 1),
        'gt_occupied_voxels': int(occ_gt),
        'predicted_occupied_voxels': int(occ_pred),
        'per_class_iou': iou.tolist(),
        'per_class_recall': (tp / gt.clamp_min(1)).tolist(),
        'per_class_gt_counts': gt.to(torch.int64).tolist(),
        'per_class_predicted_counts': pred.to(torch.int64).tolist(),
        'manmade_iou': iou[15].item(),
        'manmade_recall': (tp[15] / gt[15].clamp_min(1)).item(),
        'missing_supported_classes': [c for c in range(17) if gt[c] > 0 and pred[c] == 0],
    }
