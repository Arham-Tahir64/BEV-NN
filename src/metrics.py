"""Global masked occupancy metrics; rows=true, columns=predicted."""
import torch


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
