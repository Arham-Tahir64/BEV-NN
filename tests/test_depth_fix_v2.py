import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from data.depth_targets import projected_surface_depth, DepthTargetDataset, depth_loss
from src.models.occupancy_model import OccupancyModel
from src.models.view_transformer import ViewTransformer
from src.metrics import summarize_confusion


class DepthFixTests(unittest.TestCase):
    def test_unknown_origin_does_not_discard_projected_surface(self):
        labels = np.full((1, 1, 4), 17)
        labels[:, :, 2] = 15
        mask = np.zeros_like(labels, dtype=bool)
        mask[:, :, 2] = True
        target = projected_surface_depth(labels, mask, mask, np.eye(3)[None],
            np.eye(4)[None], (16, 16), (-.5, -.5, 0, .5, .5, 4))
        self.assertAlmostEqual(target[0, 0, 0], 2.5)
        empty = projected_surface_depth(labels, np.zeros_like(mask), mask, np.eye(3)[None],
            np.eye(4)[None], (16, 16), (-.5, -.5, 0, .5, .5, 4))
        self.assertTrue(np.isnan(empty).all())

    def test_invisible_near_surface_blocks_far_visible_surface(self):
        labels = np.full((1, 1, 4), 17)
        labels[:, :, 1:3] = 15
        observed = labels != 17
        visible = np.zeros_like(observed)
        visible[:, :, 2] = True
        target = projected_surface_depth(labels, observed, visible, np.eye(3)[None],
            np.eye(4)[None], (16, 16), (-.5, -.5, 0, .5, .5, 4))
        self.assertTrue(np.isnan(target).all())

    def test_behind_camera_and_range_are_masked(self):
        labels = np.full((1, 1, 4), 15)
        mask = np.ones_like(labels, dtype=bool)
        for bounds in ((-.5, -.5, -4, .5, .5, 0), (-.5, -.5, 65, .5, .5, 69)):
            target = projected_surface_depth(labels, mask, mask, np.eye(3)[None],
                np.eye(4)[None], (16, 16), bounds)
            self.assertTrue(np.isnan(target).all())

    def test_uniform_depth_preserves_context_scale(self):
        vt = ViewTransformer(grid_shape=(1, 1, 1), voxel_bounds=(-.1, -.1, .5, .1, .1, 1.5))
        feature = torch.full((1, 6, 2, 1, 1), 3.)
        k = torch.eye(3).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        p = torch.full((1, 6, 64, 1, 1), 1/64)
        edges = torch.linspace(.1, 64.1, 65)
        evidence = vt(feature, k, t, (16, 16), depth_probs=p,
                      depth_edges=edges, depth_fusion='residual')
        torch.testing.assert_close(evidence[:, :2], evidence[:, 2:4])
        model = OccupancyModel(pretrained=False, use_depth=True, depth_fusion='residual', feature_channels=2)
        torch.testing.assert_close(model.depth_context_projection(evidence), evidence[:, :2])
        p.zero_()
        evidence = vt(feature, k, t, (16, 16), depth_probs=p,
                      depth_edges=edges, depth_fusion='residual')
        torch.testing.assert_close(model.depth_context_projection(evidence), evidence[:, :2]*.9)

    def test_linear_loss_at_bin_boundary_has_equal_targets(self):
        logits = torch.zeros(1, 1, 2, 1, 1, requires_grad=True)
        loss, count = depth_loss(logits, torch.ones(1, 1, 1, 1), torch.tensor([0., 1., 2.]), mode='linear')
        self.assertEqual(count, 1)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_metrics_detect_missing_manmade_and_do_not_reward_free_collapse(self):
        cm = torch.zeros(18, 18, dtype=torch.int64)
        cm[15, 17] = 10
        cm[17, 17] = 90
        stats = summarize_confusion(cm)
        self.assertEqual(stats['occupied_class_miou'], 0)
        self.assertEqual(stats['manmade_recall'], 0)
        self.assertEqual(stats['missing_supported_classes'], [15])
        cm[15, 17] = 0
        cm[15, 15] = 10
        stats = summarize_confusion(cm)
        self.assertEqual(stats['occupied_precision'], 1)
        self.assertEqual(stats['occupied_recall'], 1)
        self.assertEqual(stats['manmade_iou'], 1)
        self.assertAlmostEqual(stats['occupied_class_miou'], 1/17)


if __name__ == '__main__':
    unittest.main()
