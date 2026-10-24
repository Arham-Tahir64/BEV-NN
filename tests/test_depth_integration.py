"""Depth wiring: ray geometry, unknown masking, gating and gradients."""
import unittest
import numpy as np
import torch

from data.depth_targets import first_surface_depth, depth_loss
from src.models.occupancy_model import OccupancyModel
from src.models.view_transformer import ViewTransformer


class DepthIntegrationTests(unittest.TestCase):
    def test_gate_is_not_cancelled_and_outside_depth_is_zero(self):
        vt = ViewTransformer(grid_shape=(1, 1, 3),
            voxel_bounds=(-.1, -.1, .5, .1, .1, 3.5), voxel_chunk_size=1)
        features = torch.ones(1, 6, 1, 1, 1)
        k = torch.eye(3).repeat(1, 6, 1, 1)
        transform = torch.eye(4).repeat(1, 6, 1, 1)
        transform[:, 1:, 2, 3] = -100  # Only one camera can see these voxels.
        p = torch.tensor([.8, .2]).reshape(1, 1, 2, 1, 1).repeat(1, 6, 1, 1, 1)
        p.requires_grad_()
        result = vt(features, k, transform, (16, 16), depth_probs=p,
                    depth_edges=torch.tensor([.5, 1.5, 2.5]))
        torch.testing.assert_close(result.flatten(), torch.tensor([.8, .2, 0.]))
        result.sum().backward()
        self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_first_surface_and_unknown_blocker(self):
        labels = np.full((1, 1, 4), 17)
        labels[:, :, 2] = 4
        valid = np.ones_like(labels, dtype=bool)
        transform = np.eye(4)[None]
        transform[0, 2, 3] = -.1  # Camera origin ego Z=.1.
        args = (np.eye(3)[None], transform, (16, 16), (-.5, -.5, 0, .5, .5, 4))
        z = first_surface_depth(labels, valid, *args)
        self.assertAlmostEqual(float(z[0, 0, 0]), 1.9, places=5)
        valid[:, :, 1] = False
        self.assertTrue(np.isnan(first_surface_depth(labels, valid, *args)).all())

    def test_loss_masks_unknown_and_uses_correct_boundary_bin(self):
        logits = torch.tensor([1., 3.]).reshape(1, 1, 2, 1, 1).requires_grad_()
        edges = torch.tensor([0., 1., 2.])
        loss, count = depth_loss(logits, torch.ones(1, 1, 1, 1), edges)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(loss.item(), torch.nn.functional.cross_entropy(
            torch.tensor([[1., 3.]]), torch.tensor([1])).item())
        zero, count = depth_loss(logits, torch.full((1, 1, 1, 1), float('nan')), edges)
        self.assertEqual(count, 0)
        zero.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_model_auxiliary_outputs_and_occupancy_gradient(self):
        model = OccupancyModel(pretrained=False, use_depth=True,
            grid_shape=(2, 2, 2), voxel_bounds=(-1, -1, 1, 1, 1, 3))
        images = torch.randn(1, 6, 3, 32, 32)
        k = torch.tensor([[16., 0, 16], [0, 16, 16], [0, 0, 1]]).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        out = model(images, k, t, return_aux=True)
        self.assertEqual(out['depth_logits'].shape, (1, 6, 64, 2, 2))
        out['occupancy_logits'].square().mean().backward()
        grad = model.depth_head.network[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.image_encoder.backbone.parameters()))
        with torch.no_grad():
            self.assertEqual(model(images, k, t).shape, (1, 18, 2, 2, 2))

    def test_legacy_model_has_no_depth_state(self):
        model = OccupancyModel(pretrained=False)
        self.assertFalse(any(k.startswith('depth_head.') for k in model.state_dict()))
        clone = OccupancyModel(pretrained=False)
        clone.load_state_dict(model.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
