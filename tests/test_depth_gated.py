"""Behavioral contracts for the controlled depth-only appearance experiment."""
import unittest
import torch
from src.models.view_transformer import ViewTransformer
from src.models.occupancy_model import OccupancyModel


class GatedDepthTests(unittest.TestCase):
    def geometry(self):
        vt = ViewTransformer(grid_shape=(1, 1, 3),
            voxel_bounds=(-.1, -.1, .5, .1, .1, 3.5), voxel_chunk_size=1)
        k = torch.eye(3).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        t[:, 1:, 2, 3] = -100
        return vt, k, t

    def test_uniform_gate_is_neutral_and_single_camera_gate_is_not_cancelled(self):
        vt, k, t = self.geometry()
        f = torch.full((1, 6, 1, 1, 1), 3.)
        p = torch.tensor([.5, .5]).reshape(1, 1, 2, 1, 1).repeat(1, 6, 1, 1, 1)
        edges = torch.tensor([.5, 1.5, 2.5])
        result = vt(f, k, t, (16, 16), depth_probs=p, depth_edges=edges, depth_fusion='gated')
        torch.testing.assert_close(result[:, 0].flatten(), torch.tensor([3., 3., 0.]))
        torch.testing.assert_close(result[:, 1].flatten(), torch.tensor([1., 1., 0.]))
        p[:, :, 0] = .8
        p[:, :, 1] = .2
        result = vt(f, k, t, (16, 16), depth_probs=p, depth_edges=edges, depth_fusion='gated')
        torch.testing.assert_close(result[:, 0].flatten(), torch.tensor([4.8, 1.2, 0.]))

    def test_no_appearance_survives_zero_support_and_amplification_is_bounded(self):
        vt, k, t = self.geometry()
        f = torch.ones(1, 6, 1, 1, 1)
        p = torch.zeros(1, 6, 64, 1, 1)
        p[:, :, 0] = 1
        result = vt(f, k, t, (16, 16), depth_probs=p,
                    depth_edges=torch.arange(65.)+.5, depth_fusion='gated')
        torch.testing.assert_close(result[:, 0].flatten(), torch.tensor([4., 0., 0.]))

    def test_full_model_occupancy_gradient_reaches_depth_and_state_roundtrip(self):
        torch.manual_seed(9)
        cfg = dict(pretrained=False, use_depth=True, depth_fusion='gated', use_xyz=True,
                   feature_channels=2, hidden_channels=4, grid_shape=(2, 2, 2),
                   voxel_bounds=(-1, -1, 1, 1, 1, 3))
        model = OccupancyModel(**cfg).eval()
        images = torch.randn(1, 6, 3, 32, 32)
        k = torch.tensor([[16., 0, 16], [0, 16, 16], [0, 0, 1]]).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        logits = model(images, k, t)
        logits.square().mean().backward()
        grad = model.depth_head.network[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual(model.depth_context_projection.in_channels, 3)
        clone = OccupancyModel(**cfg).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(clone(images, k, t), logits)

    def test_mode_requires_depth(self):
        with self.assertRaisesRegex(ValueError, 'requires use_depth'):
            OccupancyModel(pretrained=False, depth_fusion='gated')


if __name__ == '__main__':
    unittest.main()
