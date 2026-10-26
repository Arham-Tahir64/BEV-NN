"""Depth contrast, paired ray sampling, gradients, and checkpoint contracts."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.models.occupancy_model import OccupancyModel
from src.models.view_transformer import ViewTransformer
from train import parse_args, save_checkpoint


class ProbabilityLiftingTests(unittest.TestCase):
    def geometry(self):
        transformer = ViewTransformer(grid_shape=(1, 1, 3),
            voxel_bounds=(-.1, -.1, .5, .1, .1, 3.5), voxel_chunk_size=1)
        intrinsics = torch.eye(3).repeat(1, 6, 1, 1)
        transforms = torch.eye(4).repeat(1, 6, 1, 1)
        transforms[:, 1:, 2, 3] = -100
        return transformer, intrinsics, transforms

    def lift(self, transformer, features, intrinsics, transforms, probabilities, edges,
             image_size=(16, 16)):
        return transformer(features, intrinsics, transforms, image_size,
                           depth_probs=probabilities, depth_edges=edges,
                           depth_fusion='probability')

    def test_probability_ratio_survives_above_the_old_ceiling(self):
        transformer = ViewTransformer(grid_shape=(1, 1, 2),
            voxel_bounds=(-.1, -.1, 15.6, .1, .1, 35.6))
        _, intrinsics, transforms = self.geometry()
        features = torch.ones(1, 6, 1, 1, 1)
        probabilities = torch.zeros(1, 6, 64, 1, 1)
        probabilities[:, :, 20] = .9
        probabilities[:, :, 30] = .06
        probabilities[:, :, 40] = .04
        probabilities.requires_grad_()
        result = self.lift(transformer, features, intrinsics, transforms, probabilities,
                           torch.linspace(.1, 64.1, 65))
        torch.testing.assert_close(result[:, 0].flatten(), torch.tensor([57.6, 3.84]),
                                   rtol=1e-5, atol=1e-4)
        ratio = (result[0, 0, 0, 0, 0]/result[0, 0, 0, 0, 1]).detach().item()
        self.assertAlmostEqual(ratio, 15., places=4)
        result[:, 0].sum().backward()
        self.assertGreater(probabilities.grad[0, 0, 20].item(), 0)
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_neighbouring_background_depth_cannot_borrow_foreground_appearance(self):
        # Near foreground pixel at u=0, far background pixel at u=16.
        # Query the far bin at u=8, halfway between them. Neither far pixel
        # has foreground appearance, so the jointly sampled feature is zero.
        transformer = ViewTransformer(grid_shape=(1, 1, 1),
            voxel_bounds=(15.5, -.5, 1.5, 16.5, .5, 2.5))
        _, intrinsics, transforms = self.geometry()
        features = torch.tensor([10., 0.]).reshape(1, 1, 1, 1, 2).repeat(1, 6, 1, 1, 1)
        probabilities = torch.tensor([[1., 0.], [0., 1.]]).reshape(1, 1, 2, 1, 2).repeat(1, 6, 1, 1, 1)
        edges = torch.tensor([.5, 1.5, 2.5])
        result = self.lift(transformer, features, intrinsics, transforms, probabilities, edges, (1, 17))
        torch.testing.assert_close(result.flatten(), torch.tensor([0., 1.]))
        old = transformer(features, intrinsics, transforms, (1, 17), depth_probs=probabilities,
                          depth_edges=edges, depth_fusion='gated')
        self.assertAlmostEqual(old[0, 0, 0, 0, 0].item(), 5.)

    def test_gate_is_not_cancelled_for_a_single_camera_and_unknown_range_is_zero(self):
        transformer, intrinsics, transforms = self.geometry()
        features = torch.full((1, 6, 1, 1, 1), 3.)
        probabilities = torch.tensor([.8, .2]).reshape(1, 1, 2, 1, 1).repeat(1, 6, 1, 1, 1)
        result = self.lift(transformer, features, intrinsics, transforms, probabilities,
                           torch.tensor([.5, 1.5, 2.5]))
        torch.testing.assert_close(result[:, 0].flatten(), torch.tensor([4.8, 1.2, 0.]))
        torch.testing.assert_close(result[:, 1].flatten(), torch.tensor([1.6, .4, 0.]))
        probabilities.zero_()
        result = self.lift(transformer, features, intrinsics, transforms, probabilities,
                           torch.tensor([.5, 1.5, 2.5]))
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_depth_end_bin_edges_are_not_attenuated_or_extended_outside_range(self):
        transformer, _, _ = self.geometry()
        volume = torch.tensor([2., 6., 1., 3.]).reshape(1, 2, 2, 1, 1)
        pixels = torch.zeros(1, 1, 6, 2)
        depths = torch.tensor([.49, .5, .75, 2.25, 2.499, 2.5]).reshape(1, 1, 6)
        result = transformer.sample_probability_volume(volume, pixels, depths,
                                                        torch.tensor([.5, 1.5, 2.5]))
        torch.testing.assert_close(result[0, 0, 0], torch.tensor([0., 2., 2., 6., 6., 0.]))

    def test_uniform_depth_matches_ungated_features_including_spatial_padding(self):
        torch.manual_seed(10)
        transformer = ViewTransformer(grid_shape=(2, 2, 2),
            voxel_bounds=(-1, -1, 2, 1, 1, 4), voxel_chunk_size=3)
        intrinsics = torch.tensor([[8., 0, 27.], [0, 8, 27.], [0, 0, 1]]).repeat(1, 6, 1, 1)
        transforms = torch.eye(4).repeat(1, 6, 1, 1)
        features = torch.randn(1, 6, 2, 2, 2)
        probabilities = torch.full((1, 6, 8, 2, 2), 1/8)
        result = self.lift(transformer, features, intrinsics, transforms, probabilities,
                           torch.arange(9.), (32, 32))
        baseline = transformer(features, intrinsics, transforms, (32, 32))
        torch.testing.assert_close(result[:, :2], baseline)

    def test_invalid_views_and_unseen_voxels_contribute_zero(self):
        transformer, intrinsics, transforms = self.geometry()
        features = torch.ones(1, 6, 1, 1, 1, requires_grad=True)
        probabilities = torch.full((1, 6, 2, 1, 1), .5, requires_grad=True)
        result = self.lift(transformer, features, intrinsics, transforms, probabilities,
                           torch.tensor([.5, 1.5, 2.5]))
        result.sum().backward()
        self.assertEqual(features.grad[:, 1:].abs().sum().item(), 0)
        self.assertEqual(probabilities.grad[:, 1:].abs().sum().item(), 0)
        transforms[:, :, 2, 3] = -100
        result = self.lift(transformer, features, intrinsics, transforms, probabilities,
                           torch.tensor([.5, 1.5, 2.5]))
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_chunked_outputs_and_gradients_match_whole_grid(self):
        torch.manual_seed(4)
        cfg = dict(grid_shape=(2, 2, 2), voxel_bounds=(-1, -1, 1, 1, 1, 3))
        chunked = ViewTransformer(**cfg, voxel_chunk_size=3)
        whole = ViewTransformer(**cfg, voxel_chunk_size=8)
        features = torch.randn(1, 6, 2, 2, 2)
        logits = torch.randn(1, 6, 4, 2, 2)
        intrinsics = torch.tensor([[8., 0, 8.], [0, 8., 8.], [0, 0, 1.]]).repeat(1, 6, 1, 1)
        transforms = torch.eye(4).repeat(1, 6, 1, 1)
        outputs, gradients = [], []
        for transformer in (chunked, whole):
            f = features.clone().requires_grad_()
            l = logits.clone().requires_grad_()
            output = self.lift(transformer, f, intrinsics, transforms, l.softmax(2),
                               torch.arange(5.), (32, 32))
            output.square().sum().backward()
            outputs.append(output)
            gradients.append((f.grad, l.grad))
        torch.testing.assert_close(*outputs)
        for chunk_grad, whole_grad in zip(*gradients):
            torch.testing.assert_close(chunk_grad, whole_grad)
            self.assertTrue(torch.isfinite(chunk_grad).all())
            self.assertGreater(chunk_grad.abs().sum().item(), 0)

    def test_full_model_has_no_raw_appearance_branch_and_occupancy_trains_depth(self):
        torch.manual_seed(9)
        cfg = dict(pretrained=False, use_depth=True, depth_fusion='probability', use_xyz=True,
                   feature_channels=2, hidden_channels=4, grid_shape=(2, 2, 2),
                   voxel_bounds=(-1, -1, 1, 1, 1, 3))
        model = OccupancyModel(**cfg).eval()
        images = torch.randn(1, 6, 3, 32, 32)
        intrinsics = torch.tensor([[16., 0, 16], [0, 16, 16], [0, 0, 1]]).repeat(1, 6, 1, 1)
        transforms = torch.eye(4).repeat(1, 6, 1, 1)
        output = model(images, intrinsics, transforms)
        output.square().mean().backward()
        grad = model.depth_head.network[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual(model.depth_context_projection.in_channels, 3)
        self.assertTrue(all(p.grad is None for p in model.image_encoder.backbone.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latest.pt'
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(path, model, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                            1, {'model': cfg})
            saved = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(saved['config']['model']['depth_fusion'], 'probability')
            clone = OccupancyModel(**saved['config']['model']).eval()
            clone.load_state_dict(saved['model_state_dict'], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(clone(images, intrinsics, transforms), output)

    def test_mode_requires_depth_probabilities_and_a_depth_head(self):
        transformer, intrinsics, transforms = self.geometry()
        with self.assertRaisesRegex(ValueError, 'requires use_depth'):
            OccupancyModel(pretrained=False, depth_fusion='probability')
        with self.assertRaisesRegex(ValueError, 'requires depth probabilities'):
            transformer(torch.ones(1, 6, 1, 1, 1), intrinsics, transforms, (16, 16),
                        depth_fusion='probability')

    def test_cli_selects_new_path_only_for_new_depth_runs(self):
        for arguments, expected in (([], 'legacy'), (['--depth'], 'probability'),
                                    (['--depth', '--depth-fusion', 'residual'], 'residual'),
                                    (['--depth', '--depth-fusion', 'gated'], 'gated')):
            with self.subTest(arguments=arguments), patch.object(sys, 'argv', ['train.py', *arguments]):
                self.assertEqual(parse_args().depth_fusion, expected)
        with patch.object(sys, 'argv', ['train.py', '--depth-fusion', 'probability']), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
