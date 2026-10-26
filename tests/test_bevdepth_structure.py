"""Contracts for camera conditioning, measured depth, refinement and splatting."""
import contextlib
import io
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from data.depth_targets import depth_loss, DepthTargetDataset
from data.lidar_depth import project_lidar_depth, lidar_record
from src.models.bevdepth import (BEVDepthImageEncoder, CameraAwareDepthNet,
                                FrustumRefinement, FrustumVoxelPool, MultiScaleOccupancyDecoder)
from src.models.occupancy_model import OccupancyModel
from train import parse_args, save_checkpoint


class BEVDepthStructureTests(unittest.TestCase):
    def cameras(self):
        k = torch.eye(3).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        t[:, 1:, 0, 3] = 100
        return k, t

    def test_forward_pooling_preserves_depth_ratio_and_height(self):
        pool = FrustumVoxelPool((1, 1, 3), (-.5, -.5, .5, .5, .5, 3.5), 1, refine=False)
        k, t = self.cameras()
        context = torch.full((1, 6, 1, 1, 1), 10., requires_grad=True)
        p = torch.tensor([.8, .2]).reshape(1, 1, 2, 1, 1).repeat(1, 6, 1, 1, 1).requires_grad_()
        result = pool(context, p, torch.tensor([1., 2.]), k, t)
        torch.testing.assert_close(result.flatten(), torch.tensor([8., 2., 0.]))
        result.sum().backward()
        self.assertGreater(p.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(p.grad[:, 1:].abs().sum().item(), 0)
        self.assertGreater(context.grad[:, 0].abs().sum().item(), 0)

    def test_pooling_sums_valid_cameras_and_does_not_normalize_by_support(self):
        pool = FrustumVoxelPool((1, 1, 3), (-.5, -.5, .5, .5, .5, 3.5), 1, refine=False)
        k, t = self.cameras()
        t[:, 1] = torch.eye(4)
        context = torch.full((1, 6, 1, 1, 1), 10.)
        context[:, 1] = 5
        p = torch.tensor([.8, .2]).reshape(1, 1, 2, 1, 1).repeat(1, 6, 1, 1, 1)
        p[:, 1, :, 0, 0] = torch.tensor([.6, .4])
        result = pool(context, p, torch.tensor([1., 2.]), k, t)
        torch.testing.assert_close(result.flatten(), torch.tensor([11., 4., 0.]))

    def test_unprojection_uses_camera_to_ego_and_xyz_voxel_order(self):
        pool = FrustumVoxelPool((2, 1, 2), (-.5, -.5, .5, 1.5, .5, 2.5), 1, refine=False)
        k, t = self.cameras()
        t[:, 0, 0, 3] = -1  # Camera point x=0 becomes ego x=+1.
        result = pool(torch.ones(1, 6, 1, 1, 1), torch.ones(1, 6, 1, 1, 1), torch.tensor([2.]), k, t)
        expected = torch.zeros(1, 1, 2, 1, 2)
        expected[0, 0, 1, 0, 1] = 1
        torch.testing.assert_close(result, expected)

    def test_outside_low_and_high_bounds_never_splat_inside(self):
        pool = FrustumVoxelPool((1, 1, 1), (-.5, -.5, .5, .5, .5, 1.5), 1, refine=False)
        k, t = self.cameras()
        for translation in (.6, -.5):
            t[:, 0, 0, 3] = translation
            result = pool(torch.ones(1, 6, 1, 1, 1), torch.ones(1, 6, 1, 1, 1), torch.tensor([1.]), k, t)
            torch.testing.assert_close(result, torch.zeros_like(result))

    def test_float32_interior_upper_boundary_stays_in_last_voxel(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for axis in (0, 1):
                with self.subTest(device=device, axis=axis):
                    pool = FrustumVoxelPool((200, 200, 16),
                        (-40., -40., -1., 40., 40., 5.4), 1, refine=False).to(device)
                    k = torch.eye(3, device=device).repeat(1, 6, 1, 1)
                    t = torch.eye(4, device=device).repeat(1, 6, 1, 1)
                    interior = torch.nextafter(torch.tensor(40., device=device),
                                               torch.tensor(0., device=device))
                    t[:, :, axis, 3] = -interior
                    context = torch.ones(1, 6, 1, 1, 1, device=device, requires_grad=True)
                    probabilities = torch.ones(1, 6, 1, 1, 1, device=device)
                    result = pool(context, probabilities, torch.tensor([1.], device=device), k, t)
                    expected = torch.zeros_like(result)
                    index = [100, 100, 5]
                    index[axis] = 199
                    expected[(0, 0, *index)] = 6
                    torch.testing.assert_close(result, expected)
                    result.sum().backward()
                    torch.testing.assert_close(context.grad, torch.ones_like(context))

    def test_camera_parameters_change_depth_and_context(self):
        torch.manual_seed(12)
        head = CameraAwareDepthNet(context_channels=8, num_bins=8, in_channels=16, hidden_channels=32).eval()
        x = torch.randn(1, 6, 16, 2, 3)
        k, t = self.cameras()
        t[:] = torch.eye(4)
        k[:, :, 0, 0] = 20
        k[:, :, 1, 1] = 20
        first = head(x, k, t, (32, 48))
        k[:, :, 0, 0] = 40
        second = head(x, k, t, (32, 48))
        self.assertFalse(torch.allclose(first[0], second[0]))
        self.assertFalse(torch.allclose(first[1], second[1]))
        (second[0].square().mean()+second[1].square().mean()).backward()
        self.assertGreater(head.depth_gate.mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.offsets.weight.grad.abs().sum().item(), 0)

    def test_image_backbone_is_trainable_but_stem_and_bn_statistics_are_frozen(self):
        model = BEVDepthImageEncoder(pretrained=False).train()
        before = model.stages[0][0].bn1.running_mean.clone()
        result = model(torch.randn(1, 6, 3, 32, 48))
        self.assertEqual(result.shape, (1, 6, 128, 2, 3))
        result.square().mean().backward()
        self.assertGreater(model.stages[-1][0].conv1.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.stem.parameters()))
        torch.testing.assert_close(before, model.stages[0][0].bn1.running_mean)

    def test_refinement_operates_on_depth_width_without_mixing_image_rows(self):
        torch.manual_seed(11)
        refiner = FrustumRefinement(8, row_chunk=1).eval()
        volume = torch.randn(1, 1, 8, 5, 2, 4, requires_grad=True)
        result = refiner(volume)
        changed = volume.detach().clone()
        changed[..., 0, :] += 2
        different = refiner(changed)
        torch.testing.assert_close(result[..., 1, :], different[..., 1, :])
        self.assertFalse(torch.allclose(result[..., 0, :], different[..., 0, :]))
        result.square().mean().backward()
        self.assertGreater(volume.grad.abs().sum().item(), 0)

    def test_refinement_checkpointed_chunks_match_direct_results_and_gradients(self):
        torch.manual_seed(15)
        refiner = FrustumRefinement(8, row_chunk=1).train()
        other = FrustumRefinement(8, row_chunk=8).eval()
        other.load_state_dict(refiner.state_dict())
        one = torch.randn(1, 2, 8, 4, 2, 3, requires_grad=True)
        two = one.detach().clone().requires_grad_()
        first, second = refiner(one), other(two)
        torch.testing.assert_close(first, second)
        first.square().mean().backward()
        second.square().mean().backward()
        torch.testing.assert_close(one.grad, two.grad)

    def test_multiscale_decoder_preserves_odd_xyz_dimensions(self):
        model = MultiScaleOccupancyDecoder(channels=8)
        x = torch.randn(1, 8, 5, 7, 3, requires_grad=True)
        result = model(x)
        self.assertEqual(result.shape, (1, 18, 5, 7, 3))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bce_depth_matches_manual_onehot_and_masks_unknown(self):
        logits = torch.tensor([.2, 1., -.5]).reshape(1, 1, 3, 1, 1).repeat(1, 1, 1, 1, 2).requires_grad_()
        target = torch.tensor([1.2, float('nan')]).reshape(1, 1, 1, 2)
        actual, count = depth_loss(logits, target, torch.arange(4.), mode='bce')
        expected = F.binary_cross_entropy(logits[0, 0, :, 0, 0].softmax(0), torch.tensor([0., 1., 0.]), reduction='sum')
        torch.testing.assert_close(actual, expected)
        self.assertEqual(count, 1)
        actual.backward()
        self.assertEqual(logits.grad[..., 1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_lidar_projection_selects_nearest_point_and_handles_camera_motion(self):
        points = np.array([[0., 0, 2], [0., 0, 5], [0, 0, -1], [np.nan, 0, 1]])
        sensor_to_ego = np.eye(4)
        sensor_to_ego[2, 3] = 1
        ego_to_camera = np.eye(4)[None]
        ego_to_camera[0, 2, 3] = -.5
        depth = project_lidar_depth(points, sensor_to_ego, np.eye(3)[None], ego_to_camera, (16, 16))
        self.assertAlmostEqual(depth[0, 0, 0], 2.5)
        unknown = project_lidar_depth(np.empty((0, 3)), np.eye(4), np.eye(3)[None], np.eye(4)[None], (16, 16))
        self.assertTrue(np.isnan(unknown).all())

    def test_lidar_nearer_out_of_range_return_blocks_farther_target(self):
        result = project_lidar_depth(np.array([[0., 0, .2], [0, 0, 2.]]), np.eye(4),
                                    np.eye(3)[None], np.eye(4)[None], (16, 16), min_depth=1.)
        self.assertTrue(np.isnan(result).all())

    def test_measured_mode_fails_without_lidar_instead_of_falling_back(self):
        class Missing:
            records = [{'sample_token': 'example'}]
        with tempfile.TemporaryDirectory() as cache, self.assertRaisesRegex(FileNotFoundError, 'LIDAR_TOP'):
            DepthTargetDataset(Missing(), cache, target_mode='lidar')
        self.assertEqual(lidar_record(None, {'data': {}}), {})

    def test_bevdepth_config_roundtrip_and_occupancy_gradient(self):
        torch.manual_seed(2)
        cfg = dict(architecture='bevdepth', depth_fusion='bevdepth', use_depth=True, pretrained=False,
                   feature_channels=8, hidden_channels=8, depth_bins=8, depth_max=4.1,
                   grid_shape=(4, 4, 4), voxel_bounds=(-2, -2, 0, 2, 2, 4))
        model = OccupancyModel(**cfg).eval()
        images = torch.randn(1, 6, 3, 32, 32)
        k = torch.tensor([[16., 0, 16.], [0, 16, 16], [0, 0, 1]]).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        result = model(images, k, t, return_aux=True)
        result['occupancy_logits'].square().mean().backward()
        self.assertGreater(model.depth_head.classifier.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.image_encoder.stages[-1][0].conv1.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.view_transformer.refinement.output.weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_checkpoint(path, model, torch.optim.AdamW(model.parameters()),
                            torch.amp.GradScaler('cuda', enabled=False), 1, {'model': cfg})
            saved = torch.load(path, weights_only=True)
            restored = OccupancyModel(**saved['config']['model']).eval()
            restored.load_state_dict(saved['model_state_dict'], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(images, k, t), result['occupancy_logits'])

    def test_cli_paper_structure_defaults_and_legacy_defaults(self):
        with patch.object(sys, 'argv', ['train.py', '--architecture', 'bevdepth', '--depth']):
            args = parse_args()
            self.assertEqual((args.depth_fusion, args.depth_loss_mode, args.depth_loss_weight), ('bevdepth', 'bce', 3.))
        with patch.object(sys, 'argv', ['train.py', '--depth']):
            args = parse_args()
            self.assertEqual((args.depth_fusion, args.depth_loss_mode, args.depth_loss_weight), ('probability', 'linear', 1.))
        for flags in (['--architecture', 'bevdepth'], ['--depth', '--depth-fusion', 'bevdepth'],
                      ['--architecture', 'bevdepth', '--depth', '--depth-fusion', 'probability']):
            with patch.object(sys, 'argv', ['train.py', *flags]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
