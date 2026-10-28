"""Semantic supervision, visibility, gradient and inference contracts."""
import contextlib
import io
import math
import sys
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from src.semantic_supervision import semantic_dice_loss, make_bev_targets, bev_auxiliary_loss
from src.models.occupancy_model import OccupancyModel
from train import parse_args, train_one_epoch
from test_training_epoch import TinyModel, sample


class SemanticSupervisionTests(unittest.TestCase):
    def test_uniform_dice_and_masked_gradients(self):
        logits = torch.zeros(1, 18, 1, 1, 3, requires_grad=True)
        labels = torch.tensor([0, 17, 255]).reshape(1, 1, 1, 3)
        loss, count = semantic_dice_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(loss.item(), 0.9, places=5)
        loss.backward()
        self.assertEqual(logits.grad[..., 2].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 0, 0, 0, 1].item(), 0)  # false-positive penalty on free
        self.assertLess(logits.grad[0, 0, 0, 0, 0].item(), 0)

    def test_dice_macro_average_and_visibility(self):
        labels = torch.tensor([0, 0, 1, 17]).reshape(1, 1, 1, 4)
        logits = torch.randn(1, 18, 1, 1, 4, requires_grad=True)
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[..., -1] = False
        loss, count = semantic_dice_loss(logits, labels, mask)
        p = logits[0, :, 0, 0, :3].softmax(0)
        expected = 1 - torch.stack([(2*p[0, :2].sum()+1e-6)/(p[0].sum()+2+1e-6),
                                    (2*p[1, 2]+1e-6)/(p[1].sum()+1+1e-6)]).mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(count, 2)
        loss.backward()
        self.assertEqual(logits.grad[..., -1].abs().sum().item(), 0)

    def test_empty_and_free_dice_are_connected_zero(self):
        for label in (17, 255):
            logits = torch.zeros(1, 18, 1, 1, 2, requires_grad=True)
            labels = torch.full((1, 1, 1, 2), label)
            loss, count = semantic_dice_loss(logits, labels, labels != 255)
            self.assertEqual((loss.item(), count), (0, 0))
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_bev_topmost_and_conservative_unknown_policy(self):
        labels = torch.tensor([[0, 3, 17], [0, 17, 255], [255, 0, 17],
                               [17, 17, 17], [17, 255, 17], [255, 255, 255]]).reshape(1, 2, 3, 3)
        original = labels.clone()
        targets, valid = make_bev_targets(labels, labels != 255)
        torch.testing.assert_close(targets.flatten(), torch.tensor([3, 255, 0, 17, 255, 255]))
        self.assertEqual(valid.sum().item(), 3)
        torch.testing.assert_close(labels, original)
        logits = torch.zeros(1, 18, 2, 3, requires_grad=True)
        loss, count = bev_auxiliary_loss(logits, labels, labels != 255)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(loss.item(), math.log(18), places=6)
        loss.backward()
        self.assertEqual(logits.grad.movedim(1, -1)[~valid].abs().sum().item(), 0)

    def test_bev_invisible_occupied_voxel_cannot_be_a_target(self):
        labels = torch.tensor([0, 17]).reshape(1, 1, 1, 2)
        mask = torch.tensor([False, True]).reshape_as(labels)
        targets, valid = make_bev_targets(labels, mask)
        self.assertEqual(targets.item(), 255)
        self.assertFalse(valid.item())
        logits = torch.zeros(1, 18, 1, 1, requires_grad=True)
        loss, count = bev_auxiliary_loss(logits, labels, mask)
        loss.backward()
        self.assertEqual((loss.item(), count, logits.grad.abs().sum().item()), (0, 0, 0))

    def test_cli_flags_and_rejections(self):
        with patch.object(sys, 'argv', ['train']):
            args = parse_args()
        self.assertEqual((args.semantic_dice_weight, args.bev_aux_weight), (0, 0))
        for flags in (['--semantic-dice-weight', 'nan'], ['--bev-aux-weight', '-1'],
                      ['--bev-aux-weight', '.1']):
            with patch.object(sys, 'argv', ['train', *flags]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_training_updates_and_logs_both_losses(self):
        class AuxiliaryTiny(TinyModel):
            def __init__(self):
                super().__init__()
                self.bev_aux_head = torch.nn.Conv2d(18, 18, 1)
            def forward(self, images, intrinsics, ego_to_camera, return_aux=False):
                logits = super().forward(images, intrinsics, ego_to_camera)
                return {'occupancy_logits': logits, 'bev_logits': self.bev_aux_head(logits.mean(-1))} if return_aux else logits
        model = AuxiliaryTiny()
        before = model.bev_aux_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_one_epoch(model, DataLoader([sample()], batch_size=1), optimizer,
                                    torch.amp.GradScaler('cuda', enabled=False), torch.device('cpu'), False,
                                    semantic_dice_weight=.2, bev_aux_weight=.1)
        self.assertAlmostEqual(stats['loss'], math.log(18), places=6)
        self.assertEqual(stats['bev_aux_cells'], 1)
        self.assertEqual(stats['bev_aux_coverage'], 1)
        self.assertEqual(stats['semantic_dice_classes'], 1)
        self.assertFalse(torch.equal(before, model.bev_aux_head.weight))
        self.assertGreater(model.scores.grad.abs().sum().item(), 0)

    def test_head_preserves_seeded_core_and_skips_eval(self):
        config = dict(architecture='bevdepth', use_depth=True, depth_fusion='bevdepth', pretrained=False,
                      grid_shape=(8, 8, 4), depth_bins=4, depth_max=4.1)
        torch.manual_seed(7)
        original = OccupancyModel(**config)
        rng = torch.random.get_rng_state()
        torch.manual_seed(7)
        model = OccupancyModel(**config, use_bev_aux=True)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        core_state = {k: v for k, v in model.state_dict().items() if not k.startswith('bev_aux_head.')}
        original.load_state_dict(core_state, strict=True)
        for key, value in core_state.items():
            torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        clone = OccupancyModel(**config, use_bev_aux=True)
        clone.load_state_dict(model.state_dict(), strict=True)
        images = torch.randn(1, 6, 3, 32, 32)
        k = torch.eye(3).repeat(1, 6, 1, 1)
        t = torch.eye(4).repeat(1, 6, 1, 1)
        model.eval(); original.eval()
        with torch.no_grad(), patch.object(model.bev_aux_head, 'forward', side_effect=AssertionError('eval ran head')):
            result = model(images, k, t, return_aux=True)
            self.assertNotIn('bev_logits', result)
            torch.testing.assert_close(result['occupancy_logits'], original(images, k, t), rtol=0, atol=0)
        model.train()
        result = model(images, k, t, return_aux=True)
        self.assertEqual(result['bev_logits'].shape, (1, 18, 8, 8))
        result['bev_logits'].square().mean().backward()
        self.assertGreater(model.bev_aux_head[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.occupancy_decoder.refine.body[0].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
