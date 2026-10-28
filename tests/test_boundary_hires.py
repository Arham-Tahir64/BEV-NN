"""Boundary masking, gradient direction, metric aggregation and image options."""
import contextlib
import io
import math
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from src.losses import observed_occupancy_boundary, boundary_occupied_free_loss
from src.metrics import occupancy_boundary_counts, summarize_boundary_counts
from train import parse_args, train_one_epoch, validate
from test_training_epoch import TinyModel, sample


class BoundaryHiresTests(unittest.TestCase):
    def test_only_both_sides_of_observed_transitions(self):
        labels = torch.tensor([0, 0, 17, 17, 255, 0, 17]).reshape(1, 1, 1, 7)
        valid = torch.ones_like(labels, dtype=torch.bool)
        valid[..., 6] = False
        expected = torch.tensor([0, 1, 1, 0, 0, 0, 0], dtype=torch.bool).reshape_as(labels)
        torch.testing.assert_close(observed_occupancy_boundary(labels, valid), expected)

    def test_no_semantic_or_exterior_boundary(self):
        labels = torch.tensor([0, 4, 15, 16]).reshape(1, 1, 2, 2)
        self.assertFalse(observed_occupancy_boundary(labels, torch.ones_like(labels, dtype=torch.bool)).any())

    def test_six_neighbours_all_axes_without_diagonals(self):
        labels = torch.zeros(1, 3, 3, 3, dtype=torch.int64)
        labels[0, 1, 1, 1] = 17
        boundary = observed_occupancy_boundary(labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertEqual(boundary.sum().item(), 7)
        self.assertFalse(boundary[0, 0, 0, 0])
        for axis in range(3):
            for side in (0, 2):
                index = [0, 1, 1, 1]
                index[axis+1] = side
                self.assertTrue(boundary[tuple(index)])

    def test_does_not_join_batch_items(self):
        labels = torch.tensor([0, 17]).reshape(2, 1, 1, 1)
        self.assertFalse(observed_occupancy_boundary(labels, torch.ones_like(labels, dtype=torch.bool)).any())

    def test_gap_loss_gradient_only_at_known_endpoints(self):
        labels = torch.tensor([0, 17, 255, 17, 0, 17]).reshape(1, 1, 1, 6)
        valid = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool).reshape_as(labels)
        logits = torch.zeros(1, 18, 1, 1, 6, requires_grad=True)
        loss, count = boundary_occupied_free_loss(logits, labels, valid)
        self.assertEqual(count, 2)
        expected = (-math.log(17/18)-math.log(1/18))/2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertGreater(logits.grad[0, 17, 0, 0, 0].item(), 0)
        self.assertLess(logits.grad[0, 17, 0, 0, 1].item(), 0)
        torch.testing.assert_close(logits.grad[..., 2:], torch.zeros_like(logits.grad[..., 2:]))

    def test_empty_boundary_has_differentiable_zero(self):
        logits = torch.randn(1, 18, 2, 2, 2, requires_grad=True)
        labels = torch.full((1, 2, 2, 2), 255)
        loss, count = boundary_occupied_free_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertEqual(count, 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_boundary_metric_detects_filled_gap(self):
        target = torch.tensor([0, 17, 0]).reshape(1, 1, 1, 3)
        valid = torch.ones_like(target, dtype=torch.bool)
        perfect = summarize_boundary_counts(occupancy_boundary_counts(target, target, valid))
        self.assertEqual(perfect['boundary_f1'], 1)
        filled = summarize_boundary_counts(occupancy_boundary_counts(torch.zeros_like(target), target, valid))
        self.assertEqual(filled['boundary_f1'], 0)
        self.assertEqual(filled['boundary_recall'], 0)
        self.assertIsNone(filled['boundary_precision'])

    def test_unknown_neighbours_cannot_create_predicted_boundaries(self):
        labels = torch.tensor([0, 255, 17]).reshape(1, 1, 1, 3)
        predicted = torch.tensor([0, 17, 0]).reshape_as(labels)
        counts = occupancy_boundary_counts(predicted, labels, torch.ones_like(labels, dtype=torch.bool))
        self.assertEqual(counts.sum().item(), 0)
        self.assertIsNone(summarize_boundary_counts(counts)['boundary_f1'])

    def test_metric_aggregates_counts_before_rates(self):
        result = summarize_boundary_counts(torch.tensor([1, 2, 1])+torch.tensor([1, 2, 3]))
        self.assertEqual(result['boundary_precision'], .5)
        self.assertEqual(result['boundary_recall'], .5)
        self.assertEqual(result['boundary_f1'], .5)

    def test_training_and_validation_integration(self):
        outcomes = []
        for weight in (0., .1):
            model = TinyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            with contextlib.redirect_stdout(io.StringIO()):
                stats = train_one_epoch(model, DataLoader([sample()]), optimizer, scaler,
                    torch.device('cpu'), False, geometry_loss_weight=.2, boundary_loss_weight=weight)
                metrics = validate(model, DataLoader([sample()]), torch.device('cpu'), False)
            outcomes.append((model.scores.detach().clone(), stats))
            self.assertIn('boundary_f1', metrics)
            self.assertEqual(metrics['boundary_gt_voxels'], 2)
        self.assertNotIn('boundary_loss', outcomes[0][1])
        self.assertEqual(outcomes[1][1]['boundary_voxels'], 2)
        self.assertGreater(outcomes[1][0][17], outcomes[0][0][17])

    def test_cli_high_resolution_and_loss_defaults(self):
        with patch('sys.argv', ['train.py']):
            args = parse_args()
            self.assertEqual(tuple(args.image_size), (320, 576))
            self.assertEqual(args.boundary_loss_weight, 0.)
        with patch('sys.argv', ['train.py', '--image-size', '480', '864',
                               '--geometry-loss-weight', '.2', '--boundary-loss-weight', '.1']):
            args = parse_args()
            self.assertEqual(args.image_size, [480, 864])
            self.assertEqual(args.boundary_loss_weight, .1)
        for options in (['--image-size', '0', '864'], ['--image-size', '480', '-1'],
                        ['--boundary-loss-weight', '-1'], ['--boundary-loss-weight', 'nan'],
                        ['--boundary-loss-weight', 'inf']):
            with patch('sys.argv', ['train.py', *options]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_negative_runtime_boundary_weight_rejected(self):
        model = TinyModel()
        with self.assertRaises(ValueError):
            train_one_epoch(model, [], torch.optim.SGD(model.parameters(), lr=.1),
                torch.amp.GradScaler('cuda', enabled=False), torch.device('cpu'), False,
                boundary_loss_weight=-.1)


if __name__ == '__main__':
    unittest.main()
