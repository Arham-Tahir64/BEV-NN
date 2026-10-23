"""Check which voxels contribute to semantic occupancy supervision."""
import math
import unittest

import torch

from train import compute_loss


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.zeros(1, 18, 1, 1, 4, requires_grad=True)
        self.labels = torch.tensor([0, 17, 255, 4]).reshape(1, 1, 1, 4)
        self.mask = torch.tensor([True, True, True, False]).reshape(1, 1, 1, 4)

    def test_valid_zero_and_free_classes_contribute_without_mutating_labels(self):
        original = self.labels.clone()
        loss = compute_loss(self.logits, self.labels, self.mask)
        self.assertAlmostEqual(loss.item(), math.log(18), places=6)
        loss.backward()
        gradient = self.logits.grad[0, :, 0, 0]
        self.assertAlmostEqual(gradient[0, 0].item(), -17 / 36, places=6)
        self.assertAlmostEqual(gradient[17, 1].item(), -17 / 36, places=6)
        self.assertEqual(gradient[:, 2:].abs().sum().item(), 0.)
        torch.testing.assert_close(self.labels, original)

    def test_ignored_voxel_scores_cannot_change_loss(self):
        changed = self.logits.detach().clone()
        changed[..., 2:] = torch.randn_like(changed[..., 2:]) * 100
        torch.testing.assert_close(
            compute_loss(changed, self.labels, self.mask),
            compute_loss(self.logits, self.labels, self.mask),
        )

    def test_no_supervision_returns_none(self):
        self.assertIsNone(compute_loss(self.logits, self.labels, torch.zeros_like(self.mask)))
        self.assertIsNone(compute_loss(self.logits, torch.full_like(self.labels, 255), torch.ones_like(self.mask)))

    def test_incompatible_inputs_rejected(self):
        with self.assertRaises(ValueError):
            compute_loss(self.logits, self.labels[..., :2], self.mask)
        with self.assertRaises(TypeError):
            compute_loss(self.logits, self.labels.float(), self.mask)
        with self.assertRaises(TypeError):
            compute_loss(self.logits, self.labels, self.mask.long())


if __name__ == "__main__":
    unittest.main()
