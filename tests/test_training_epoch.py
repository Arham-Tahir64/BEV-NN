"""Check epoch updates, skipped batches, and loss aggregation."""
import contextlib
import io
import math
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import train_one_epoch


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(18))
        self.calls = 0

    def forward(self, images, intrinsics, ego_to_camera):
        self.calls += 1
        offset = torch.zeros_like(self.scores)
        offset[0] = 1.
        scores = self.scores[None, :] + images[:, 0, 0, 0, 0, None] * offset
        return scores[:, :, None, None, None].expand(-1, -1, 1, 1, 2)


def sample(boost=0., mask=(True, True), labels=(0, 17)):
    return {
        "images": torch.full((6, 3, 1, 1), boost),
        "intrinsics": torch.eye(3).repeat(6, 1, 1),
        "ego_to_camera": torch.eye(4).repeat(6, 1, 1),
        "semantics": torch.tensor(labels).reshape(1, 1, 2),
        "valid_mask": torch.tensor(mask).reshape(1, 1, 2),
    }


class TrainingEpochTests(unittest.TestCase):
    def run_epoch(self, model, samples, lr=0.):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_one_epoch(
                model, DataLoader(samples, batch_size=1), optimizer,
                scaler, torch.device("cpu"), amp_enabled=False,
            )
        return stats, optimizer

    def test_loss_is_weighted_by_supervised_voxels(self):
        model = TinyModel()
        stats, _ = self.run_epoch(model, [
            sample(mask=(True, False)),
            sample(boost=2.),
        ])
        expected = (math.log(18) + 2 * (math.log(math.exp(2) + 17) - 1)) / 3
        self.assertAlmostEqual(stats["loss"], expected, places=6)
        self.assertEqual(stats["valid_voxels"], 3)

    def test_valid_batch_updates_weights_but_ignore_batches_do_not_step(self):
        model = TinyModel()
        before = model.scores.detach().clone()
        stats, optimizer = self.run_epoch(model, [
            sample(mask=(False, False)),
            sample(labels=(0, 0)),
            sample(labels=(255, 255)),
        ], lr=0.01)
        self.assertEqual(stats["batches"], 1)
        self.assertEqual(stats["skipped_batches"], 2)
        self.assertEqual(model.calls, 1)
        self.assertEqual(optimizer.state[model.scores]["step"].item(), 1)
        self.assertFalse(torch.equal(before, model.scores))
        self.assertTrue(torch.isfinite(model.scores).all())

    def test_no_supervision_cannot_report_zero_loss(self):
        for samples in ([], [sample(mask=(False, False))]):
            with self.subTest(empty=not samples):
                with self.assertRaisesRegex(RuntimeError, "no supervised voxels"):
                    self.run_epoch(TinyModel(), samples)

    def test_nonfinite_loss_stops_before_update(self):
        model = TinyModel()
        with self.assertRaises(FloatingPointError):
            self.run_epoch(model, [sample(boost=float("nan"))], lr=0.01)
        torch.testing.assert_close(model.scores, torch.zeros_like(model.scores))


if __name__ == "__main__":
    unittest.main()
