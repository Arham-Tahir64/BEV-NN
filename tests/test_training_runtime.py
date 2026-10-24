"""Checks for held-out evaluation, checkpoint recovery and script setup."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from train import main, save_checkpoint, train_one_epoch, validate
from test_training_epoch import TinyModel, sample


class SmallDataset(Dataset):
    num_classes = 18
    grid_shape = (1, 1, 2)
    voxel_bounds = (-1., -1., 0., 1., 1., 2.)
    image_size = (1, 1)
    camera_order = tuple(f"camera-{i}" for i in range(6))
    ignore_index = 255
    occupancy_root = Path("unused")

    def __init__(self, split):
        self.scene_names = (f"{split}-scene",)
        self.records = [{"sample_token": f"{split}-{i}"} for i in range(2 if split == "train" else 1)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return sample(labels=(0, 0))


class TrainingRuntimeTests(unittest.TestCase):
    def test_validation_masks_weights_and_does_not_change_model(self):
        model = TinyModel().train()
        before = model.scores.detach().clone()
        modes = []
        model.register_forward_pre_hook(
            lambda module, args: modes.append((module.training, torch.is_grad_enabled()))
        )
        loader = DataLoader([
            sample(mask=(True, False)), sample(boost=2.), sample(labels=(255, 255)),
        ], batch_size=1)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = validate(model, loader, torch.device("cpu"), False)
        expected = (math.log(18) + 2 * (math.log(math.exp(2) + 17) - 1)) / 3
        self.assertAlmostEqual(stats["loss"], expected, places=6)
        self.assertAlmostEqual(stats["accuracy"], 2 / 3)
        self.assertEqual(stats["skipped_batches"], 1)
        self.assertEqual(modes, [(False, False), (False, False)])
        torch.testing.assert_close(before, model.scores)
        self.assertIsNone(model.scores.grad)

    def test_validation_without_supervision_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "no supervised voxels"):
            validate(TinyModel(), DataLoader([sample(mask=(False, False))]), torch.device("cpu"), False)

    def test_checkpoint_round_trip_includes_optimizer_and_config(self):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(model, DataLoader([sample()]), optimizer, scaler, torch.device("cpu"), False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "latest.pt"
            save_checkpoint(path, model, optimizer, scaler, 3, {"model": {"num_classes": 18}})
            state = torch.load(path, map_location="cpu", weights_only=True)
            restored = TinyModel()
            restored.load_state_dict(state["model_state_dict"])
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
            restored_optimizer.load_state_dict(state["optimizer_state_dict"])
            torch.testing.assert_close(restored.scores, model.scores)
            self.assertEqual(restored_optimizer.state[restored.scores]["step"].item(), 1)
            self.assertEqual(state["epoch"], 3)
            self.assertEqual(state["config"]["model"]["num_classes"], 18)
            self.assertIn("scaler_state_dict", state)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_main_full_split_automatically_validates_and_saves_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = argparse.Namespace(
                epochs=2, lr=0.01, weight_decay=0., batch_size=1, workers=0,
                seed=0, device="cpu", amp=True, output_dir=output,
                overfit_samples=0, validate=None, class_balance="none", xyz=False,
                depth=False, depth_bins=64, depth_min=0.1, depth_max=64.1,
                depth_loss_weight=1.0, depth_cache=output / "depth_cache",
            )
            with patch("train.parse_args", return_value=args), \
                 patch("train.NuScenesOccupancyDataset", side_effect=SmallDataset), \
                 patch("train.OccupancyModel", side_effect=lambda **kwargs: TinyModel()), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["selection_metric"], "validation_loss")
            self.assertFalse(config["amp_enabled"])
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["validation"]["valid_voxels"] == 2 for row in rows))
            latest = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(latest["epoch"], 2)
            self.assertEqual(latest["config"]["metrics"]["selection_metric"], "validation_loss")
            self.assertTrue((output / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
