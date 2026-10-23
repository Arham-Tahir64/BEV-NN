"""Check training-only weight estimation and weighted-loss arithmetic."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train import compute_loss, estimate_class_weights, train_one_epoch
from test_training_epoch import TinyModel, sample


class ClassBalanceTests(unittest.TestCase):
    def test_only_selected_visible_training_labels_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.npz"
            np.savez_compressed(path,
                semantics=np.array([0,0,17,4,255],dtype=np.uint8).reshape(1,1,5),
                mask_camera=np.array([1,1,1,0,1],dtype=np.uint8).reshape(1,1,5))
            source = SimpleNamespace(records=[
                {"label_path": Path(directory) / "not-selected.npz"},
                {"label_path": path},
            ])
            weights, counts = estimate_class_weights(Subset(source,[1]))
            self.assertEqual(counts.sum().item(), 3)
            self.assertEqual(counts[0].item(), 2)
            self.assertEqual(counts[17].item(), 1)
            self.assertEqual(counts[4].item(), 0)
            self.assertAlmostEqual((weights[0]/weights[17]).item(), 1/math.sqrt(2), places=6)
            self.assertEqual(weights[4].item(), 1.)

    def test_weighted_loss_matches_manual_mean_and_masked_gradients(self):
        logits = torch.zeros(1,18,1,1,3)
        logits[:,0] = 2.
        logits.requires_grad_()
        target = torch.tensor([0,17,255]).reshape(1,1,1,3)
        mask = torch.ones_like(target,dtype=torch.bool)
        weights = torch.ones(18)
        weights[0], weights[17] = 2., .5
        loss = compute_loss(logits,target,mask,weights)
        denominator = math.log(math.exp(2)+17)
        expected = (2*(denominator-2)+.5*denominator)/2.5
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertEqual(logits.grad[...,2].abs().sum().item(),0.)

    def test_epoch_aggregation_uses_sum_of_target_weights(self):
        weights = torch.ones(18)
        weights[0], weights[17] = 2., .5
        model = TinyModel()
        loader = DataLoader([sample(mask=(True,False)),sample(boost=2.)])
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.)
        scaler = torch.amp.GradScaler("cuda",enabled=False)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = train_one_epoch(model,loader,optimizer,scaler,torch.device("cpu"),False,weights)
        denominator = math.log(math.exp(2)+17)
        expected = (2*math.log(18)+2*(denominator-2)+.5*denominator)/4.5
        self.assertAlmostEqual(stats["loss"],expected,places=6)
        self.assertEqual(stats["valid_voxels"],3)

    def test_invalid_weights_rejected(self):
        logits = torch.zeros(1,18,1,1,1)
        labels = torch.zeros(1,1,1,1,dtype=torch.int64)
        valid = torch.ones_like(labels,dtype=torch.bool)
        for bad in (torch.zeros(18),torch.full((18,),float("nan")),torch.ones(17)):
            with self.assertRaises(ValueError):
                compute_loss(logits,labels,valid,bad)


if __name__ == "__main__":
    unittest.main()
