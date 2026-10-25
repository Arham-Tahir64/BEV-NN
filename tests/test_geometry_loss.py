"""Test occupancy grouping, unknown masking, numerical stability, and integration."""
import contextlib
import io
import math
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from src.losses import occupied_free_loss
from train import train_one_epoch, parse_args
from test_training_epoch import TinyModel, sample


class GeometryLossTests(unittest.TestCase):
    def test_matches_binary_probability_and_masks_unknown_gradients(self):
        torch.manual_seed(0)
        scores = torch.randn(1,18,1,1,4,requires_grad=True)
        labels = torch.tensor([0,17,255,15]).reshape(1,1,1,4)
        mask = torch.tensor([True,True,True,False]).reshape_as(labels)
        loss,count = occupied_free_loss(scores,labels,mask)
        p = scores.softmax(1)[:,17].flatten()
        expected = (-torch.log1p(-p[0])-p[1].log())/2
        torch.testing.assert_close(loss,expected)
        self.assertEqual(count,2)
        loss.backward()
        torch.testing.assert_close(scores.grad[...,2:],torch.zeros_like(scores.grad[...,2:]))

    def test_all_occupied_classes_have_same_binary_target(self):
        x = torch.zeros(1,18,1,1,1)
        mask = torch.ones(1,1,1,1,dtype=torch.bool)
        losses = [occupied_free_loss(x,torch.full((1,1,1,1),label),mask)[0]
                  for label in (0,4,15,16)]
        self.assertTrue(all(torch.equal(v,losses[0]) for v in losses))

    def test_gradient_pushes_false_occupancy_towards_free(self):
        x = torch.zeros(1,18,1,1,1,requires_grad=True)
        loss,_ = occupied_free_loss(x,torch.full((1,1,1,1),17),torch.ones(1,1,1,1,dtype=torch.bool))
        loss.backward()
        self.assertLess(x.grad[:,17].item(),0)
        self.assertTrue((x.grad[:,:17]>0).all())

    def test_fp16_extreme_logits_are_finite_and_differentiable(self):
        x = torch.zeros(1,18,1,1,2,dtype=torch.float16)
        x[:,0,...,0]=10000
        x[:,17,...,1]=10000
        x.requires_grad_()
        loss,count = occupied_free_loss(x,torch.tensor([17,15]).reshape(1,1,1,2),
                                         torch.ones(1,1,1,2,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(),0)

    def test_no_supervision_is_differentiable_zero(self):
        x = torch.zeros(1,18,1,1,1,requires_grad=True)
        loss,count = occupied_free_loss(x,torch.full((1,1,1,1),255),torch.ones(1,1,1,1,dtype=torch.bool))
        self.assertEqual(count,0)
        loss.backward()
        torch.testing.assert_close(x.grad,torch.zeros_like(x))

    def test_enabled_loss_changes_updates_but_preserves_semantic_loss_reporting(self):
        models=[TinyModel(),TinyModel()]
        stats=[]
        for model,weight in zip(models,(0.,.5)):
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            with contextlib.redirect_stdout(io.StringIO()):
                stats.append(train_one_epoch(model,DataLoader([sample(labels=(17,17))]),
                    optimizer,scaler,torch.device('cpu'),False,geometry_loss_weight=weight))
        self.assertAlmostEqual(stats[0]['loss'],stats[1]['loss'],places=6)
        self.assertAlmostEqual(stats[1]['geometry_loss'],math.log(18),places=6)
        self.assertEqual(stats[1]['geometry_voxels'],2)
        self.assertNotIn('geometry_loss',stats[0])
        self.assertGreater(models[1].scores[17].item(),models[0].scores[17].item())

    def test_invalid_weight_rejected(self):
        model=TinyModel()
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        for weight in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                train_one_epoch(model,[],optimizer,scaler,torch.device('cpu'),False,
                                geometry_loss_weight=weight)

    def test_cli_disables_by_default_and_rejects_invalid_weights(self):
        with patch('sys.argv',['train.py']):
            self.assertEqual(parse_args().geometry_loss_weight,0)
        for value in ('-1','nan','inf'):
            with patch('sys.argv',['train.py','--geometry-loss-weight',value]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


if __name__=='__main__': unittest.main()
