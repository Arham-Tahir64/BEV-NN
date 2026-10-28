import contextlib
import io
import sys
import unittest
from unittest.mock import patch
import torch
from data.depth_targets import depth_range_weights, depth_loss
from src.models.bevdepth import FrustumRefinement, SupportedFrustumRefinement
from src.models.occupancy_model import OccupancyModel
from train import parse_args


class DepthSupportTests(unittest.TestCase):
    def test_capped_mean_one_range_weights(self):
        values = torch.tensor([5.]*90+[15.]*9+[30.])
        w = depth_range_weights(values)
        torch.testing.assert_close(w.mean(), torch.tensor(1.))
        self.assertLessEqual(w.max().item(), 2.)
        self.assertGreater(w[-1], w[0])
        torch.testing.assert_close(depth_range_weights(values, 1), torch.ones_like(values))
        torch.testing.assert_close(depth_range_weights(torch.tensor([5.,6.])), torch.ones(2))

    def test_range_boundaries_and_equal_contribution_without_cap(self):
        values = torch.tensor([1., 2., 10., 20.])
        w = depth_range_weights(values, 10)
        torch.testing.assert_close(w, torch.tensor([2/3,2/3,4/3,4/3]))
        self.assertEqual(depth_range_weights(torch.empty(0)).numel(), 0)
        for cap in (0, .9, float('nan'), float('inf')):
            with self.assertRaises(ValueError): depth_range_weights(values, cap)

    def test_weighted_losses_match_manual_and_ignore_unknown(self):
        target = torch.tensor([5.,6.,7.,25.,float('nan')]).reshape(1,1,1,5)
        edges = torch.linspace(.1,64.1,65)
        for mode in ('hard','linear','bce'):
            logits = torch.randn(1,1,64,1,5,requires_grad=True)
            weighted,n = depth_loss(logits,target,edges,mode,range_balance=True)
            individual = torch.stack([depth_loss(logits[...,i:i+1],target[...,i:i+1],edges,mode)[0] for i in range(4)])
            weights = depth_range_weights(target.flatten()[:4])
            torch.testing.assert_close(weighted,(individual*weights).sum()/weights.sum())
            self.assertEqual(n,4)
            weighted.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad[...,4].abs().sum().item(),0)
        empty,n = depth_loss(logits,torch.full_like(target,float('nan')),edges,range_balance=True)
        self.assertEqual((empty.item(),n),(0,0))

    def test_supported_gate_never_extends_beyond_neighbor_bins(self):
        p=torch.zeros(1,1,9,2,3);p[:,:,4]=1
        gate=SupportedFrustumRefinement.support_gate(p)
        torch.testing.assert_close(gate.sum(2),torch.ones(1,1,2,3))
        self.assertEqual(gate[:,:,:3].abs().sum().item(),0)
        self.assertEqual(gate[:,:,6:].abs().sum().item(),0)
        for depth in (0,8):
            p.zero_();p[:,:,depth]=1
            torch.testing.assert_close(SupportedFrustumRefinement.support_gate(p).sum(2),torch.ones(1,1,2,3))

    def test_zero_gain_identity_then_gradients_and_no_unsupported_features(self):
        block=SupportedFrustumRefinement(8)
        p=torch.zeros(1,1,9,2,3);p[:,:,4]=1
        context=torch.randn(1,1,8,2,3,requires_grad=True)
        volume=context.unsqueeze(3)*p.unsqueeze(2)
        result=block(volume,p)
        torch.testing.assert_close(result,volume,rtol=0,atol=0)
        result.square().sum().backward(retain_graph=True)
        self.assertTrue(torch.isfinite(block.gain.grad))
        self.assertGreater(block.gain.grad.abs().item(),0)
        block.zero_grad();context.grad=None
        with torch.no_grad():block.gain.fill_(.5)
        result=block(volume,p)
        self.assertEqual(result[:,:,:,:3].abs().sum().item(),0)
        self.assertEqual(result[:,:,:,6:].abs().sum().item(),0)
        result.square().mean().backward()
        self.assertGreater(block.output.weight.grad.abs().sum().item(),0)
        self.assertGreater(context.grad.abs().sum().item(),0)

    def test_legacy_state_and_supported_strict_restore(self):
        args=dict(architecture='bevdepth',use_depth=True,depth_fusion='bevdepth',pretrained=False,
                  grid_shape=(8,8,4),depth_bins=4,depth_max=4.1)
        torch.manual_seed(2);old=OccupancyModel(**args)
        torch.manual_seed(2);legacy=OccupancyModel(**args,frustum_refinement='legacy')
        legacy.load_state_dict(old.state_dict(),strict=True)
        self.assertIs(type(legacy.view_transformer.refinement),FrustumRefinement)
        supported=OccupancyModel(**args,frustum_refinement='supported')
        clone=OccupancyModel(**args,frustum_refinement='supported')
        clone.load_state_dict(supported.state_dict(),strict=True)
        with self.assertRaises(RuntimeError):legacy.load_state_dict(supported.state_dict(),strict=True)

    def test_cli_defaults_and_invalid_combinations(self):
        with patch.object(sys,'argv',['train']):args=parse_args()
        self.assertEqual(args.frustum_refinement,'legacy');self.assertFalse(args.depth_range_balance)
        for extra in (['--depth-range-balance'],['--depth-range-cap','nan'],['--depth-range-cap','.5'],
                      ['--frustum-refinement','supported']):
            with patch.object(sys,'argv',['train',*extra]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):parse_args()


if __name__=='__main__':unittest.main()
