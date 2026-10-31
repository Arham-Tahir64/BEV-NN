import unittest
import torch
from foundation4d.model import Foundation4DLite
from foundation4d.geometry import supported_mean
from foundation4d.losses import boundary_binary_loss,observed_ray_loss


class GeometryTests(unittest.TestCase):
    def test_support_cannot_cancel_in_normalization(self):
        v=torch.ones(1,1,1,1,1,8);w=torch.ones(1,1,1,1,1)
        strong=supported_mean(v,w,w)[0];weak=supported_mean(v,w,w*.01)[0]
        self.assertLess(float(weak.max()),float(strong.max())/20)
        self.assertEqual(float(supported_mean(v,w,w*0)[0].abs().sum()),0)

    def test_depth_and_fine_features_receive_occupancy_gradients(self):
        torch.set_num_threads(2)
        m=Foundation4DLite(channels=8,bev_shape=(2,2),grid_shape=(4,4,16),
            voxel_bounds=(-1,-1,1,1,1,5),layers=1,query_chunk=3,pretrained=False,geometry_aware=True).eval()
        images=torch.randn(1,6,3,32,32)
        k=torch.tensor([[8.,0,16],[0,8,16],[0,0,1]])[None,None].repeat(1,6,1,1)
        t=torch.eye(4)[None,None].repeat(1,6,1,1)
        h=m.forward_heads(images,k,t)
        h['binary_logits'].square().mean().backward()
        self.assertTrue(torch.isfinite(h['binary_logits']).all())
        self.assertGreater(float(m.depth_head.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.projectors[0].fine_projection.weight.grad.abs().sum()),0)
        with torch.no_grad():
            invalid=m.forward_heads(images,k,t,camera_mask=torch.zeros(1,6,dtype=torch.bool))
            self.assertTrue(torch.isfinite(invalid['binary_logits']).all())
            perm=torch.tensor([5,3,0,2,1,4])
            other=m.forward_heads(images[:,perm],k[:,perm],t[:,perm])
            torch.testing.assert_close(h['binary_logits'],other['binary_logits'],rtol=1e-4,atol=1e-5)

    def test_boundary_unknown_zero_gradient(self):
        x=torch.zeros(1,1,1,3,requires_grad=True);y=torch.tensor([[[[0,17,255]]]])
        boundary_binary_loss(x,y,torch.ones_like(y,dtype=torch.bool)).backward()
        self.assertLess(x.grad[0,0,0,0],0);self.assertGreater(x.grad[0,0,0,1],0)
        self.assertEqual(x.grad[0,0,0,2],0)

    def test_ray_does_not_carve_unknown_or_behind_surface(self):
        x=torch.zeros(1,1,1,5,requires_grad=True);y=torch.tensor([[[[17,255,0,17,0]]]])
        k=torch.tensor([[1.,0,8],[0,1,8],[0,0,1]])[None,None]
        t=torch.eye(4)[None,None]
        observed_ray_loss(x,y,torch.ones_like(y,dtype=torch.bool),k,t,(-.1,-.1,1,.1,.1,6),(16,16)).backward()
        g=x.grad.flatten()
        self.assertGreater(g[0],0);self.assertLess(g[2],0)
        self.assertEqual(g[1],0);self.assertEqual(g[3],0);self.assertEqual(g[4],0)
