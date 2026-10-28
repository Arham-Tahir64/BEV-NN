import math
import tempfile
import unittest
from pathlib import Path
import torch
from foundation4d.model import Foundation4DLite,PillarProjector,project_points
from foundation4d.losses import occupancy_losses,observed_occupancy_boundary
from foundation4d.runtime import atomic_save,rng_state,restore_rng,seed_all,build_model,run_epoch

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def cameras(self,n=1):
        k=torch.tensor([[1.,0,16],[0,1,16],[0,0,1]])[None,None].repeat(1,n,1,1)
        t=torch.eye(4)[None,None].repeat(1,n,1,1)
        return k,t

    def test_projection_pixel_centres_depth_and_motion(self):
        k,t=self.cameras();points=torch.tensor([[[0.,0,1]],[[0,0,-1]],[[100,0,1]]])
        uv,valid=project_points(points,k,t,(32,32))
        torch.testing.assert_close(uv[0,0,0,0],torch.tensor([16.,16.]))
        self.assertEqual(valid.flatten().tolist(),[True,False,False])
        t[:,:,0,3]=-1
        uv,_=project_points(torch.tensor([[[1.,0,1.]]]),k,t,(32,32))
        torch.testing.assert_close(uv.flatten(),torch.tensor([16.,16.]))

    def projector(self):
        m=PillarProjector(8,(1,1),(-.5,-.5,.5,.5,.5,1.5),heights=1,samples=1,chunk=1)
        with torch.no_grad():
            m.score.weight.zero_();m.score.bias.zero_();m.merge.weight.copy_(torch.eye(8))
            for parameter in m.position.parameters():parameter.zero_()
        return m

    def test_sampled_feature_centres_camera_fusion_permutation_and_mask(self):
        m=self.projector();k,t=self.cameras(2)
        features=torch.tensor([[1.,2.],[3.,4.]])[None,None,None].repeat(1,2,8,1,1)
        features[:,1]*=2
        result=m(features,k,t,(32,32))
        torch.testing.assert_close(result,torch.full_like(result,6.))
        perm=torch.tensor([1,0])
        torch.testing.assert_close(m(features[:,perm],k[:,perm],t[:,perm],(32,32)),result)
        t[:,1,2,3]=-10
        torch.testing.assert_close(m(features,k,t,(32,32)),torch.full_like(result,4.))

    def test_all_invalid_zero_evidence_and_finite_gradients(self):
        m=self.projector();k,t=self.cameras();t[:,:,2,3]=-10
        features=torch.randn(1,1,8,2,2,requires_grad=True)
        result=m(features,k,t,(32,32));self.assertEqual(result.abs().sum().item(),0)
        result.sum().backward();self.assertEqual(features.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_chunking_preserves_output_and_learned_offsets_receive_gradients(self):
        kwargs=dict(channels=8,bev_shape=(3,2),bounds=(-1.,-1.,1.,1.,1.,3.),heights=2,samples=4,learned_offsets=True)
        small=PillarProjector(**kwargs,chunk=1);large=PillarProjector(**kwargs,chunk=100)
        large.load_state_dict(small.state_dict());k,t=self.cameras()
        features=torch.randn(1,1,8,4,4,requires_grad=True)
        a=small(features,k,t,(64,64));b=large(features,k,t,(64,64))
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.square().mean().backward()
        self.assertGreater(small.offset.weight.grad.abs().sum().item(),0)
        self.assertGreater(features.grad.abs().sum().item(),0)

    def test_full_model_xyz_head_layout_and_inference_contract(self):
        model=Foundation4DLite(channels=8,bev_shape=(4,3),grid_shape=(8,6,4),pretrained=False,layers=1,samples=1)
        model.eval();k,t=self.cameras(6);images=torch.randn(1,6,3,32,32)
        with torch.no_grad():
            model.binary_head.weight.zero_();model.binary_head.bias.copy_(torch.tensor([-2.,-1.,1.,2.]))
            model.semantic_head.weight.zero_();model.semantic_head.bias.copy_(torch.arange(17*4).float())
            result=model(images,k,t,return_aux=True)
            self.assertEqual(result['semantic_logits'].shape,(1,17,8,6,4))
            torch.testing.assert_close(result['semantic_logits'][0,3,0,0],torch.arange(12.,16.))
            predicted=model.predict(result)
            torch.testing.assert_close(predicted,result['occupancy_logits'].argmax(1))
            self.assertTrue((predicted[...,:2]==17).all());self.assertTrue((predicted[...,2:]==16).all())
            mask=torch.zeros(1,6,dtype=torch.bool)
            zero=model.forward_heads(images,k,t,camera_mask=mask)
            self.assertTrue(torch.isfinite(zero['binary_logits']).all())

    def test_losses_mask_free_ignore_and_preserve_class_zero(self):
        binary=torch.zeros(1,1,1,3,requires_grad=True);semantic=torch.zeros(1,17,1,1,3,requires_grad=True)
        labels=torch.tensor([0,17,255]).reshape(1,1,1,3);mask=torch.ones_like(labels,dtype=torch.bool)
        terms=occupancy_losses(dict(binary_logits=binary,semantic_logits=semantic),labels,mask)
        self.assertEqual((terms['known'],terms['occupied']),(2,1))
        self.assertAlmostEqual(terms['binary'].item(),math.log(2),places=6)
        self.assertAlmostEqual(terms['semantic'].item(),math.log(17),places=6)
        (terms['binary']+terms['semantic']).backward()
        self.assertEqual(binary.grad[...,2].abs().sum().item(),0)
        self.assertEqual(semantic.grad[...,1:].abs().sum().item(),0)
        self.assertLess(binary.grad[...,0].item(),0);self.assertGreater(binary.grad[...,1].item(),0)

    def test_all_free_and_unknown_are_safe(self):
        for label in (17,255):
            b=torch.zeros(1,1,1,2,requires_grad=True);s=torch.zeros(1,17,1,1,2,requires_grad=True)
            y=torch.full_like(b,label,dtype=torch.long);v=torch.ones_like(y,dtype=torch.bool)
            terms=occupancy_losses(dict(binary_logits=b,semantic_logits=s),y,v)
            (terms['binary']+terms['semantic']).backward()
            self.assertEqual(s.grad.abs().sum().item(),0);self.assertTrue(torch.isfinite(b.grad).all())

    def test_boundary_exact_endpoints_and_unknown(self):
        labels=torch.tensor([0,17,255,0]).reshape(1,1,1,4)
        result=observed_occupancy_boundary(labels,labels!=255)
        self.assertEqual(result.flatten().tolist(),[True,True,False,False])

    def test_checkpoint_weights_only_and_rng_roundtrip(self):
        config=dict(architecture='foundation4d_lite',channels=8,bev_shape=[4,4],grid_shape=[8,8,4],layers=1,samples=1,pretrained=False)
        model=build_model(config);optimizer=torch.optim.AdamW(model.parameters())
        p=next(p for p in model.parameters() if p.requires_grad);p.sum().backward();optimizer.step()
        seed_all(9);state=rng_state();expected=torch.rand(3)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            atomic_save(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=state,config=config))
            saved=torch.load(path,map_location='cpu',weights_only=True)
            clone=build_model(saved['config']);clone.load_state_dict(saved['model'],strict=True)
            opt=torch.optim.AdamW(clone.parameters());opt.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng']);torch.testing.assert_close(expected,torch.rand(3))
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_model_configuration_rejected(self):
        for extra in ({'channels':7},{'threshold':1},{'query_chunk':0},{'layers':0},{'voxel_bounds':[0,0,0,0,1,1]}):
            with self.assertRaises(ValueError):Foundation4DLite(pretrained=False,**extra)


if __name__=='__main__':unittest.main()
