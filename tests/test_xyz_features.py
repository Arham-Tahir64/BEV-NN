"""Check XYZ ordering, unchanged decoder, checkpoint compatibility and gradients."""
import unittest
import torch
from torch.nn import functional as F

from src.models.occupancy_model import OccupancyModel


class XYZFeatureTests(unittest.TestCase):
    def make_model(self, use_xyz):
        torch.manual_seed(0)
        return OccupancyModel(
            feature_channels=4, hidden_channels=4, pretrained=False,
            grid_shape=(2,3,4), voxel_bounds=(-1.,-1.,2.,1.,1.,4.),
            voxel_chunk_size=7, use_xyz=use_xyz,
        )

    def test_coordinates_follow_voxel_centres_and_device_dtype(self):
        model = self.make_model(True)
        self.assertEqual(tuple(model.voxel_coordinates.shape),(1,3,2,3,4))
        torch.testing.assert_close(model.voxel_coordinates[0,:,0,0,0],
                                   torch.tensor([-.5,-2/3,-.75]))
        torch.testing.assert_close(model.voxel_coordinates[0,:,-1,-1,-1],
                                   torch.tensor([.5,2/3,.75]))
        self.assertLess(model.voxel_coordinates.abs().max().item(),1.)
        self.assertNotIn("voxel_coordinates",model.state_dict())
        model.double()
        self.assertEqual(model.voxel_coordinates.dtype,torch.float64)

    def test_initialization_preserves_baseline_decoder_and_old_state_keys(self):
        baseline = self.make_model(False)
        xyz = self.make_model(True)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value,xyz.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(xyz.occupancy_decoder.in_channels,4)
        self.assertEqual(set(xyz.state_dict())-set(baseline.state_dict()),{"xyz_projection.weight"})
        baseline.load_state_dict(baseline.state_dict(),strict=True)

    def test_zero_initialized_xyz_preserves_output_and_receives_gradients(self):
        baseline = self.make_model(False)
        model = self.make_model(True)
        images = torch.randn(1,6,3,32,48)
        k = torch.tensor([[8.,0.,12.],[0.,8.,8.],[0.,0.,1.]]).repeat(1,6,1,1)
        t = torch.eye(4).repeat(1,6,1,1)
        with torch.no_grad():
            expected = baseline(images,k,t)
        logits = model(images,k,t)
        torch.testing.assert_close(logits,expected)
        targets = (torch.arange(24)%18).reshape(1,2,3,4)
        F.cross_entropy(logits,targets).backward()
        self.assertIsNotNone(model.xyz_projection.weight.grad)
        self.assertTrue(torch.isfinite(model.xyz_projection.weight.grad).all())
        self.assertGreater(model.xyz_projection.weight.grad.abs().sum().item(),0.)
        self.assertGreater(model.image_encoder.projection.weight.grad.abs().sum().item(),0.)
        self.assertTrue(all(p.grad is None for p in model.image_encoder.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
