"""Regression checks for voxel lifting, chunking and gradient propagation."""
import unittest
import torch

from src.models.view_transformer import ViewTransformer


class ViewTransformerTests(unittest.TestCase):
    def setUp(self):
        self.model = ViewTransformer(
            grid_shape=(2, 2, 2),
            voxel_bounds=(-1., -1., 2., 1., 1., 4.),
            voxel_chunk_size=3,
        )
        self.features = torch.tensor(
            [[0., 1.], [2., 3.], [10., 11.], [12., 13.]]
        ).reshape(1, 1, 2, 2, 2).repeat(1, 6, 1, 1, 1)
        self.intrinsics = torch.tensor(
            [[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]]
        ).repeat(1, 6, 1, 1)
        self.transforms = torch.eye(4).repeat(1, 6, 1, 1)
        self.transforms[:, 1, 0, 3] = 1.
        self.transforms[:, 2:, 2, 3] = -10.

    def lift(self, model=None, features=None, transforms=None):
        return (model if model is not None else self.model)(
            self.features if features is None else features,
            self.intrinsics,
            self.transforms if transforms is None else transforms,
            (32, 32),
        )

    def test_known_geometry_and_feature_ramp_preserve_voxel_order(self):
        volume = self.lift()
        # A bilinear ramp evaluates to x_feature + 2*y_feature.
        # Cameras 0 and 1 see the grid; camera 1 has a +1 camera-X shift.
        expected = []
        for x in (-0.5, 0.5):
            for y in (-0.5, 0.5):
                for z in (2.5, 3.5):
                    u = 8. + 8. * (x + 0.5) / z
                    v = 8. + 8. * y / z
                    expected.append(u / 16. + 2. * v / 16.)
        expected = torch.tensor(expected).reshape(1, 1, 2, 2, 2)
        expected = torch.cat((expected, expected + 10.), dim=1)
        torch.testing.assert_close(volume, expected)

    def test_voxels_without_valid_cameras_are_zero(self):
        transforms = self.transforms.clone()
        transforms[:, :, 2, 3] = -10.
        volume = self.lift(transforms=transforms)
        torch.testing.assert_close(volume, torch.zeros_like(volume))

    def test_chunked_outputs_and_gradients_match_whole_grid(self):
        whole = ViewTransformer(
            grid_shape=(2, 2, 2),
            voxel_bounds=(-1., -1., 2., 1., 1., 4.),
            voxel_chunk_size=8,
        )
        chunk_features = self.features.clone().requires_grad_()
        whole_features = self.features.clone().requires_grad_()
        chunk_output = self.lift(features=chunk_features)
        whole_output = self.lift(model=whole, features=whole_features)
        torch.testing.assert_close(chunk_output, whole_output)
        chunk_output.square().sum().backward()
        whole_output.square().sum().backward()
        torch.testing.assert_close(chunk_features.grad, whole_features.grad)
        self.assertGreater(chunk_features.grad.abs().sum().item(), 0.)
        self.assertEqual(chunk_features.grad[:, 2:].abs().sum().item(), 0.)

    def test_rejects_inconsistent_inputs(self):
        cases = [
            (self.features[:, :5], self.intrinsics, self.transforms, (32, 32), ValueError),
            (self.features, self.intrinsics[:, :5], self.transforms, (32, 32), ValueError),
            (self.features, self.intrinsics, self.transforms[:, :5], (32, 32), ValueError),
            (self.features, self.intrinsics, self.transforms, (16, 32), ValueError),
            (self.features.long(), self.intrinsics, self.transforms, (32, 32), TypeError),
        ]
        for features, intrinsics, transforms, size, error in cases:
            with self.subTest(error=error, shape=features.shape, size=size):
                with self.assertRaises(error):
                    self.model(features, intrinsics, transforms, size)
        with self.assertRaises(ValueError):
            ViewTransformer(voxel_chunk_size=0)


if __name__ == "__main__":
    unittest.main()
