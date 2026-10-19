"""Training tensor and calibration regression checks."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from data.nuscenes_dataset import (
    NuScenesOccupancyDataset, prepare_camera_image, resize_intrinsics,
    IMAGE_MEAN, IMAGE_STD, make_dataloader,
)


class TrainingDatasetTests(unittest.TestCase):
    def test_resize_intrinsics_preserves_pixel_centres(self):
        k = np.array([[100., 0, 50], [0, 120, 40], [0, 0, 1]])
        resized = resize_intrinsics(k, (100, 200), (50, 50))
        np.testing.assert_allclose(resized, [[25, 0, 12.125], [0, 60, 19.75], [0, 0, 1]])
        np.testing.assert_array_equal(resize_intrinsics(k, (100, 200), (100, 200)), k)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.image_path = root / "rgb.png"
        Image.new("RGB", (8, 4), (255, 128, 0)).save(self.image_path)
        self.label_path = root / "labels.npz"
        self.labels = np.full((200, 200, 16), 17, dtype=np.uint8)
        self.labels[0, 0, :3] = [0, 255, 4]
        self.mask = np.ones_like(self.labels, dtype=bool)
        self.mask[0, 0, 2] = False
        self.dataset = NuScenesOccupancyDataset.__new__(NuScenesOccupancyDataset)
        self.dataset.image_size = (4, 8)
        self.dataset.split = "mini_train"
        self.dataset.records = [{
            "sample_token": "test", "scene_name": "scene-test",
            "image_paths": (self.image_path,) * 6, "label_path": self.label_path,
            "intrinsics": np.tile(np.eye(3, dtype=np.float32), (6, 1, 1)),
            "ego_to_camera": np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)),
        }]
        self.save()

    def save(self):
        np.savez_compressed(self.label_path, semantics=self.labels, mask_camera=self.mask)

    def test_rgb_normalization_and_channel_order(self):
        image = prepare_camera_image(self.image_path, (2, 4))
        expected = (np.array([1, 128 / 255, 0]) - IMAGE_MEAN) / IMAGE_STD
        self.assertEqual(tuple(image.shape), (3, 2, 4))
        self.assertEqual(image.dtype, torch.float32)
        np.testing.assert_allclose(image[:, 0, 0].numpy(), expected, atol=1e-6)

    def test_dense_targets_preserve_free_ignore_and_invisible(self):
        item = self.dataset[0]
        np.testing.assert_array_equal(item["semantics"].numpy(), self.labels)
        self.assertEqual(item["semantics"].dtype, torch.int64)
        self.assertTrue(item["valid_mask"][0, 0, 0])  # occupied class zero
        self.assertFalse(item["valid_mask"][0, 0, 1])  # ignore
        self.assertFalse(item["valid_mask"][0, 0, 2])  # invisible occupied
        self.assertTrue(item["valid_mask"][0, 0, 3])  # visible free
        item["intrinsics"].zero_()
        self.assertEqual(self.dataset.records[0]["intrinsics"][0, 0, 0], 1)
        batch = next(iter(make_dataloader(self.dataset, pin_memory=False)))
        self.assertEqual(tuple(batch["images"].shape), (1, 6, 3, 4, 8))

    def test_invalid_class_rejected(self):
        self.labels[0, 0, 0] = 18
        self.save()
        with self.assertRaisesRegex(ValueError, "class IDs"):
            self.dataset[0]

    def test_wrong_grid_rejected(self):
        self.labels = self.labels[:2]
        self.mask = self.mask[:2]
        self.save()
        with self.assertRaisesRegex(ValueError, "Expected grid"):
            self.dataset[0]


if __name__ == "__main__":
    unittest.main()
