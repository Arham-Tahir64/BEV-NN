"""Occupancy archive validation tests; fixtures are temporary synthetic arrays."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.occupancy_labels import load_occupancy_labels, load_sample_occupancy


class OccupancyLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "labels.npz"
        self.semantics = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        self.semantics[0, 0, 0] = 255
        self.mask = np.indices(self.semantics.shape).sum(axis=0) % 2
        self.payload = {
            "semantics": self.semantics,
            "mask_camera": self.mask.astype(np.uint8),
        }

    def write(self, **updates):
        np.savez_compressed(self.path, **(self.payload | updates))

    def test_preserves_labels_and_returns_boolean_camera_mask(self):
        self.write()
        arrays = load_occupancy_labels(self.path)
        np.testing.assert_array_equal(arrays["semantics"], self.semantics)
        self.assertEqual(arrays["semantics"].dtype, np.uint8)
        self.assertEqual(int(arrays["semantics"][0, 0, 0]), 255)
        for key in ("mask_camera",):
            self.assertEqual(arrays[key].dtype, np.bool_)
            np.testing.assert_array_equal(arrays[key], self.payload[key])
        self.assertEqual(set(arrays), {"semantics", "mask_camera"})
        self.path.unlink()  # The archive handle is closed; arrays are detached.
        self.assertEqual(int(arrays["semantics"][0, 0, 0]), 255)

    def test_does_not_decode_unused_archive_arrays(self):
        # An object array would fail allow_pickle=False if accessed.
        self.write(mask_lidar=np.array(["unused"], dtype=object))
        arrays = load_occupancy_labels(self.path)
        self.assertEqual(set(arrays), {"semantics", "mask_camera"})
        np.testing.assert_array_equal(arrays["semantics"], self.semantics)

    def test_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "separate annotations"):
            load_occupancy_labels(self.path)

    def test_missing_required_arrays(self):
        for key in self.payload:
            with self.subTest(key=key):
                np.savez(self.path, **{k: v for k, v in self.payload.items() if k != key})
                with self.assertRaisesRegex(ValueError, key):
                    load_occupancy_labels(self.path)

    def test_bad_semantics_shape_or_dtype(self):
        for semantics in (np.zeros((2, 3)), np.zeros((0, 3, 4), dtype=np.int32),
                          self.semantics.astype(float)):
            with self.subTest(shape=semantics.shape, dtype=semantics.dtype):
                self.write(semantics=semantics)
                with self.assertRaises(ValueError):
                    load_occupancy_labels(self.path)

    def test_mismatched_masks(self):
        for key in ("mask_camera",):
            with self.subTest(key=key):
                self.write(**{key: np.zeros((3, 2, 4), dtype=bool)})
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_occupancy_labels(self.path)

    def test_nonbinary_masks(self):
        for key in ("mask_camera",):
            for value in (-1, 2, 255):
                with self.subTest(key=key, value=value):
                    self.write(**{key: np.full(self.semantics.shape, value, dtype=np.int32)})
                    with self.assertRaisesRegex(ValueError, "0/1"):
                        load_occupancy_labels(self.path)

    def test_object_arrays_rejected(self):
        self.write(semantics=np.full(self.semantics.shape, "unsafe", dtype=object))
        with self.assertRaises(ValueError):
            load_occupancy_labels(self.path)

    def test_resolves_scene_and_sample_token(self):
        records = {
            ("sample", "sample-token"): {"scene_token": "scene-token"},
            ("scene", "scene-token"): {"name": "scene-0061"},
        }
        class Metadata:
            def get(self, table, token):
                return records[(table, token)]
        path = self.root / "scene-0061" / "sample-token" / "labels.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, **self.payload)
        arrays = load_sample_occupancy(Metadata(), "sample-token", self.root)
        np.testing.assert_array_equal(arrays["semantics"], self.semantics)


if __name__ == "__main__":
    unittest.main()
