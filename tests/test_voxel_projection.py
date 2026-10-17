"""Geometry tests with independent expected coordinates and temporary labels."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from data.nuscenes_dataset import CAMERAS
from data.voxel_projection import (
    GRID_SHAPE, PALETTE, occupied_voxel_centres, camera_geometry,
    project_centres, visible_projection, draw_voxel_overlay, build_occupancy_timeline,
)


class Metadata:
    def __init__(self, records):
        self.records = records

    def get(self, table, token):
        return self.records[(table, token)]


class ProjectionTests(unittest.TestCase):
    def occupancy(self):
        return {"semantics": np.full(GRID_SHAPE, 17, np.uint8),
                "mask_camera": np.ones(GRID_SHAPE, bool)}

    def test_centres_have_half_voxel_offset_and_correct_axis_order(self):
        occ = self.occupancy()
        occ["semantics"][0, 1, 2] = 11
        points, labels = occupied_voxel_centres(occ)
        np.testing.assert_allclose(points, [[-39.8, -39.4, 0]])
        np.testing.assert_array_equal(labels, [11])

    def test_free_ignore_and_camera_invisible_voxels_are_excluded(self):
        occ = self.occupancy()
        occ["semantics"][0, 0, 0] = 255
        occ["semantics"][1, 1, 1] = 4
        occ["mask_camera"][1, 1, 1] = False
        points, labels = occupied_voxel_centres(occ)
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(labels.size, 0)

    def test_surface_filter_removes_interior_but_keeps_outer_shell(self):
        occ = self.occupancy()
        occ["semantics"][3:6, 3:6, 3:6] = 15
        points, _ = occupied_voxel_centres(occ)
        self.assertEqual(len(points), 26)
        centre = np.array([-40, -40, -1]) + np.array([4.5, 4.5, 4.5]) * 0.4
        self.assertFalse(np.any(np.all(np.isclose(points, centre), axis=1)))

    def test_wrong_grid_and_class_mapping_fail(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            occupied_voxel_centres({"semantics": np.zeros((2, 2, 2)), "mask_camera": np.ones((2, 2, 2))})
        occ = self.occupancy()
        occ["semantics"][0, 0, 0] = 99
        with self.assertRaisesRegex(ValueError, "semantic IDs"):
            occupied_voxel_centres(occ)

    def test_pinhole_projection_and_frustum_filter(self):
        points = np.array([[0, 0, 2], [1, 0, 2], [0, 0, -2], [4, 0, 2], [np.nan, 0, 2]])
        intrinsics = np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]])
        result = project_centres(points, np.arange(5), np.eye(4), intrinsics, (200, 100))
        np.testing.assert_allclose(result["uv"], [[100, 50], [150, 50]])
        np.testing.assert_array_equal(result["labels"], [0, 1])
        np.testing.assert_allclose(result["depth"], [2, 2])

    def test_reference_time_and_camera_time_poses_and_extrinsics(self):
        q = np.sqrt(0.5)
        records = {
            ("sample_data", "reference"): {"timestamp": 100, "ego_pose_token": "reference-pose"},
            ("ego_pose", "reference-pose"): {"rotation": [1, 0, 0, 0], "translation": [10, 0, 0]},
            ("ego_pose", "camera-pose"): {"rotation": [1, 0, 0, 0], "translation": [12, 0, 0]},
            ("calibrated_sensor", "calibration"): {
                "rotation": [q, 0, 0, q], "translation": [1, 0, 0],
                "camera_intrinsic": np.eye(3).tolist(),
            },
        }
        sample = {"timestamp": 100, "data": {"REFERENCE": "reference"}}
        for channel in CAMERAS:
            sample["data"][channel] = channel
            records[("sample_data", channel)] = {
                "timestamp": 110, "ego_pose_token": "camera-pose",
                "calibrated_sensor_token": "calibration", "width": 1600, "height": 900,
            }
        result = camera_geometry(Metadata(records), sample)
        for geometry in result.values():
            transformed = geometry["ego_to_camera"] @ [4, 2, 5, 1]
            np.testing.assert_allclose(transformed, [2, -1, 5, 1], atol=1e-12)
            self.assertEqual(geometry["image_size"], (1600, 900))
        sample["timestamp"] = 999
        with self.assertRaisesRegex(ValueError, "exact sample-time"):
            camera_geometry(Metadata(records), sample)

    def test_z_buffer_keeps_nearer_centre_independently_of_input_order(self):
        projection = {"uv": np.array([[12., 14.], [13., 15.], [60., 30.]]),
                      "depth": np.array([10., 2., 3.]), "labels": np.array([15, 4, 11]),
                      "image_size": (100, 100)}
        result = visible_projection(projection, (50, 50), spacing=10)
        self.assertEqual(set(result["labels"]), {4, 11})
        np.testing.assert_allclose(result["uv"][result["labels"] == 4], [[6.25, 7.25]])

    def test_drawing_uses_semantic_colour_and_preserves_original(self):
        image = Image.new("RGB", (50, 50))
        projection = {"uv": np.array([[20., 20.]]), "depth": np.array([1.]),
                      "labels": np.array([11]), "image_size": (50, 50)}
        overlay = draw_voxel_overlay(image, projection)
        self.assertEqual(overlay.getpixel((20, 20)), tuple(PALETTE[11]))
        self.assertEqual(image.getpixel((20, 20)), (0, 0, 0))
        projection["uv"] = np.empty((0, 2))
        projection["depth"] = np.empty(0)
        projection["labels"] = np.empty(0, dtype=int)
        np.testing.assert_array_equal(draw_voxel_overlay(image, projection), image)

    def test_gt_timeline_uses_exact_keyframes_and_reports_missing(self):
        records = {}
        for i, token in enumerate(("first", "second")):
            records[("sample", token)] = {
                "token": token, "timestamp": 100 + i * 500_000,
                "data": {channel: f"{token}-{channel}" for channel in CAMERAS},
                "next": "second" if i == 0 else "",
            }
        nusc = Metadata(records)
        nusc.scene = [{"name": "scene-0061", "first_sample_token": "first"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene-0061" / "first" / "labels.npz"
            path.parent.mkdir(parents=True)
            path.touch()
            frames = build_occupancy_timeline(nusc, occupancy_root=folder)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["sample_token"], "first")
            self.assertEqual(frames[0]["camera_tokens"], records[("sample", "first")]["data"])
            with self.assertRaises(ValueError):
                build_occupancy_timeline(nusc, scene_name="missing", occupancy_root=folder)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                build_occupancy_timeline(nusc, occupancy_root=folder)


if __name__ == "__main__":
    unittest.main()
