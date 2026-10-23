"""Regression checks for diagnostic occupancy metrics and top-down rendering."""
import unittest
import numpy as np

from evaluate import occupancy_metrics, top_down_labels


class EvaluationTests(unittest.TestCase):
    def test_confusion_iou_and_ignore_mask(self):
        target = np.array([0,17,17,4,255])
        predicted = np.array([0,17,4,17,0])
        result = occupancy_metrics(predicted, target, np.ones(5, dtype=bool))
        self.assertEqual(result["valid_voxels"], 4)
        self.assertEqual(result["voxel_accuracy"], .5)
        self.assertEqual(result["classes"][0]["iou"], 1.)
        self.assertEqual(result["classes"][4]["iou"], 0.)
        self.assertAlmostEqual(result["free_iou"], 1/3)
        self.assertIsNone(result["classes"][1]["iou"])
        self.assertAlmostEqual(result["occupied_class_miou"], .5)
        self.assertAlmostEqual(result["binary_occupied_iou"], 1/3)

    def test_free_only_predictions_do_not_imply_occupied_performance(self):
        result = occupancy_metrics(
            np.full(4,17), np.array([0,17,4,17]), np.ones(4, dtype=bool)
        )
        self.assertEqual(result["voxel_accuracy"], result["free_only_baseline_accuracy"])
        self.assertEqual(result["occupied_class_miou"], 0.)
        self.assertEqual(result["binary_occupied_iou"], 0.)
        self.assertEqual(result["predicted_occupied_voxels"], 0)
        self.assertEqual(result["gt_occupied_voxels"], 2)

    def test_top_down_selects_highest_visible_occupied_and_distinguishes_unknown(self):
        target = np.array([[[11,4,17]], [[17,17,17]], [[15,15,15]]])
        mask = np.array([[[True,True,True]], [[True,True,True]], [[False,False,False]]])
        np.testing.assert_array_equal(top_down_labels(target,mask), [[4],[17],[18]])

    def test_no_supervision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No supervised"):
            occupancy_metrics(np.array([17]), np.array([255]), np.array([True]))


if __name__ == "__main__":
    unittest.main()
