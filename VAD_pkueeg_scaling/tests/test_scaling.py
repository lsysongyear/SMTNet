import unittest

import numpy as np

from scaling.metrics import compute_metrics
from scaling.duration import prefix_allocation
from scaling.subset import SCALES, assert_nested_subsets, build_scaling_split


class ScalingSubsetTests(unittest.TestCase):
    def test_seed_545_keeps_validation_and_test_fixed(self):
        split = build_scaling_split(0.05, subset_seed=1001, split_seed=545)
        self.assertEqual(split.val_stories, (10, 18, 19, 21, 28))
        self.assertEqual(split.test_stories, (1, 7, 11, 12, 17))

    def test_all_scales_keep_complete_partition_unit_pools(self):
        expected_counts = [40] * len(SCALES)
        actual_counts = [
            len(build_scaling_split(scale, 1001, 545).train_stories)
            for scale in SCALES
        ]
        self.assertEqual(actual_counts, expected_counts)
        assert_nested_subsets(1001, 545)

    def test_subset_seed_no_longer_changes_prefix_membership(self):
        first = build_scaling_split(0.05, subset_seed=1001, split_seed=545)
        second = build_scaling_split(0.05, subset_seed=9999, split_seed=545)
        self.assertEqual(first.train_stories, second.train_stories)

    def test_duration_prefix_can_end_inside_third_trial(self):
        target, selected = prefix_allocation([100] * 50, 0.05)
        self.assertEqual(target, 250)
        self.assertEqual(selected[:4], (100, 100, 50, 0))

    def test_full_scale_contains_all_training_stories(self):
        split = build_scaling_split(1.0, subset_seed=1001, split_seed=545)
        self.assertEqual(split.train_stories, split.train_pool)


class MetricTests(unittest.TestCase):
    def test_macro_accuracy_is_mean_class_recall(self):
        targets = np.array([0, 0, 1, 1])
        predictions = np.array([0, 0, 0, 1])
        metrics = compute_metrics(targets, predictions)
        self.assertAlmostEqual(metrics["macro_acc"], 0.75)
        self.assertAlmostEqual(metrics["binary_acc"], 0.75)


if __name__ == "__main__":
    unittest.main()
