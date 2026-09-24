import unittest

import numpy as np

from scaling.duration import prefix_allocation
from scaling.metrics import compute_metrics
from scaling.subset import (
    ALL_SUBJECTS,
    SCALES,
    assert_paper_split,
    assert_nested_subsets,
    build_subject_splits,
    nested_subset_indices,
    partition_pairs,
)
from speech_code.utils import _allocate_sparkulee_prefix


COMMON_STORIES = (
    "audiobook-2-1",
    "audiobook-2-2",
    "audiobook-3",
    "audiobook-5-1",
    "audiobook-5-3",
    "audiobook-6-1",
)


class SubjectSplitTests(unittest.TestCase):
    def setUp(self):
        self.subject_stories = {
            subject: COMMON_STORIES + (f"subject-{subject}-only",)
            for subject in ALL_SUBJECTS
        }

    def test_seed_five_reproduces_shared_holdout(self):
        splits = build_subject_splits(self.subject_stories, split_seed=5)
        for split in splits.values():
            self.assertEqual(split.val_stories, ("audiobook-2-1",))
            self.assertEqual(split.test_stories, ("audiobook-2-2",))

    def test_seed_eighty_four_reproduces_candidate_eighteen(self):
        splits = build_subject_splits(self.subject_stories, split_seed=84)
        assert_paper_split(splits)
        for split in splits.values():
            self.assertEqual(split.val_stories, ("audiobook-5-1",))
            self.assertEqual(split.test_stories, ("audiobook-5-3",))

    def test_every_subject_has_disjoint_partitions(self):
        splits = build_subject_splits(self.subject_stories, split_seed=5)
        for split in splits.values():
            train = set(split.train_stories)
            val = set(split.val_stories)
            test = set(split.test_stories)
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
        self.assertEqual(len(partition_pairs(splits, "val")), len(ALL_SUBJECTS))
        self.assertEqual(len(partition_pairs(splits, "test")), len(ALL_SUBJECTS))

        train_story_union = {
            story for split in splits.values() for story in split.train_stories
        }
        val_story_union = {
            story for split in splits.values() for story in split.val_stories
        }
        test_story_union = {
            story for split in splits.values() for story in split.test_stories
        }
        self.assertFalse(train_story_union & val_story_union)
        self.assertFalse(train_story_union & test_story_union)
        self.assertFalse(val_story_union & test_story_union)


class WindowSubsetTests(unittest.TestCase):
    def test_duration_prefix_is_exact_and_ordered(self):
        target, selected = prefix_allocation([100] * 50, 0.05)
        self.assertEqual(target, 250)
        self.assertEqual(selected[:4], (100, 100, 50, 0))

    def test_expected_counts_and_nesting(self):
        expected = [50, 100, 200, 400, 500, 600, 800, 1000]
        actual = [len(nested_subset_indices(1000, scale, 1001)) for scale in SCALES]
        self.assertEqual(actual, expected)
        assert_nested_subsets(1000, 1001)

    def test_full_scale_keeps_original_order(self):
        self.assertEqual(nested_subset_indices(10, 1.0, 1001), tuple(range(10)))

    def test_prefix_allocation_and_collection_use_distinct_orders(self):
        lexical_items = (
            ("story-10", "/story-10.npy", None, 100),
            ("story-2", "/story-2.npy", None, 100),
        )
        target, allocation_plan, collection_plan = _allocate_sparkulee_prefix(
            lexical_items, 0.5
        )
        self.assertEqual(target, 100)
        self.assertEqual(
            [(item[0], item[4]) for item in allocation_plan],
            [("story-2", 100), ("story-10", 0)],
        )
        self.assertEqual(
            [(item[0], item[4]) for item in collection_plan],
            [("story-10", 0), ("story-2", 100)],
        )

    def test_full_prefix_collection_matches_baseline_order(self):
        lexical_items = (
            ("story-10", "/story-10.npy", None, 100),
            ("story-2", "/story-2.npy", None, 100),
        )
        _, _, collection_plan = _allocate_sparkulee_prefix(lexical_items, 1.0)
        self.assertEqual([item[0] for item in collection_plan], ["story-10", "story-2"])
        self.assertEqual([item[4] for item in collection_plan], [100, 100])


class MetricTests(unittest.TestCase):
    def test_macro_accuracy_is_mean_class_recall(self):
        targets = np.array([0, 0, 1, 1])
        predictions = np.array([0, 0, 0, 1])
        metrics = compute_metrics(targets, predictions)
        self.assertAlmostEqual(metrics["macro_acc"], 0.75)
        self.assertAlmostEqual(metrics["binary_acc"], 0.75)


if __name__ == "__main__":
    unittest.main()
