import random
import unittest

from scaling.duration import canonical_prefix_allocation, prefix_allocation
from scaling.subset import (
    SCALES,
    assert_nested_subsets,
    build_run_split,
    nested_run_subset,
)


def _run_keys(count=90):
    return [("0", str(index), "story", "1") for index in range(count)]


class RunSplitTests(unittest.TestCase):
    def test_meg_split_has_original_partition_sizes(self):
        split = build_run_split(_run_keys(), split_seed=46)
        self.assertEqual(len(split.train), 72)
        self.assertEqual(len(split.val), 9)
        self.assertEqual(len(split.test), 9)

    def test_split_is_deterministic_and_disjoint(self):
        first = build_run_split(_run_keys(), split_seed=46)
        second = build_run_split(_run_keys(), split_seed=46)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train) & set(first.val))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.val) & set(first.test))

    def test_split_preserves_baseline_shuffled_slice_order(self):
        run_keys = _run_keys()
        shuffled = run_keys.copy()
        random.Random(46).shuffle(shuffled)
        heldout_count = len(shuffled) // 10

        split = build_run_split(run_keys, split_seed=46)

        self.assertEqual(split.test, tuple(shuffled[:heldout_count]))
        self.assertEqual(
            split.val, tuple(shuffled[heldout_count : 2 * heldout_count])
        )
        self.assertEqual(split.train, tuple(shuffled[2 * heldout_count :]))


class NestedRunSubsetTests(unittest.TestCase):
    def test_duration_prefix_is_exact_and_ordered(self):
        target, selected = prefix_allocation([100] * 50, 0.05)
        self.assertEqual(target, 250)
        self.assertEqual(selected[:4], (100, 100, 50, 0))

    def setUp(self):
        self.train_runs = build_run_split(_run_keys(), split_seed=46).train

    def test_expected_counts_and_nesting(self):
        expected = [4, 7, 14, 29, 36, 43, 58, 72]
        actual = [
            len(nested_run_subset(self.train_runs, scale, 1001)) for scale in SCALES
        ]
        self.assertEqual(actual, expected)
        assert_nested_subsets(self.train_runs, 1001)

    def test_full_scale_preserves_original_order(self):
        self.assertEqual(nested_run_subset(self.train_runs, 1.0, 1001), self.train_runs)


class CanonicalPrefixAllocationTests(unittest.TestCase):
    def setUp(self):
        self.a = ("0", "a", "story", "1")
        self.b = ("0", "b", "story", "1")
        self.c = ("0", "c", "story", "1")
        self.canonical_order = (self.a, self.b, self.c)
        self.collection_order = (self.c, self.a, self.b)
        self.keyed_lengths = ((self.c, 300), (self.a, 100), (self.b, 200))

    def test_partial_scale_allocates_canonical_prefix(self):
        target, allocation, allocation_order = canonical_prefix_allocation(
            self.keyed_lengths, self.canonical_order, 0.5
        )

        self.assertEqual(target, 300)
        self.assertEqual(allocation_order, self.canonical_order)
        self.assertEqual(allocation, {self.a: 100, self.b: 200, self.c: 0})
        self.assertEqual(
            [allocation[key] for key in self.collection_order], [0, 100, 200]
        )

    def test_full_scale_keeps_all_runs_in_collection_order(self):
        target, allocation, _ = canonical_prefix_allocation(
            self.keyed_lengths, self.canonical_order, 1.0
        )

        self.assertEqual(target, 600)
        self.assertEqual(
            [allocation[key] for key in self.collection_order], [300, 100, 200]
        )


if __name__ == "__main__":
    unittest.main()
