import unittest

from scaling.duration import prefix_allocation
from scaling.subset import (
    ALL_SUBJECTS,
    SCALES,
    assert_nested_subsets,
    build_story_split,
    nested_story_subset,
    normalize_scale,
    scale_tag,
)


class StoryScalingTest(unittest.TestCase):
    def setUp(self):
        self.stories = tuple(range(1, 61))
        self.split = build_story_split(self.stories, split_seed=5)

    def test_seed5_split_matches_baseline(self):
        self.assertEqual(self.split.val, (9, 10, 26, 32, 56))
        self.assertEqual(self.split.test, (36, 38, 44, 50, 55))
        self.assertEqual(len(self.split.train), 50)
        self.assertFalse(set(self.split.train) & set(self.split.val))
        self.assertFalse(set(self.split.train) & set(self.split.test))
        self.assertFalse(set(self.split.val) & set(self.split.test))

    def test_story_subsets_are_nested(self):
        assert_nested_subsets(self.split.train, subset_seed=1001)
        previous = set()
        expected_counts = (2, 5, 10, 20, 25, 30, 40, 50)
        for scale, expected_count in zip(SCALES, expected_counts):
            current = set(nested_story_subset(self.split.train, scale, 1001))
            self.assertEqual(len(current), expected_count)
            self.assertTrue(previous.issubset(current))
            self.assertTrue(current.issubset(set(self.split.train)))
            previous = current

    def test_subset_is_deterministic(self):
        first = nested_story_subset(self.split.train, 0.4, 1001)
        second = nested_story_subset(self.split.train, 0.4, 1001)
        different = nested_story_subset(self.split.train, 0.4, 1002)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_full_scale_preserves_complete_training_pool(self):
        self.assertEqual(
            nested_story_subset(self.split.train, 1.0, 1001),
            self.split.train,
        )

    def test_scale_validation_and_tags(self):
        self.assertEqual(normalize_scale(0.1), 0.10)
        self.assertEqual(scale_tag(0.05), "005")
        self.assertEqual(scale_tag(0.50), "050")
        self.assertEqual(scale_tag(1.0), "100")
        with self.assertRaises(ValueError):
            normalize_scale(0.3)

    def test_all_subjects_are_retained(self):
        self.assertEqual(ALL_SUBJECTS, tuple(f"{i:02d}" for i in range(1, 13)))

    def test_duration_prefix_is_exact_and_ordered(self):
        target, selected = prefix_allocation([100] * 50, 0.05)
        self.assertEqual(target, 250)
        self.assertEqual(selected[:4], (100, 100, 50, 0))


if __name__ == "__main__":
    unittest.main()
