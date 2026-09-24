import itertools
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_shared_story_splits import (
    first_seeds_for_ordered_pairs,
    parse_subject_spec,
)


COMMON_STORIES = (
    "audiobook-2-1",
    "audiobook-2-2",
    "audiobook-3",
    "audiobook-5-1",
    "audiobook-5-3",
    "audiobook-6-1",
)


class SharedStorySearchTests(unittest.TestCase):
    def test_subject_spec_has_expected_23_subjects(self):
        subjects = parse_subject_spec("1-3,5-16,18-24,26")
        self.assertEqual(len(subjects), 23)
        self.assertEqual(subjects[0], "001")
        self.assertEqual(subjects[-1], "026")

    def test_seed_map_covers_every_ordered_pair(self):
        seed_map = first_seeds_for_ordered_pairs(COMMON_STORIES)
        self.assertEqual(set(seed_map), set(itertools.permutations(COMMON_STORIES, 2)))
        self.assertEqual(len(seed_map), 30)

    def test_seed_five_matches_current_shared_holdout(self):
        seed_map = first_seeds_for_ordered_pairs(COMMON_STORIES)
        self.assertEqual(
            seed_map[("audiobook-2-1", "audiobook-2-2")],
            5,
        )


if __name__ == "__main__":
    unittest.main()
