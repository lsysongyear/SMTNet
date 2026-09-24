import copy
import importlib.util
import random
import unittest
from pathlib import Path


DATA_PATH = Path(
    "/gpfs/share/home/2201112028/lsycode/ICASSP_2027/"
    "VAD_SparKULee/SparKULee/preprocess/eeg_1-80Hz"
)
BASELINE_UTILS = Path(
    "/gpfs/share/home/2201112028/lsycode/ICASSP_2027/"
    "VAD_SparKULee/speech_code/utils.py"
)


@unittest.skipUnless(DATA_PATH.exists(), "SParKULee server data is unavailable")
class ServerDataIntegrationTests(unittest.TestCase):
    def test_pair_filters_and_smallest_subset(self):
        import yaml

        from scaling.subset import (
            ALL_SUBJECTS,
            build_subject_splits,
            discover_subject_stories,
            partition_pairs,
        )
        from speech_code.utils import get_datasets_from_config
        from train_scaling import (
            PROJECT_ROOT,
            _configure_partitions,
            _flatten_sample_metadata,
        )

        with (PROJECT_ROOT / "configs/scaling/config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        stories = discover_subject_stories(DATA_PATH)
        splits = build_subject_splits(stories, split_seed=5)
        _configure_partitions(config, splits, scale=0.05, protocol="shared_story")

        train_dataset, val_dataset, test_dataset, _ = get_datasets_from_config(config["data"])
        train_metadata = _flatten_sample_metadata(train_dataset)
        val_metadata = _flatten_sample_metadata(val_dataset)
        test_metadata = _flatten_sample_metadata(test_dataset)

        train_pairs = {(subject, story) for subject, story, _ in train_metadata}
        val_pairs = {(subject, story) for subject, story, _ in val_metadata}
        test_pairs = {(subject, story) for subject, story, _ in test_metadata}
        self.assertTrue(train_pairs <= set(partition_pairs(splits, "train")))
        self.assertEqual(
            val_pairs,
            set(partition_pairs(splits, "val")),
        )
        self.assertEqual(
            test_pairs,
            set(partition_pairs(splits, "test")),
        )

        train_story_union = {story for _, story in train_pairs}
        val_story_union = {story for _, story in val_pairs}
        test_story_union = {story for _, story in test_pairs}
        self.assertFalse(train_story_union & val_story_union)
        self.assertFalse(train_story_union & test_story_union)
        self.assertFalse(val_story_union & test_story_union)
        self.assertEqual({subject for subject, _ in train_pairs}, set(ALL_SUBJECTS))

    def test_full_scale_sample_order_matches_baseline(self):
        import yaml

        from scaling.subset import build_subject_splits, discover_subject_stories
        from speech_code.utils import get_datasets_from_config
        from train_scaling import (
            PROJECT_ROOT,
            _configure_partitions,
            _dataset_config,
            _flatten_sample_metadata,
        )

        with (PROJECT_ROOT / "configs/scaling/config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        stories = discover_subject_stories(DATA_PATH)
        splits = build_subject_splits(stories, split_seed=5)
        _configure_partitions(config, splits, scale=1.0, protocol="shared_story")

        random.seed(1)
        scaling_train, _, _, _ = get_datasets_from_config(config["data"])
        scaling_metadata = _flatten_sample_metadata(scaling_train)

        spec = importlib.util.spec_from_file_location("sparkulee_baseline_utils", BASELINE_UTILS)
        baseline_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_utils)
        baseline_config = copy.deepcopy(_dataset_config(config, "train"))
        baseline_config.pop("include_pairs", None)
        baseline_config.pop("recording_fraction", None)
        random.seed(1)
        baseline_train = baseline_utils.SparKULee_VAD(**baseline_config)
        baseline_metadata = [
            (subject.replace("sub-", "", 1), story, float(onset))
            for subject, story, _, onset, _ in baseline_train.samples
        ]

        self.assertEqual(len(scaling_metadata), 26384)
        self.assertEqual(scaling_metadata, baseline_metadata)


if __name__ == "__main__":
    unittest.main()
