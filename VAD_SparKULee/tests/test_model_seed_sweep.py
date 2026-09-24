import copy
import unittest

from model_seed_sweep import (
    SEED_KEY,
    configuration_invariant_hash,
    parse_int_spec,
    trained_config_invariant_hash,
)


class ModelSeedSweepTest(unittest.TestCase):
    def test_parse_seed_range(self):
        self.assertEqual(parse_int_spec("1-3,7"), (1, 2, 3, 7))

    def test_invariant_hash_ignores_only_seed_metadata_and_output(self):
        config = {
            "model": {"brain_magic_speech_v7": {"dropout": 0.1}},
            "optimizer": {"config": {"lr": 0.001}},
            "general": {
                "seed": 1,
                "output_path": "/tmp/seed1",
                "model_seed_sweep": {"model_seed": 1},
            },
        }
        space = {SEED_KEY: [1], "('optimizer', 'config', 'lr')": [0.001]}
        changed_seed = copy.deepcopy(config)
        changed_seed["general"].update({
            "seed": 20,
            "output_path": "/tmp/seed20",
            "model_seed_sweep": {"model_seed": 20},
        })
        changed_space = copy.deepcopy(space)
        changed_space[SEED_KEY] = [20]
        self.assertEqual(
            configuration_invariant_hash(config, space),
            configuration_invariant_hash(changed_seed, changed_space),
        )

        changed_fixed_split = copy.deepcopy(changed_seed)
        changed_fixed_split["general"]["model_seed_sweep"]["fixed_split_seed"] = 85
        self.assertNotEqual(
            configuration_invariant_hash(config, space),
            configuration_invariant_hash(changed_fixed_split, changed_space),
        )

        changed_lr = copy.deepcopy(changed_seed)
        changed_lr["optimizer"]["config"]["lr"] = 0.002
        self.assertNotEqual(
            configuration_invariant_hash(config, space),
            configuration_invariant_hash(changed_lr, changed_space),
        )

    def test_trained_config_hash_ignores_seed_paths_but_not_training_settings(self):
        config = {
            "model": {"brain_magic_speech_v7": {"dropout": 0.1}},
            "optimizer": {"config": {"lr": 0.001}},
            "general": {
                "seed": 1,
                "output_path": "/tmp/seed1",
                "checkpoint_path": "/tmp/seed1/checkpoints",
                "split_seed": 84,
                "split_story_audit": {
                    "train_stories": ["audiobook-1"],
                    "val_stories": ["audiobook-5-1"],
                    "test_stories": ["audiobook-5-3"],
                },
                "model_seed_sweep": {
                    "index": 0,
                    "model_seed": 1,
                    "fixed_split_seed": 84,
                },
            },
        }
        changed_seed = copy.deepcopy(config)
        changed_seed["general"].update({
            "seed": 20,
            "output_path": "/tmp/seed20",
            "checkpoint_path": "/tmp/seed20/checkpoints",
        })
        changed_seed["general"]["model_seed_sweep"].update({
            "index": 19,
            "model_seed": 20,
        })
        self.assertEqual(
            trained_config_invariant_hash(config),
            trained_config_invariant_hash(changed_seed),
        )

        changed_lr = copy.deepcopy(changed_seed)
        changed_lr["optimizer"]["config"]["lr"] = 0.002
        self.assertNotEqual(
            trained_config_invariant_hash(config),
            trained_config_invariant_hash(changed_lr),
        )


if __name__ == "__main__":
    unittest.main()
