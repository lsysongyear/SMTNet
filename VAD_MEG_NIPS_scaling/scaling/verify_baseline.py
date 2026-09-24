import argparse
import json
import sys
from pathlib import Path

import yaml
from pnpl.datasets.libribrain2025.constants import RUN_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scaling.subset import build_run_split


EXPECTED_MODEL = {
    "input_dim": 204,
    "attention_dim": 128,
    "output_dim": 1,
    "kernel_size": 40,
    "depthwise_kernel": 30,
    "num_blocks1": 8,
    "num_blocks2": 19,
    "dropout": 0.1,
}


def _dataset_config(config, partition):
    return next(iter(config["data"]["datasets"][partition][0].values()))


def _normalized_run_keys(values):
    return tuple(tuple(str(value) for value in run_key) for run_key in values)


def verify(old_config_path, scaling_config_path):
    with Path(old_config_path).open(encoding="utf-8") as handle:
        old = yaml.safe_load(handle)
    with Path(scaling_config_path).open(encoding="utf-8") as handle:
        scaling = yaml.safe_load(handle)

    split_seed = int(old["general"]["split_seed"])
    model_seed = int(old["general"]["seed"])
    split = build_run_split(RUN_KEYS, split_seed)
    expected_partitions = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }
    for partition, expected in expected_partitions.items():
        actual = _normalized_run_keys(
            _dataset_config(old, partition)["include_run_keys"]
        )
        if actual != expected:
            raise AssertionError(f"{partition} run keys differ from the old baseline")

    old_model = old["model"]["brain_magic_speech_v7"]
    scaling_model = scaling["model"]["brain_magic_speech_v7"]
    if old_model != EXPECTED_MODEL or scaling_model != EXPECTED_MODEL:
        raise AssertionError("brain_magic_speech_v7 hyperparameters do not match")
    if split_seed not in range(42, 47):
        raise AssertionError(f"Unexpected old split seed: {split_seed}")
    if scaling["general"]["seed"] != model_seed:
        raise AssertionError("model seed differs from the old baseline")

    result = {
        "status": "compatible",
        "old_config": str(Path(old_config_path).resolve()),
        "split_seed": split_seed,
        "model_seed": model_seed,
        "train_runs": len(split.train),
        "val_runs": len(split.val),
        "test_runs": len(split.test),
        "model": EXPECTED_MODEL,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-config", required=True)
    parser.add_argument(
        "--scaling-config",
        default=str(PROJECT_ROOT / "configs/scaling/config.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.old_config, args.scaling_config), indent=2))


if __name__ == "__main__":
    main()
