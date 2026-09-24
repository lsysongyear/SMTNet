import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytorch_lightning as lightning
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scaling.evaluate import evaluate_run
from scaling.subset import (
    ALL_SUBJECTS,
    build_story_split,
    normalize_scale,
    scale_tag,
)
from speech_code.speech_utils import my_run_training
from speech_code.train_v1 import infer_split_items
from speech_code.utils import adapt_config_to_data, get_datasets_from_config


EXPECTED_MODEL_CONFIG = {
    "input_dim": 204,
    "attention_dim": 128,
    "output_dim": 1,
    "kernel_size": 40,
    "depthwise_kernel": 30,
    "n_subjects": 12,
    "num_blocks1": 8,
    "num_blocks2": 19,
    "dropout": 0.1,
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _safe_run_id(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not value:
        raise ValueError("run_id cannot be empty")
    return value


def _default_run_id():
    slurm_id = os.getenv("SLURM_JOB_ID")
    if slurm_id:
        return f"slurm_{slurm_id}"
    return datetime.now().strftime("local_%Y%m%d_%H%M%S")


def _dataset_config(config, partition):
    return next(iter(config["data"]["datasets"][partition][0].values()))


def _configure_partitions(config, story_split, scale):
    partition_stories = {
        "train": story_split.train,
        "val": story_split.val,
        "test": story_split.test,
    }
    for partition, stories in partition_stories.items():
        dataset_config = _dataset_config(config, partition)
        dataset_config["include_subjects"] = list(ALL_SUBJECTS)
        dataset_config["stories"] = list(stories)
        dataset_config["data_type"] = partition
        dataset_config["recording_fraction"] = float(
            scale if partition == "train" else 1.0
        )


def _flatten_sample_metadata(dataset):
    if hasattr(dataset, "samples"):
        return [
            (
                str(sample[0]).replace("sub-", "", 1),
                int(sample[1]),
                float(sample[3]),
            )
            for sample in dataset.samples
        ]
    if isinstance(dataset, ConcatDataset):
        result = []
        for child in dataset.datasets:
            result.extend(_flatten_sample_metadata(child))
        return result
    raise TypeError(f"Cannot extract sample metadata from {type(dataset).__name__}")


def _coverage_hours(metadata, window_seconds):
    intervals = defaultdict(list)
    for subject, story, onset in metadata:
        intervals[(subject, story)].append((onset, onset + window_seconds))

    covered_seconds = 0.0
    for pair_intervals in intervals.values():
        pair_intervals.sort()
        start, end = pair_intervals[0]
        for next_start, next_end in pair_intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered_seconds += end - start
                start, end = next_start, next_end
        covered_seconds += end - start
    return covered_seconds / 3600.0


def _duration_summary(dataset):
    children = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    metadata = [child.scaling_metadata for child in children]
    sfreqs = {float(item["duration_sfreq"]) for item in metadata}
    if len(sfreqs) != 1:
        raise RuntimeError(f"Inconsistent duration sample rates: {sorted(sfreqs)}")
    sfreq = sfreqs.pop()
    subjects = {}
    for item in metadata:
        subjects.update(item["subjects"])
    full_samples = sum(item["full_samples"] for item in subjects.values())
    selected_samples = sum(item["selected_samples"] for item in subjects.values())
    return {
        "duration_sfreq": sfreq,
        "full_hours": full_samples / sfreq / 3600.0,
        "selected_hours": selected_samples / sfreq / 3600.0,
        "actual_fraction": selected_samples / full_samples,
        "subjects": subjects,
    }


def _serialize_split(story_split):
    return {
        "train": list(story_split.train),
        "val": list(story_split.val),
        "test": list(story_split.test),
    }


def _build_manifest(args, subjects, stories, story_split, run_dir):
    return {
        "experiment": "sem4lang_train_only_duration_scaling",
        "created_at_utc": _utc_now(),
        "run_dir": str(Path(run_dir).resolve()),
        "model": "brain_magic_speech_v7",
        "protocol": "protocol_duration_prefix_train_only",
        "model_config": EXPECTED_MODEL_CONFIG,
        "scale": args.scale,
        "scale_percent": int(round(args.scale * 100)),
        "scale_unit": "per-subject chronological effective recording duration",
        "scaled_partitions": ["train"],
        "train_fraction": args.scale,
        "val_fraction": 1.0,
        "test_fraction": 1.0,
        "split_strategy": "per_sub_5t5v",
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "subset_seed_role": "deprecated compatibility field; no effect on prefix selection",
        "model_seed": args.model_seed,
        "subjects": list(subjects),
        "n_subjects": len(subjects),
        "all_story_count": len(stories),
        "full_train_story_count": len(story_split.train),
        "train_story_order": list(story_split.train),
        "story_split": _serialize_split(story_split),
        "train_subject_story_pair_count": len(subjects) * len(story_split.train),
        "val_subject_story_pair_count": len(subjects) * len(story_split.val),
        "test_subject_story_pair_count": len(subjects) * len(story_split.test),
        "selection_protocol": "per-subject top-5 checkpoint and threshold by val_macro_acc",
        "primary_metric": "macro_acc",
        "leakage_check": "train stories are disjoint from validation/test stories",
    }


def main(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if list(config["model"]) != ["brain_magic_speech_v7"]:
        raise ValueError("Scaling config must contain only brain_magic_speech_v7")
    if config["model"]["brain_magic_speech_v7"] != EXPECTED_MODEL_CONFIG:
        raise ValueError("brain_magic_speech_v7 parameters do not match the canonical 8/19 model")

    args.scale = normalize_scale(args.scale)
    subjects, stories = infer_split_items(config)
    subjects = tuple(str(subject).replace("sub-", "").zfill(2) for subject in subjects)
    stories = tuple(int(story) for story in stories)
    if subjects != ALL_SUBJECTS:
        raise ValueError(f"Expected subjects {ALL_SUBJECTS}, found {subjects}")

    story_split = build_story_split(stories, args.split_seed)
    if set(story_split.train) & (set(story_split.val) | set(story_split.test)):
        raise RuntimeError("Training stories overlap validation/test stories")
    _configure_partitions(config, story_split, args.scale)

    preview = _build_manifest(args, subjects, stories, story_split, PROJECT_ROOT)
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    torch.set_float32_matmul_precision("high")
    lightning.seed_everything(args.model_seed, workers=True)
    config["general"]["seed"] = args.model_seed
    output_root = Path(config["general"]["output_path"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_id = _safe_run_id(args.run_id or _default_run_id())
    run_dir = (
        output_root
        / "brain_magic_speech_v7"
        / "protocol_duration_prefix_train_only"
        / f"split_seed_{args.split_seed}"
        / f"subset_seed_{args.subset_seed}"
        / f"model_seed_{args.model_seed}"
        / f"scale_{scale_tag(args.scale)}"
        / f"run_{run_id}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    config["general"]["run_dir"] = str(run_dir)
    config["general"]["checkpoint_path"] = str(checkpoint_dir)
    config["general"]["scaling"] = {
        "protocol": "protocol_duration_prefix_train_only",
        "scale": args.scale,
        "train_fraction": args.scale,
        "val_fraction": 1.0,
        "test_fraction": 1.0,
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "model_seed": args.model_seed,
        "story_split": _serialize_split(story_split),
        "train_story_order": list(story_split.train),
    }
    if args.max_epochs is not None:
        config["trainer"]["max_epochs"] = args.max_epochs

    shutil.copy2(
        PROJECT_ROOT / "speech_code/models/my_modules/BrainNetwork_v7.py",
        run_dir / "model.py",
    )
    manifest = _build_manifest(args, subjects, stories, story_split, run_dir)
    _write_json(run_dir / "scaling_manifest.json", manifest)
    _write_json(
        run_dir / "status.json",
        {"status": "loading_data", "updated_at_utc": _utc_now()},
    )

    train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(
        config["data"]
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, **config["data"]["dataloader"]
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **config["data"]["dataloader"]
    )
    adapt_config_to_data(config, train_loader, labels)

    metadata = _flatten_sample_metadata(train_dataset)
    if len(metadata) != len(train_dataset):
        raise RuntimeError("Training sample metadata length mismatch")
    window_seconds = float(
        _dataset_config(config, "train")["tmax"]
        - _dataset_config(config, "train").get("tmin", 0.0)
    )
    manifest.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "train_recording": _duration_summary(train_dataset),
            "val_recording": _duration_summary(val_dataset),
            "train_window_hours_with_overlap": len(train_dataset)
            * window_seconds
            / 3600.0,
            "unique_selected_meg_coverage_hours": _coverage_hours(
                metadata, window_seconds
            ),
        }
    )
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    _write_json(run_dir / "scaling_manifest.json", manifest)
    _write_json(
        run_dir / "status.json",
        {"status": "training", "updated_at_utc": _utc_now()},
    )

    trainer, module = my_run_training(
        train_loader,
        val_loader,
        config,
        len(labels),
        best_model_metric="val_loss",
        best_model_metric_mode="min",
    )
    if not trainer.is_global_zero:
        return
    del module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _write_json(
        run_dir / "status.json",
        {"status": "evaluating", "updated_at_utc": _utc_now()},
    )
    summary = evaluate_run(run_dir, config, story_split)
    _write_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "updated_at_utc": _utc_now(),
            "primary_metric": "macro_acc",
            "macro_acc": summary["metrics"]["macro_acc"],
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scaling/config.yaml")
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--split-seed", type=int, default=5)
    parser.add_argument("--subset-seed", type=int, default=1001)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
