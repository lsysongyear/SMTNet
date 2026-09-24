import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytorch_lightning as lightning
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scaling.evaluate import evaluate_run
from scaling.subset import ALL_SUBJECTS, assert_nested_subsets, build_scaling_split, scale_tag
from speech_code.speech_utils import my_run_training
from speech_code.utils import adapt_config_to_data, get_datasets_from_config


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


def _set_partition(config, partition, subjects, stories, recording_fraction):
    for dataset_entry in config["data"]["datasets"][partition]:
        dataset_config = next(iter(dataset_entry.values()))
        dataset_config["include_subjects"] = list(subjects)
        dataset_config["stories"] = list(stories)
        dataset_config["data_type"] = partition
        dataset_config["recording_fraction"] = float(recording_fraction)


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


def _build_manifest(args, split, run_dir):
    return {
        "experiment": "pkueeg_train_only_duration_scaling",
        "created_at_utc": _utc_now(),
        "run_dir": str(run_dir.resolve()),
        "model": "brain_magic_speech_v7",
        "protocol": "protocol_duration_prefix_train_only",
        "scale": split.scale,
        "scale_percent": int(round(split.scale * 100)),
        "scale_unit": "per-subject chronological effective recording duration",
        "scaled_partitions": ["train"],
        "train_fraction": split.scale,
        "val_fraction": 1.0,
        "test_fraction": 1.0,
        "train_story_count": len(split.train_stories),
        "full_train_story_count": len(split.train_pool),
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "subset_seed_role": "deprecated compatibility field; no effect on prefix selection",
        "model_seed": args.model_seed,
        "subjects": list(ALL_SUBJECTS),
        "train_story_pool": list(split.train_pool),
        "train_story_order": list(split.permutation),
        "train_stories": list(split.train_stories),
        "val_stories": list(split.val_stories),
        "test_stories": list(split.test_stories),
        "selection_protocol": "per-subject top-5 checkpoint and threshold by val_macro_acc",
        "primary_metric": "macro_acc",
    }


def main(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if list(config["model"]) != ["brain_magic_speech_v7"]:
        raise ValueError("Scaling config must contain only brain_magic_speech_v7")

    assert_nested_subsets(args.subset_seed, args.split_seed)
    split = build_scaling_split(args.scale, args.subset_seed, args.split_seed)
    _set_partition(config, "train", ALL_SUBJECTS, split.train_stories, split.scale)
    _set_partition(config, "val", ALL_SUBJECTS, split.val_stories, 1.0)
    _set_partition(config, "test", ALL_SUBJECTS, split.test_stories, 1.0)

    plan = _build_manifest(args, split, PROJECT_ROOT)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
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
        / f"scale_{scale_tag(split.scale)}"
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
        "scale": split.scale,
        "train_fraction": split.scale,
        "val_fraction": 1.0,
        "test_fraction": 1.0,
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "model_seed": args.model_seed,
    }
    if args.max_epochs is not None:
        config["trainer"]["max_epochs"] = args.max_epochs

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    shutil.copy2(
        PROJECT_ROOT / "speech_code/models/my_modules/BrainNetwork_v7.py",
        run_dir / "model.py",
    )

    manifest = _build_manifest(args, split, run_dir)
    _write_json(run_dir / "scaling_manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "loading_data", "updated_at_utc": _utc_now()})

    train_dataset, val_dataset, _, labels = get_datasets_from_config(config["data"])
    train_loader = DataLoader(
        train_dataset, shuffle=True, **config["data"]["dataloader"]
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **config["data"]["dataloader"])
    adapt_config_to_data(config, train_loader, labels)

    manifest.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_recording": _duration_summary(train_dataset),
            "val_recording": _duration_summary(val_dataset),
            "train_window_hours_with_overlap": len(train_dataset) * 12.0 / 3600.0,
        }
    )
    _write_json(run_dir / "scaling_manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "training", "updated_at_utc": _utc_now()})

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

    _write_json(run_dir / "status.json", {"status": "evaluating", "updated_at_utc": _utc_now()})
    summary = evaluate_run(run_dir, config, list(ALL_SUBJECTS))
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
    parser.add_argument("--split-seed", type=int, default=545)
    parser.add_argument("--subset-seed", type=int, default=1001)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
