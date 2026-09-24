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
from pnpl.datasets.libribrain2025.constants import RUN_KEYS
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scaling.evaluate import evaluate_run
from scaling.subset import build_run_split, normalize_scale, scale_tag
from speech_code.speech_utils import my_run_training
from speech_code.utils import adapt_config_to_data, get_datasets_from_config


DEFAULT_SPLIT_SEED = 46
BASELINE_MODEL_SEED = 42
BASELINE_TEST_MACRO_ACC = {
    42: 0.8899269700050354,
    43: 0.8770555853843689,
    44: 0.8857651948928833,
    45: 0.8792514204978943,
    46: 0.8941377997398376,
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
    return f"slurm_{slurm_id}" if slurm_id else datetime.now().strftime("local_%Y%m%d_%H%M%S")


def _dataset_config(config, partition):
    return next(iter(config["data"]["datasets"][partition][0].values()))


def _configure_fixed_split(config, split, scale):
    partitions = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }
    for partition, run_keys in partitions.items():
        dataset_config = _dataset_config(config, partition)
        dataset_config.pop("exclude_run_keys", None)
        dataset_config["include_run_keys"] = [list(run_key) for run_key in run_keys]
        dataset_config["data_type"] = partition
        dataset_config["recording_fraction"] = float(
            scale if partition == "train" else 1.0
        )


def _sample_metadata(dataset):
    if hasattr(dataset, "samples"):
        return [
            (tuple(str(value) for value in sample[:4]), float(sample[4]))
            for sample in dataset.samples
        ]
    if isinstance(dataset, ConcatDataset):
        result = []
        for child in dataset.datasets:
            result.extend(_sample_metadata(child))
        return result
    raise TypeError(f"Cannot extract sample metadata from {type(dataset).__name__}")


def _coverage_hours(metadata, window_seconds):
    intervals = defaultdict(list)
    for run_key, onset in metadata:
        intervals[run_key].append((onset, onset + window_seconds))

    covered_seconds = 0.0
    for run_intervals in intervals.values():
        run_intervals.sort()
        start, end = run_intervals[0]
        for next_start, next_end in run_intervals[1:]:
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


def _serialize_runs(run_keys):
    return [list(run_key) for run_key in run_keys]


def _baseline_reference(split_seed):
    macro_acc = BASELINE_TEST_MACRO_ACC.get(split_seed)
    if macro_acc is None:
        return None
    return {
        "source_run": (
            "VAD_MEG_NIPS brain_magic_speech_v7 "
            f"run{split_seed - 42}:seed{BASELINE_MODEL_SEED}"
        ),
        "split_seed": split_seed,
        "model_seed": BASELINE_MODEL_SEED,
        "test_macro_acc": macro_acc,
    }


def _manifest(args, split, run_dir):
    return {
        "experiment": "meg_nips_train_only_duration_scaling",
        "created_at_utc": _utc_now(),
        "run_dir": str(Path(run_dir).resolve()),
        "model": "brain_magic_speech_v7",
        "protocol": "protocol_duration_prefix_train_only",
        "scale": args.scale,
        "scale_percent": int(round(args.scale * 100)),
        "scale_unit": "chronological effective recording duration",
        "scaled_partitions": ["train"],
        "train_fraction": args.scale,
        "val_fraction": 1.0,
        "test_fraction": 1.0,
        "split_seed": args.split_seed,
        "subset_seed": args.subset_seed,
        "subset_seed_role": "deprecated compatibility field; no effect on prefix selection",
        "model_seed": args.model_seed,
        "full_train_run_count": len(split.train),
        "train_run_keys": _serialize_runs(split.train),
        "val_run_keys": _serialize_runs(split.val),
        "test_run_keys": _serialize_runs(split.test),
        "selection_protocol": "top-5 checkpoint and threshold by val_macro_acc",
        "primary_metric": "macro_acc",
        "baseline_reference": _baseline_reference(args.split_seed),
    }


def main(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if list(config["model"]) != ["brain_magic_speech_v7"]:
        raise ValueError("Scaling config must contain only brain_magic_speech_v7")
    args.scale = normalize_scale(args.scale)
    split = build_run_split(RUN_KEYS, args.split_seed)
    _configure_fixed_split(config, split, args.scale)

    plan = _manifest(args, split, PROJECT_ROOT)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    torch.set_float32_matmul_precision("high")
    lightning.seed_everything(args.model_seed, workers=True)
    config["general"]["seed"] = args.model_seed
    config["general"]["split_seed"] = args.split_seed
    if args.max_epochs is not None:
        config["trainer"]["max_epochs"] = args.max_epochs

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
        "full_train_run_keys": _serialize_runs(split.train),
        "train_run_order": _serialize_runs(split.train),
        "val_run_keys": _serialize_runs(split.val),
        "test_run_keys": _serialize_runs(split.test),
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    shutil.copy2(
        PROJECT_ROOT / "speech_code/models/my_modules/BrainNetwork_v7.py",
        run_dir / "model.py",
    )
    manifest = _manifest(args, split, run_dir)
    _write_json(run_dir / "scaling_manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "loading_data", "updated_at_utc": _utc_now()})

    train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(config["data"])
    metadata = _sample_metadata(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=True, **config["data"]["dataloader"])
    val_loader = DataLoader(val_dataset, shuffle=False, **config["data"]["dataloader"])
    adapt_config_to_data(config, train_loader, labels)

    window_seconds = float(_dataset_config(config, "train")["tmax"]) - float(
        _dataset_config(config, "train").get("tmin", 0.0)
    )
    manifest.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "train_recording": _duration_summary(train_dataset),
            "val_recording": _duration_summary(val_dataset),
            "train_window_hours_with_overlap": len(train_dataset) * window_seconds / 3600.0,
            "unique_selected_meg_coverage_hours": _coverage_hours(metadata, window_seconds),
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
    summary = evaluate_run(
        run_dir,
        config,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        labels=labels,
    )
    macro_acc = summary["metrics"]["macro_acc"]["mean"]
    status = {
        "status": "completed",
        "updated_at_utc": _utc_now(),
        "primary_metric": "macro_acc",
        "macro_acc": macro_acc,
    }
    reference = _baseline_reference(args.split_seed)
    if args.scale == 1.0 and args.model_seed == BASELINE_MODEL_SEED and reference:
        status["baseline_delta"] = macro_acc - reference["test_macro_acc"]
    _write_json(run_dir / "status.json", status)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scaling/config.yaml")
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--subset-seed", type=int, default=1001)
    parser.add_argument("--model-seed", type=int, default=BASELINE_MODEL_SEED)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
