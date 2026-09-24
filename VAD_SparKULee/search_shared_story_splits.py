#!/usr/bin/env python3
"""Prepare and aggregate exhaustive shared-story split searches."""

import argparse
import csv
import itertools
import json
import math
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "brain_magic_speech_v7"
DEFAULT_SUBJECTS = "1-3,5-16,18-24,26"


def parse_subject_spec(spec, width=3):
    subjects = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            subjects.update(range(int(start), int(end) + 1))
        else:
            subjects.add(int(token))
    return tuple(f"{subject:0{width}d}" for subject in sorted(subjects))


def _dataset_config(config, partition="train"):
    return next(iter(config["data"]["datasets"][partition][0].values()))


def discover_subject_stories(config, subjects):
    dataset = _dataset_config(config)
    data_path = Path(dataset["data_path"])
    preproc = dataset.get("preproc", "BP")
    fs_tag = dataset.get("eeg_fs_tag", f"{int(dataset.get('sfreq', 250))}Hz")
    selected = set(subjects)
    subject_stories = {subject: set() for subject in subjects}

    for path in sorted(data_path.iterdir()):
        if path.suffix != ".npy":
            continue
        parts = path.stem.split("_")
        if len(parts) < 4 or parts[-2] != preproc or parts[-1] != fs_tag:
            continue
        subject = parts[0].replace("sub-", "", 1)
        audio_key = "_".join(parts[1:-2])
        if subject not in selected or not audio_key.startswith("audio-"):
            continue
        subject_stories[subject].add(audio_key.replace("audio-", "", 1))

    missing = [subject for subject, stories in subject_stories.items() if not stories]
    if missing:
        raise RuntimeError(f"No matching data found for subjects: {missing}")
    return {
        subject: tuple(sorted(stories))
        for subject, stories in subject_stories.items()
    }


def common_stories(subject_stories):
    shared = set.intersection(*(set(stories) for stories in subject_stories.values()))
    if len(shared) < 3:
        raise RuntimeError(
            f"Need at least three common stories, found {len(shared)}: {sorted(shared)}"
        )
    return tuple(sorted(shared))


def first_seeds_for_ordered_pairs(stories, max_seed=100000):
    """Match DataSplitter: shuffle, then test=position 0 and val=position 1."""
    stories = tuple(sorted(stories))
    required = len(stories) * (len(stories) - 1)
    found = {}
    for seed in range(max_seed):
        shuffled = list(stories)
        random.Random(seed).shuffle(shuffled)
        pair = (shuffled[1], shuffled[0])  # (validation, test)
        found.setdefault(pair, seed)
        if len(found) == required:
            break
    if len(found) != required:
        missing = set(itertools.permutations(stories, 2)) - set(found)
        raise RuntimeError(f"Could not cover all ordered pairs: {sorted(missing)}")
    return found


def verify_seed_map(subject_stories, stories, seed_map):
    from speech_code.data_split import DataSplitter

    subjects = tuple(subject_stories)
    audits = {}
    for pair, seed in seed_map.items():
        expected_val, expected_test = pair
        splitter = DataSplitter(
            subjects,
            stories,
            n_folds=5,
            seed=seed,
            subject_stories_map=subject_stories,
        )
        splits = splitter.get_split("per_sub", 0)
        train_union = {
            story for split in splits.values() for story in split["train_stories"]
        }
        val_union = {
            story for split in splits.values() for story in split["val_stories"]
        }
        test_union = {
            story for split in splits.values() for story in split["test_stories"]
        }
        if val_union != {expected_val} or test_union != {expected_test}:
            raise RuntimeError(
                f"Seed {seed} produced val={sorted(val_union)}, test={sorted(test_union)}; "
                f"expected val={expected_val}, test={expected_test}"
            )
        overlap = (train_union & val_union) | (train_union & test_union) | (
            val_union & test_union
        )
        if overlap:
            raise RuntimeError(f"Seed {seed} has global story leakage: {sorted(overlap)}")
        audits[pair] = {
            "train_story_union": sorted(train_union),
            "validation_story_union": sorted(val_union),
            "test_story_union": sorted(test_union),
            "overlap_story_count": 0,
        }
    return audits


def _slug(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _fixed_search_space(model_config, config):
    search_space = {}
    for key, value in model_config.items():
        search_space[repr(("model", MODEL_NAME, key))] = [value]
    search_space[repr(("general", "seed"))] = [1]
    search_space[repr(("loss", "config", "weight"))] = [
        list(config["loss"]["config"]["weight"])
    ]
    search_space[repr(("general", "loss_type"))] = [
        config["general"].get("loss_type", "mse")
    ]
    search_space[repr(("optimizer", "config", "lr"))] = [
        config["optimizer"]["config"]["lr"]
    ]
    return search_space


def prepare(args):
    base_config_path = Path(args.base_config).resolve()
    model_sweep_path = Path(args.model_sweep).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot directory already exists: {snapshot_dir}")

    with base_config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    with model_sweep_path.open(encoding="utf-8") as handle:
        model_sweep = yaml.safe_load(handle)
    if MODEL_NAME not in model_sweep:
        raise KeyError(f"{MODEL_NAME} is missing from {model_sweep_path}")

    subjects = parse_subject_spec(args.subjects)
    subject_stories = discover_subject_stories(base_config, subjects)
    shared = common_stories(subject_stories)
    seed_map = first_seeds_for_ordered_pairs(shared)
    split_audits = verify_seed_map(subject_stories, shared, seed_map)
    ordered_pairs = list(itertools.permutations(shared, 2))

    snapshot_dir.mkdir(parents=True)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = []
    tsv_rows = []
    for index, (val_story, test_story) in enumerate(ordered_pairs):
        split_seed = seed_map[(val_story, test_story)]
        candidate_name = (
            f"candidate_{index:02d}_val_{_slug(val_story)}__test_{_slug(test_story)}"
        )
        candidate_snapshot = snapshot_dir / candidate_name
        candidate_output = output_root / candidate_name
        candidate_snapshot.mkdir()

        config = deepcopy(base_config)
        model_config = deepcopy(model_sweep[MODEL_NAME])
        config["model"] = {MODEL_NAME: model_config}
        config["general"]["model_name"] = MODEL_NAME
        config["general"]["output_path"] = str(candidate_output)
        config["general"]["split_protocol"] = "global_shared_story_search"
        config["general"]["search_candidate"] = {
            "index": index,
            "validation_story": val_story,
            "test_story": test_story,
            "split_seed": split_seed,
        }

        config_path = candidate_snapshot / "config.yaml"
        search_space_path = candidate_snapshot / "search-space.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        with search_space_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                _fixed_search_space(model_config, config), handle, sort_keys=False
            )

        candidate = {
            "index": index,
            "split_seed": split_seed,
            "validation_story": val_story,
            "test_story": test_story,
            "train_common_stories": [
                story for story in shared if story not in {val_story, test_story}
            ],
            "split_audit": split_audits[(val_story, test_story)],
            "config_path": str(config_path),
            "search_space_path": str(search_space_path),
            "output_path": str(candidate_output),
        }
        candidates.append(candidate)
        tsv_rows.append(
            [
                index,
                split_seed,
                val_story,
                test_story,
                config_path,
                search_space_path,
                candidate_output,
            ]
        )

    manifest = {
        "protocol": "exhaustive_ordered_shared_story_holdout_search",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": list(subjects),
        "n_subjects": len(subjects),
        "common_stories": list(shared),
        "n_candidates": len(candidates),
        "model": MODEL_NAME,
        "model_config": model_sweep[MODEL_NAME],
        "output_root": str(output_root),
        "selection_warning": (
            "Every common story is used during the exhaustive search. Ranking by test "
            "Macro_Acc is exploratory and is not an unbiased final-test estimate."
        ),
        "candidates": candidates,
    }
    manifest_path = snapshot_dir / "manifest.json"
    candidates_path = snapshot_dir / "candidates.tsv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with candidates_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(tsv_rows)

    print(json.dumps({
        "manifest": str(manifest_path),
        "candidates_tsv": str(candidates_path),
        "output_root": str(output_root),
        "n_candidates": len(candidates),
        "common_stories": list(shared),
    }, indent=2))


def _mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def aggregate(args):
    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows = []
    for candidate in manifest["candidates"]:
        output_path = Path(candidate["output_path"])
        summaries = list(output_path.rglob("pretrain_eval_summary.json"))
        row = dict(candidate)
        if not summaries:
            row["status"] = "missing"
            rows.append(row)
            continue

        summary_path = max(summaries, key=lambda path: path.stat().st_mtime)
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        logs = sorted(summary_path.parent.glob("sub-*/pretrain_eval/test_log.json"))
        val_scores = []
        test_scores = []
        for path in logs:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            val_scores.append(float(value["val_macro_acc"]))
            test_scores.append(float(value["macro_acc"]))

        if len(val_scores) != manifest["n_subjects"]:
            row.update({
                "status": "incomplete",
                "summary_path": str(summary_path),
                "completed_subjects": len(val_scores),
            })
            rows.append(row)
            continue

        val_mean, val_std = _mean_std(val_scores)
        test_mean, test_std = _mean_std(test_scores)
        summary_mean = float(summary["macro_acc"]["mean"])
        summary_std = float(summary["macro_acc"]["std"])
        if not (
            math.isclose(test_mean, summary_mean, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(test_std, summary_std, rel_tol=0.0, abs_tol=1e-12)
        ):
            row.update({
                "status": "invalid_summary",
                "summary_path": str(summary_path),
                "completed_subjects": len(val_scores),
            })
            rows.append(row)
            continue

        trained_configs = list(summary_path.parent.glob("pretrain/run*/config.yaml"))
        if not trained_configs:
            row.update({
                "status": "missing_trained_config",
                "summary_path": str(summary_path),
                "completed_subjects": len(val_scores),
            })
            rows.append(row)
            continue
        trained_config_path = max(
            trained_configs, key=lambda path: path.stat().st_mtime
        )
        with trained_config_path.open(encoding="utf-8") as handle:
            trained_config = yaml.safe_load(handle)
        audit = trained_config["general"].get("split_story_audit", {})
        train_union = set(audit.get("train_stories", []))
        val_union = set(audit.get("val_stories", []))
        test_union = set(audit.get("test_stories", []))
        overlap = (train_union & val_union) | (train_union & test_union) | (
            val_union & test_union
        )
        if (
            val_union != {candidate["validation_story"]}
            or test_union != {candidate["test_story"]}
            or overlap
        ):
            row.update({
                "status": "invalid_split",
                "summary_path": str(summary_path),
                "trained_config_path": str(trained_config_path),
                "completed_subjects": len(val_scores),
            })
            rows.append(row)
            continue

        row.update({
            "status": "completed",
            "summary_path": str(summary_path),
            "trained_config_path": str(trained_config_path),
            "completed_subjects": len(val_scores),
            "val_macro_acc_mean": val_mean,
            "val_macro_acc_std": val_std,
            "test_macro_acc_mean": test_mean,
            "test_macro_acc_std": test_std,
        })
        rows.append(row)

    completed = [row for row in rows if row["status"] == "completed"]
    by_validation = sorted(
        completed, key=lambda row: row["val_macro_acc_mean"], reverse=True
    )
    by_test = sorted(
        completed, key=lambda row: row["test_macro_acc_mean"], reverse=True
    )
    report = {
        "protocol": manifest["protocol"],
        "manifest": str(manifest_path),
        "n_candidates": manifest["n_candidates"],
        "n_completed": len(completed),
        "primary_ranking": "validation_macro_acc_mean",
        "test_ranking_policy": "exploratory_only_due_to_test_selection_bias",
        "best_by_validation": by_validation[0] if by_validation else None,
        "best_by_test_exploratory": by_test[0] if by_test else None,
        "validation_ranking": by_validation,
        "test_ranking_exploratory": by_test,
        "all_candidates": rows,
    }

    output_root = Path(manifest["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "shared_story_search_summary.json"
    csv_path = output_root / "shared_story_search_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    fieldnames = [
        "index", "split_seed", "validation_story", "test_story", "status",
        "completed_subjects", "val_macro_acc_mean", "val_macro_acc_std",
        "test_macro_acc_mean", "test_macro_acc_std", "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "n_completed": len(completed),
        "best_by_validation": report["best_by_validation"],
        "best_by_test_exploratory": report["best_by_test_exploratory"],
    }, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--base-config", default="configs/speech/my_run/config.yaml"
    )
    prepare_parser.add_argument(
        "--model-sweep", default="configs/speech/my_run/model_sweep.yaml"
    )
    prepare_parser.add_argument("--snapshot-dir", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    prepare_parser.set_defaults(func=prepare)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.set_defaults(func=aggregate)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
