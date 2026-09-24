#!/usr/bin/env python3
"""Prepare and aggregate a fixed-split SParKULee model-seed sweep."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml


MODEL_NAME = "brain_magic_speech_v7"
DEFAULT_SUBJECTS = "1-3,5-16,18-24,26"
SEED_KEY = repr(("general", "seed"))


def parse_int_spec(spec):
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            start, end = int(start), int(end)
            if end < start:
                raise ValueError(f"Descending range is not allowed: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values or any(value < 0 for value in values):
        raise ValueError("Seeds must be non-negative integers")
    if len(values) != len(set(values)):
        raise ValueError("Seeds must be unique")
    return tuple(values)


def configuration_invariant_hash(config, search_space):
    config = deepcopy(config)
    general = config.setdefault("general", {})
    general.pop("output_path", None)
    general.pop("seed", None)
    sweep_metadata = general.get("model_seed_sweep")
    if isinstance(sweep_metadata, dict):
        sweep_metadata.pop("index", None)
        sweep_metadata.pop("model_seed", None)
    search_space = deepcopy(search_space)
    search_space.pop(SEED_KEY, None)
    payload = {"config": config, "search_space": search_space}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def trained_config_invariant_hash(config):
    config = deepcopy(config)
    general = config.setdefault("general", {})
    for key in (
        "seed",
        "output_path",
        "result_dir",
        "run_dir",
        "checkpoint_path",
        "model_path",
    ):
        general.pop(key, None)
    general.pop("model_seed_sweep", None)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_hashes(runtime_root):
    runtime_root = Path(runtime_root).resolve()
    paths = [runtime_root / "model_seed_sweep.py"]
    paths.extend(sorted((runtime_root / "speech_code").rglob("*.py")))
    return {
        str(path.relative_to(runtime_root)): file_sha256(path) for path in paths
    }


def _assert_runtime_unchanged(manifest):
    runtime_root = Path(manifest["runtime_root"]).resolve()
    actual = _runtime_hashes(runtime_root)
    expected = manifest["runtime_source_sha256"]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(actual) & set(expected)
            if actual[path] != expected[path]
        )
        raise RuntimeError(
            "Runtime snapshot changed: "
            f"missing={missing}, added={added}, changed={changed}"
        )


def _mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _distribution(values):
    mean, population_std = _mean_std(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = sample_std / math.sqrt(len(values))
    t_critical = 2.093024054 if len(values) == 20 else 1.96
    return {
        "mean": mean,
        "population_std": population_std,
        "sample_std": sample_std,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "standard_error": standard_error,
        "ci95": [
            mean - t_critical * standard_error,
            mean + t_critical * standard_error,
        ],
    }


def prepare(args):
    from search_shared_story_splits import (
        common_stories,
        discover_subject_stories,
        parse_subject_spec,
        verify_seed_map,
    )

    base_config_path = Path(args.base_config).resolve()
    base_search_space_path = Path(args.base_search_space).resolve()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot directory already exists: {snapshot_dir}")

    with base_config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    with base_search_space_path.open(encoding="utf-8") as handle:
        base_search_space = yaml.safe_load(handle)
    if MODEL_NAME not in base_config["model"]:
        raise KeyError(f"{MODEL_NAME} is missing from the base config")
    if SEED_KEY not in base_search_space:
        raise KeyError(f"{SEED_KEY} is missing from the base search space")
    if any(len(values) != 1 for values in base_search_space.values()):
        raise ValueError("Every base search-space entry must contain exactly one value")

    seeds = parse_int_spec(args.seeds)
    if seeds != tuple(range(1, 21)):
        raise ValueError("This experiment requires model seeds 1 through 20 exactly")
    source_candidate = base_config["general"].get("search_candidate", {})
    expected_candidate = {
        "index": 18,
        "validation_story": args.validation_story,
        "test_story": args.test_story,
        "split_seed": args.split_seed,
    }
    if source_candidate != expected_candidate:
        raise ValueError(
            f"Base config is not the expected candidate 18 snapshot: {source_candidate}"
        )
    subjects = parse_subject_spec(args.subjects)
    subject_stories = discover_subject_stories(base_config, subjects)
    shared = common_stories(subject_stories)
    pair = (args.validation_story, args.test_story)
    if pair[0] == pair[1] or not set(pair).issubset(shared):
        raise ValueError(f"Validation/test must be distinct common stories: {pair}")
    split_audit = verify_seed_map(
        subject_stories, shared, {pair: args.split_seed}
    )[pair]

    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    snapshot_dir.mkdir(parents=True)
    output_root.mkdir(parents=True)
    project_root = Path(__file__).resolve().parent
    runtime_root = snapshot_dir / "runtime"
    shutil.copytree(
        project_root / "speech_code",
        runtime_root / "speech_code",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(Path(__file__).resolve(), runtime_root / "model_seed_sweep.py")

    reference_output = Path(base_config["general"]["output_path"]).resolve()
    reference_trained_configs = list(
        reference_output.rglob("pretrain/run*/config.yaml")
    )
    if len(reference_trained_configs) != 1:
        raise RuntimeError(
            "Expected exactly one original candidate 18 trained config, found "
            f"{len(reference_trained_configs)} below {reference_output}"
        )
    reference_trained_config_path = reference_trained_configs[0]
    with reference_trained_config_path.open(encoding="utf-8") as handle:
        reference_trained_config = yaml.safe_load(handle)
    reference_trained_hash = trained_config_invariant_hash(
        reference_trained_config
    )
    tasks = []
    tsv_rows = []
    expected_hash = None
    for index, model_seed in enumerate(seeds):
        task_name = f"seed_{model_seed:03d}"
        task_snapshot = snapshot_dir / task_name
        task_output = output_root / task_name
        task_snapshot.mkdir()

        config = deepcopy(base_config)
        config["general"]["output_path"] = str(task_output)
        config["general"]["seed"] = model_seed
        config["general"]["model_seed_sweep"] = {
            "index": index,
            "model_seed": model_seed,
            "fixed_split_seed": args.split_seed,
            "validation_story": args.validation_story,
            "test_story": args.test_story,
        }
        search_space = deepcopy(base_search_space)
        search_space[SEED_KEY] = [model_seed]

        invariant_hash = configuration_invariant_hash(config, search_space)
        if expected_hash is None:
            expected_hash = invariant_hash
        elif invariant_hash != expected_hash:
            raise RuntimeError("A non-seed configuration difference was introduced")

        config_path = task_snapshot / "config.yaml"
        search_space_path = task_snapshot / "search-space.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        with search_space_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(search_space, handle, sort_keys=False)

        task = {
            "index": index,
            "model_seed": model_seed,
            "config_path": str(config_path),
            "search_space_path": str(search_space_path),
            "output_path": str(task_output),
            "configuration_invariant_sha256": invariant_hash,
        }
        tasks.append(task)
        tsv_rows.append(
            [index, model_seed, config_path, search_space_path, task_output]
        )

    manifest = {
        "protocol": "fixed_candidate18_model_seed_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_config": base_config["model"][MODEL_NAME],
        "subjects": list(subjects),
        "n_subjects": len(subjects),
        "model_seeds": list(seeds),
        "n_seeds": len(seeds),
        "split_seed": args.split_seed,
        "validation_story": args.validation_story,
        "test_story": args.test_story,
        "split_audit": split_audit,
        "configuration_invariant_sha256": expected_hash,
        "seed_semantics": (
            "training_random_seed_controls_window_jitter_initialization_"
            "shuffle_dropout_and_workers"
        ),
        "deterministic_algorithms": False,
        "runtime_root": str(runtime_root),
        "runtime_source_sha256": _runtime_hashes(runtime_root),
        "input_sha256": {
            "base_config": file_sha256(base_config_path),
            "base_search_space": file_sha256(base_search_space_path),
        },
        "reference_trained_config": str(reference_trained_config_path),
        "reference_trained_config_sha256": file_sha256(
            reference_trained_config_path
        ),
        "reference_trained_config_invariant_sha256": reference_trained_hash,
        "base_config": str(base_config_path),
        "base_search_space": str(base_search_space_path),
        "output_root": str(output_root),
        "selection_policy": (
            "Rank model seeds by validation Macro_Acc only. Test ranking is exploratory."
        ),
        "tasks": tasks,
    }
    manifest_path = snapshot_dir / "manifest.json"
    tasks_path = snapshot_dir / "tasks.tsv"
    subjects_path = snapshot_dir / "subjects.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with tasks_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(tsv_rows)
    subjects_path.write_text(args.subjects + "\n", encoding="utf-8")

    print(json.dumps({
        "manifest": str(manifest_path),
        "tasks_tsv": str(tasks_path),
        "subjects_file": str(subjects_path),
        "output_root": str(output_root),
        "n_seeds": len(seeds),
        "configuration_invariant_sha256": expected_hash,
    }, indent=2))


def _candidate_row(task, manifest):
    row = dict(task)
    output_path = Path(task["output_path"])
    with Path(task["config_path"]).open(encoding="utf-8") as handle:
        snapshot_config = yaml.safe_load(handle)
    with Path(task["search_space_path"]).open(encoding="utf-8") as handle:
        snapshot_search_space = yaml.safe_load(handle)
    if (
        int(snapshot_config["general"].get("seed", -1)) != task["model_seed"]
        or snapshot_config["general"].get("output_path") != str(output_path)
        or snapshot_search_space.get(SEED_KEY) != [task["model_seed"]]
        or configuration_invariant_hash(
            snapshot_config, snapshot_search_space
        ) != manifest["configuration_invariant_sha256"]
        or task["configuration_invariant_sha256"]
        != manifest["configuration_invariant_sha256"]
    ):
        row["status"] = "invalid_snapshot"
        return row
    summaries = list(output_path.rglob("pretrain_eval_summary.json"))
    if len(summaries) != 1:
        row["status"] = "missing" if not summaries else "ambiguous_summary"
        return row

    summary_path = summaries[0]
    result_base = summary_path.parent
    subject_logs = sorted(result_base.glob("sub-*/pretrain_eval/test_log.json"))
    if len(subject_logs) != manifest["n_subjects"]:
        row.update({
            "status": "incomplete",
            "completed_subjects": len(subject_logs),
            "summary_path": str(summary_path),
        })
        return row

    val_scores = []
    metric_values = {"macro_acc": [], "macro_f1": [], "binary_acc": []}
    subject_metrics = {}
    for path in subject_logs:
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        best_ckpt = Path(metrics["best_ckpt"]).resolve()
        if not best_ckpt.is_file():
            row.update({"status": "invalid_checkpoint_origin"})
            return row
        threshold = float(metrics["threshold"])
        if not 0.01 <= threshold <= 0.99:
            row.update({"status": "invalid_threshold"})
            return row
        subject = path.parents[1].name.removeprefix("sub-")
        val_score = float(metrics["val_macro_acc"])
        val_scores.append(val_score)
        for metric in metric_values:
            metric_values[metric].append(float(metrics[metric]))
        subject_metrics[subject] = {
            "validation_macro_acc": val_score,
            "test_macro_acc": float(metrics["macro_acc"]),
            "test_macro_f1": float(metrics["macro_f1"]),
            "test_binary_acc": float(metrics["binary_acc"]),
            "threshold": threshold,
            "best_ckpt": str(best_ckpt),
        }
    if set(subject_metrics) != set(manifest["subjects"]):
        row.update({"status": "invalid_subject_set"})
        return row

    trained_configs = list(result_base.glob("pretrain/run*/config.yaml"))
    if len(trained_configs) != 1:
        row.update({"status": "ambiguous_trained_config"})
        return row
    trained_config_path = trained_configs[0]
    with trained_config_path.open(encoding="utf-8") as handle:
        trained_config = yaml.safe_load(handle)
    trained_config_hash = trained_config_invariant_hash(trained_config)
    general = trained_config["general"]
    checkpoint_dir = Path(general["checkpoint_path"])
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    best_pointer = result_base / "pretrain/best_ckpt_fold0.txt"
    subject_npz = sorted(result_base.glob("sub-*/pretrain_eval/test_results.npz"))
    if not (len(checkpoints) == 5 and best_pointer.is_file()):
        row.update({"status": "invalid_checkpoints"})
        return row
    pointed_checkpoint = Path(best_pointer.read_text(encoding="utf-8").strip()).resolve()
    resolved_checkpoints = {path.resolve() for path in checkpoints}
    if pointed_checkpoint not in resolved_checkpoints:
        row.update({"status": "invalid_best_checkpoint_pointer"})
        return row
    if len(subject_npz) != manifest["n_subjects"]:
        row.update({"status": "incomplete_subject_outputs"})
        return row
    for metrics in subject_metrics.values():
        if Path(metrics["best_ckpt"]).resolve() not in resolved_checkpoints:
            row.update({"status": "invalid_checkpoint_origin"})
            return row
    audit = general.get("split_story_audit", {})
    train_stories = set(audit.get("train_stories", []))
    val_stories = set(audit.get("val_stories", []))
    test_stories = set(audit.get("test_stories", []))
    overlap = (
        (train_stories & val_stories)
        | (train_stories & test_stories)
        | (val_stories & test_stories)
    )
    if (
        int(general.get("seed", -1)) != task["model_seed"]
        or int(general.get("split_seed", -1)) != manifest["split_seed"]
        or val_stories != {manifest["validation_story"]}
        or test_stories != {manifest["test_story"]}
        or overlap
        or trained_config["model"][MODEL_NAME] != manifest["model_config"]
        or trained_config_hash
        != manifest["reference_trained_config_invariant_sha256"]
        or general.get("model_seed_sweep") != {
            "index": task["index"],
            "model_seed": task["model_seed"],
            "fixed_split_seed": manifest["split_seed"],
            "validation_story": manifest["validation_story"],
            "test_story": manifest["test_story"],
        }
    ):
        row.update({"status": "invalid_configuration"})
        return row

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary.get("n_subjects", -1)) != manifest["n_subjects"]:
        row.update({"status": "invalid_summary_subject_count"})
        return row
    val_mean, val_std = _mean_std(val_scores)
    test_stats = {
        metric: _mean_std(values) for metric, values in metric_values.items()
    }
    for metric, (mean, std) in test_stats.items():
        if not (
            math.isclose(
                mean, float(summary[metric]["mean"]), rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(
                std, float(summary[metric]["std"]), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            row.update({"status": "invalid_summary"})
            return row
    test_mean, test_std = test_stats["macro_acc"]

    row.update({
        "status": "completed",
        "completed_subjects": len(subject_logs),
        "val_macro_acc_mean": val_mean,
        "val_macro_acc_std": val_std,
        "test_macro_acc_mean": test_mean,
        "test_macro_acc_std": test_std,
        "test_macro_f1_mean": test_stats["macro_f1"][0],
        "test_macro_f1_std": test_stats["macro_f1"][1],
        "test_binary_acc_mean": test_stats["binary_acc"][0],
        "test_binary_acc_std": test_stats["binary_acc"][1],
        "checkpoint_count": len(checkpoints),
        "summary_path": str(summary_path),
        "trained_config_path": str(trained_config_path),
        "trained_config_invariant_sha256": trained_config_hash,
        "subject_metrics": subject_metrics,
    })
    return row


def verify_runtime(args):
    with Path(args.manifest).resolve().open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _assert_runtime_unchanged(manifest)
    print(json.dumps({
        "runtime_root": manifest["runtime_root"],
        "verified_files": len(manifest["runtime_source_sha256"]),
        "status": "ok",
    }))


def aggregate(args):
    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _assert_runtime_unchanged(manifest)
    rows = [_candidate_row(task, manifest) for task in manifest["tasks"]]
    completed = [row for row in rows if row["status"] == "completed"]
    trained_hashes = {
        row["trained_config_invariant_sha256"] for row in completed
    }
    integrity_ok = (
        len(completed) == manifest["n_seeds"]
        and all(row["status"] == "completed" for row in rows)
        and len(trained_hashes) == 1
        and trained_hashes == {
            manifest["reference_trained_config_invariant_sha256"]
        }
    )
    by_validation = sorted(
        completed, key=lambda row: row["val_macro_acc_mean"], reverse=True
    )
    by_test = sorted(
        completed, key=lambda row: row["test_macro_acc_mean"], reverse=True
    )
    per_subject_across_seed = {}
    if integrity_ok:
        for subject in manifest["subjects"]:
            values = [
                row["subject_metrics"][subject]["test_macro_acc"]
                for row in completed
            ]
            per_subject_across_seed[subject] = _distribution(values)
    report = {
        "protocol": manifest["protocol"],
        "manifest": str(manifest_path),
        "n_seeds": manifest["n_seeds"],
        "n_completed": len(completed),
        "integrity_ok": integrity_ok,
        "trained_config_invariant_sha256": (
            next(iter(trained_hashes)) if len(trained_hashes) == 1 else None
        ),
        "fixed_split": {
            "split_seed": manifest["split_seed"],
            "validation_story": manifest["validation_story"],
            "test_story": manifest["test_story"],
        },
        "primary_reporting": "across_seed_distribution_without_seed_selection",
        "best_by_validation_diagnostic": by_validation[0] if by_validation else None,
        "best_by_test_exploratory": by_test[0] if by_test else None,
        "validation_ranking_diagnostic": by_validation,
        "test_ranking_exploratory": by_test,
        "across_seed_validation_macro_acc": (
            _distribution([row["val_macro_acc_mean"] for row in completed])
            if integrity_ok else None
        ),
        "across_seed_test_macro_acc": (
            _distribution([row["test_macro_acc_mean"] for row in completed])
            if integrity_ok else None
        ),
        "per_subject_across_seed_test_macro_acc": per_subject_across_seed,
        "all_seeds": rows,
    }

    output_root = Path(manifest["output_root"])
    json_path = output_root / "model_seed_sweep_summary.json"
    csv_path = output_root / "model_seed_sweep_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    fieldnames = [
        "index", "model_seed", "status", "completed_subjects",
        "val_macro_acc_mean", "val_macro_acc_std",
        "test_macro_acc_mean", "test_macro_acc_std",
        "test_macro_f1_mean", "test_macro_f1_std",
        "test_binary_acc_mean", "test_binary_acc_std",
        "checkpoint_count", "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "n_completed": len(completed),
        "integrity_ok": integrity_ok,
        "best_by_validation_diagnostic": report["best_by_validation_diagnostic"],
        "best_by_test_exploratory": report["best_by_test_exploratory"],
        "across_seed_test_macro_acc": report["across_seed_test_macro_acc"],
    }, indent=2))
    if not integrity_ok:
        statuses = {row["model_seed"]: row["status"] for row in rows}
        raise RuntimeError(f"Incomplete or inconsistent seed sweep: {statuses}")


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--base-config", required=True)
    prepare_parser.add_argument("--base-search-space", required=True)
    prepare_parser.add_argument("--snapshot-dir", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--seeds", default="1-20")
    prepare_parser.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    prepare_parser.add_argument("--split-seed", type=int, default=84)
    prepare_parser.add_argument("--validation-story", default="audiobook-5-1")
    prepare_parser.add_argument("--test-story", default="audiobook-5-3")
    prepare_parser.set_defaults(func=prepare)

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.set_defaults(func=aggregate)

    verify_parser = commands.add_parser("verify-runtime")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.set_defaults(func=verify_runtime)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
