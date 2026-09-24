#!/usr/bin/env python3
"""Re-run only per-subject evaluation for one completed split-search candidate."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as lightning
import torch
import yaml

from speech_code.train_v1 import (
    _parse_subject_range,
    _pretrain_result_base,
    infer_split_items,
    run_per_subject_eval,
)


def _newest(paths):
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("No completed training config was found")
    return max(paths, key=lambda path: path.stat().st_mtime)


def main(args):
    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    candidates = {int(item["index"]): item for item in manifest["candidates"]}
    if args.candidate_index not in candidates:
        raise ValueError(f"Unknown candidate index: {args.candidate_index}")
    candidate = candidates[args.candidate_index]
    output_path = Path(candidate["output_path"])
    trained_config_path = _newest(output_path.rglob("pretrain/run*/config.yaml"))
    with trained_config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    split_seed = int(config["general"]["split_seed"])
    if split_seed != int(candidate["split_seed"]):
        raise RuntimeError(
            f"Split seed mismatch: trained={split_seed}, expected={candidate['split_seed']}"
        )
    audit = config["general"].get("split_story_audit", {})
    if set(audit.get("val_stories", [])) != {candidate["validation_story"]}:
        raise RuntimeError("Validation story in trained config does not match manifest")
    if set(audit.get("test_stories", [])) != {candidate["test_story"]}:
        raise RuntimeError("Test story in trained config does not match manifest")

    checkpoint_dir = Path(config["general"]["checkpoint_path"])
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    all_subjects, _, subject_stories_map = infer_split_items(config)
    configured_subjects = config["general"].get("pretrain_subjects")
    eval_subjects = (
        sorted(_parse_subject_range(configured_subjects, width=3))
        if configured_subjects
        else sorted(all_subjects)
    )
    if eval_subjects != sorted(manifest["subjects"]):
        raise RuntimeError("Evaluated subjects do not match the search manifest")

    torch.set_float32_matmul_precision("high")
    lightning.seed_everything(config["general"]["seed"], workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"candidate={candidate['index']} split_seed={split_seed} "
        f"val={candidate['validation_story']} test={candidate['test_story']}"
    )
    print(f"trained_config={trained_config_path}")
    print(f"checkpoints={len(checkpoints)} device={device}")
    run_per_subject_eval(
        str(checkpoint_dir),
        split_seed,
        eval_subjects,
        subject_stories_map,
        config,
        device,
    )

    result_base = Path(_pretrain_result_base(str(checkpoint_dir)))
    summary_path = result_base / "pretrain_eval_summary.json"
    subject_logs = sorted(result_base.glob("sub-*/pretrain_eval/test_log.json"))
    if not summary_path.is_file() or len(subject_logs) != manifest["n_subjects"]:
        raise RuntimeError(
            f"Incomplete re-evaluation: summary={summary_path.is_file()}, "
            f"subject_logs={len(subject_logs)}/{manifest['n_subjects']}"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary.get("n_subjects", -1)) != manifest["n_subjects"]:
        raise RuntimeError("Summary subject count does not match manifest")
    print(f"verified_summary={summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-index", required=True, type=int)
    main(parser.parse_args())
