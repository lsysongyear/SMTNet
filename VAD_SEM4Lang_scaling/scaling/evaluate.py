import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scaling.metrics import compute_metrics
from scaling.subset import ALL_SUBJECTS, StorySplit
from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.utils import get_dataset_partition_from_config


def _extract_subject_ids(batch, device):
    if len(batch) < 3 or not isinstance(batch[2], dict):
        return None
    values = batch[2].get("subject_id")
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.to(device)
    return torch.tensor(values, dtype=torch.long, device=device)


def _build_loader(config, partition, subject, stories):
    partition_config = copy.deepcopy(config["data"]["datasets"][partition])
    dataset_config = next(iter(partition_config[0].values()))
    dataset_config["include_subjects"] = [subject]
    dataset_config["stories"] = list(stories)
    dataset_config["data_type"] = partition
    dataset = get_dataset_partition_from_config(partition_config)
    return DataLoader(dataset, shuffle=False, **config["data"]["dataloader"])


def _run_inference(model, loader, device):
    probabilities = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device)
            subject_ids = _extract_subject_ids(batch, device)
            probabilities.append(model(x, subject_ids=subject_ids).cpu().flatten())
            targets.append(batch[1].flatten())
    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def _thresholds(config):
    general = config["general"]
    start = float(general.get("threshold_min", 0.01))
    stop = float(general.get("threshold_max", 0.99))
    step = float(general.get("threshold_step", 0.01))
    count = int(round((stop - start) / step)) + 1
    return start + np.arange(count) * step


def evaluate_run(run_dir, config, story_split, device=None):
    run_dir = Path(run_dir)
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = _thresholds(config)
    output_root = run_dir / "per_subject"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for subject in ALL_SUBJECTS:
        print(f"[EVAL] sub-{subject}: building fixed validation/test loaders")
        val_loader = _build_loader(config, "val", subject, story_split.val)
        test_loader = _build_loader(config, "test", subject, story_split.test)

        best_val_macro_acc = -1.0
        best_checkpoint = None
        best_threshold = None
        for checkpoint in checkpoints:
            model = ClassificationModule.load_from_checkpoint(str(checkpoint)).to(device)
            val_probabilities, val_targets = _run_inference(model, val_loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for threshold in thresholds:
                predictions = (val_probabilities >= threshold).astype(np.int64)
                macro_acc = compute_metrics(val_targets, predictions)["macro_acc"]
                if macro_acc > best_val_macro_acc:
                    best_val_macro_acc = macro_acc
                    best_checkpoint = checkpoint
                    best_threshold = float(threshold)

        model = ClassificationModule.load_from_checkpoint(str(best_checkpoint)).to(device)
        test_probabilities, test_targets = _run_inference(model, test_loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        predictions = (test_probabilities >= best_threshold).astype(np.int64)
        metrics = compute_metrics(test_targets, predictions)
        metrics.update(
            {
                "subject": subject,
                "val_stories": list(story_split.val),
                "test_stories": list(story_split.test),
                "threshold": best_threshold,
                "best_checkpoint": str(best_checkpoint.resolve()),
                "val_macro_acc": float(best_val_macro_acc),
            }
        )
        subject_dir = output_root / f"sub-{subject}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            subject_dir / "test_results.npz",
            probabilities=test_probabilities.astype(np.float32),
            targets=test_targets.astype(np.int64),
            predictions=predictions,
            threshold=np.float32(best_threshold),
        )
        with (subject_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        rows.append(metrics)
        print(
            f"[EVAL] sub-{subject}: val_macro_acc={best_val_macro_acc:.4f}, "
            f"test_macro_acc={metrics['macro_acc']:.4f}, threshold={best_threshold:.2f}"
        )

    metric_names = ("macro_acc", "macro_f1", "binary_acc")
    summary = {
        "n_subjects": len(rows),
        "selection_metric": "val_macro_acc",
        "metrics": {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "std": float(np.std([row[name] for row in rows])),
            }
            for name in metric_names
        },
    }
    with (run_dir / "per_subject_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    compatibility_summary = {
        "n_subjects": summary["n_subjects"],
        **summary["metrics"],
    }
    with (run_dir / "pretrain_eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(compatibility_summary, handle, indent=2)
    with (run_dir / "per_subject_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _split_from_config(config):
    serialized = config["general"]["scaling"]["story_split"]
    return StorySplit(
        train=tuple(serialized["train"]),
        val=tuple(serialized["val"]),
        test=tuple(serialized["test"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    summary = evaluate_run(run_dir, config, _split_from_config(config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
