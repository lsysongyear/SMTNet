import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_code.speech_utils import my_run_test
from speech_code.utils import get_dataset_partition_from_config, my_log_results


def _top1_metrics(results):
    keys = [key for key in results if key.startswith("top1:")]
    if len(keys) != 1:
        raise RuntimeError(f"Expected exactly one top1 result, found {keys}")
    return keys[0], results[keys[0]]


def evaluate_run(run_dir, config, val_dataset=None, test_dataset=None, labels=None):
    run_dir = Path(run_dir)
    if val_dataset is None:
        val_dataset = get_dataset_partition_from_config(
            config["data"]["datasets"]["val"]
        )
    if test_dataset is None:
        test_dataset = get_dataset_partition_from_config(
            config["data"]["datasets"]["test"]
        )
    if labels is None:
        labels = val_dataset.datasets[0].labels_sorted
    loader_config = config["data"]["dataloader"]
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_config)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_config)

    general = config["general"]
    start = float(general.get("threshold_min", 0.01))
    stop = float(general.get("threshold_max", 0.99))
    step = float(general.get("threshold_step", 0.01))
    count = int(round((stop - start) / step)) + 1
    thresholds = [round(start + index * step, 10) for index in range(count)]
    checkpoint_dir = str(run_dir / "checkpoints")
    weight = config["loss"]["config"]["weight"]

    val_results, best_model_paths, best_thresholds = my_run_test(
        val_loader,
        checkpoint_dir,
        labels,
        weight,
        thresholds,
        prefix="val_",
        test_results_path=str(run_dir / "val_results.npz"),
    )
    my_log_results(val_results, str(run_dir), "val_log.json")
    test_results, _, _ = my_run_test(
        test_loader,
        checkpoint_dir,
        labels,
        weight,
        thresholds,
        prefix="test_",
        test_results_path=str(run_dir / "test_results.npz"),
        fixed_ckpt_path=best_model_paths[0],
        fixed_threshold=best_thresholds[0],
    )
    my_log_results(test_results, str(run_dir), "test_log.json")

    val_key, val_metrics = _top1_metrics(val_results)
    test_key, test_metrics = _top1_metrics(test_results)
    row = {
        "subject": "0",
        "val_macro_acc": val_metrics["val_macro_acc"],
        "macro_acc": test_metrics["test_macro_acc"],
        "macro_f1": test_metrics["test_macro_f1"],
        "binary_acc": test_metrics["test_binary_acc"],
        "threshold": float(best_thresholds[0]),
        "best_checkpoint": str(Path(best_model_paths[0]).resolve()),
        "val_result_key": val_key,
        "test_result_key": test_key,
    }
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
    with (run_dir / "per_subject_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    summary = {
        "n_subjects": 1,
        "selection_metric": "val_macro_acc",
        "metrics": {
            "macro_acc": {"mean": row["macro_acc"], "std": 0.0},
            "macro_f1": {"mean": row["macro_f1"], "std": 0.0},
            "binary_acc": {"mean": row["binary_acc"], "std": 0.0},
        },
    }
    with (run_dir / "per_subject_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(json.dumps(evaluate_run(run_dir, config), indent=2))


if __name__ == "__main__":
    main()
