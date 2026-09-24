import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def _completed_runs(scale_dir):
    runs = []
    for status_path in scale_dir.glob("run_*/status.json"):
        with status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("status") == "completed":
            runs.append(status_path.parent)
    return sorted(runs, key=lambda path: path.stat().st_mtime)


def collect_split_results(model_root, subset_seed=1001, model_seed=42):
    rows = []
    model_root = Path(model_root)
    for split_dir in sorted(model_root.glob("split_seed_*")):
        split_seed = int(split_dir.name.rsplit("_", 1)[1])
        results_root = (
            split_dir / f"subset_seed_{subset_seed}" / f"model_seed_{model_seed}"
        )
        for scale_dir in sorted(results_root.glob("scale_*")):
            runs = _completed_runs(scale_dir)
            if not runs:
                print(f"Skipping incomplete scale directory: {scale_dir}")
                continue
            run_dir = runs[-1]
            if len(runs) > 1:
                print(f"Using newest of {len(runs)} completed runs for {scale_dir}")
            with (run_dir / "scaling_manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            with (run_dir / "per_subject_summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)

            reference = manifest.get("baseline_reference") or {}
            macro_acc = summary["metrics"]["macro_acc"]["mean"]
            train_recording = manifest["train_recording"]
            val_recording = manifest["val_recording"]
            rows.append(
                {
                    "split_seed": split_seed,
                    "scale": manifest["scale"],
                    "scale_percent": manifest["scale_percent"],
                    "train_recording_hours": train_recording["selected_hours"],
                    "full_train_recording_hours": train_recording["full_hours"],
                    "actual_train_fraction": train_recording["actual_fraction"],
                    "val_recording_hours": val_recording["selected_hours"],
                    "full_val_recording_hours": val_recording["full_hours"],
                    "actual_val_fraction": val_recording["actual_fraction"],
                    "train_samples": manifest["train_samples"],
                    "val_samples": manifest["val_samples"],
                    "test_samples": manifest.get("test_samples", ""),
                    "unique_selected_meg_coverage_hours": manifest[
                        "unique_selected_meg_coverage_hours"
                    ],
                    "train_window_hours_with_overlap": manifest[
                        "train_window_hours_with_overlap"
                    ],
                    "macro_acc": macro_acc,
                    "macro_f1": summary["metrics"]["macro_f1"]["mean"],
                    "binary_acc": summary["metrics"]["binary_acc"]["mean"],
                    "baseline_reference_macro_acc": reference.get("test_macro_acc", ""),
                    "baseline_delta_100pct": (
                        macro_acc - reference["test_macro_acc"]
                        if manifest["scale_percent"] == 100
                        and "test_macro_acc" in reference
                        else ""
                    ),
                    "run_dir": str(run_dir.resolve()),
                }
            )
    return sorted(rows, key=lambda row: (row["scale"], row["split_seed"]))


def aggregate_by_scale(split_rows):
    grouped = defaultdict(list)
    for row in split_rows:
        grouped[row["scale"]].append(row)

    summary_rows = []
    for scale, rows in sorted(grouped.items()):
        def values(name):
            return np.asarray([float(row[name]) for row in rows], dtype=float)

        macro_acc = values("macro_acc")
        macro_f1 = values("macro_f1")
        binary_acc = values("binary_acc")
        train_hours = values("train_recording_hours")
        val_hours = values("val_recording_hours")
        train_fractions = values("actual_train_fraction")
        val_fractions = values("actual_val_fraction")
        samples = values("train_samples")
        baseline_values = [
            float(row["baseline_reference_macro_acc"])
            for row in rows
            if row["baseline_reference_macro_acc"] != ""
        ]
        summary_rows.append(
            {
                "scale": scale,
                "scale_percent": rows[0]["scale_percent"],
                "n_splits": len(rows),
                "split_seeds": ",".join(str(row["split_seed"]) for row in rows),
                "actual_train_fraction_mean": float(np.mean(train_fractions)),
                "actual_val_fraction_mean": float(np.mean(val_fractions)),
                "train_samples_mean": float(np.mean(samples)),
                "train_samples_std": float(np.std(samples)),
                "train_recording_hours_mean": float(np.mean(train_hours)),
                "train_recording_hours_std": float(np.std(train_hours)),
                "val_recording_hours_mean": float(np.mean(val_hours)),
                "val_recording_hours_std": float(np.std(val_hours)),
                "macro_acc_mean": float(np.mean(macro_acc)),
                "macro_acc_split_std": float(np.std(macro_acc)),
                "macro_acc_min": float(np.min(macro_acc)),
                "macro_acc_max": float(np.max(macro_acc)),
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_split_std": float(np.std(macro_f1)),
                "binary_acc_mean": float(np.mean(binary_acc)),
                "binary_acc_split_std": float(np.std(binary_acc)),
                "baseline_reference_mean": (
                    float(np.mean(baseline_values))
                    if rows[0]["scale_percent"] == 100 and baseline_values
                    else ""
                ),
                "baseline_delta_mean_100pct": (
                    float(np.mean(macro_acc) - np.mean(baseline_values))
                    if rows[0]["scale_percent"] == 100 and baseline_values
                    else ""
                ),
            }
        )
    return summary_rows


def _write_table(rows, csv_path, json_path):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def write_outputs(split_rows, summary_rows, output_dir):
    if not split_rows:
        raise RuntimeError("No completed scaling runs were found")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(
        split_rows,
        output_dir / "scaling_split_results.csv",
        output_dir / "scaling_split_results.json",
    )
    _write_table(
        summary_rows,
        output_dir / "scaling_summary.csv",
        output_dir / "scaling_summary.json",
    )

    plot_rows = [
        row for row in summary_rows if row["scale_percent"] not in {40, 60}
    ]
    x = np.arange(len(plot_rows))
    y = np.asarray([row["macro_acc_mean"] for row in plot_rows])
    yerr = np.asarray([row["macro_acc_split_std"] for row in plot_rows])
    tick_labels = [
        f"{row['scale_percent']}%\n({row['train_recording_hours_mean']:.1f}h)"
        for row in plot_rows
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, label="Mean across split seeds")
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, label="Split-seed SD")
    for position, row in zip(x, plot_rows):
        ax.annotate(
            f"{100 * row['macro_acc_mean']:.2f}%",
            (position, row["macro_acc_mean"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlabel("Training-data scale and selected training duration")
    ax.set_ylabel("Test macro accuracy (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.set_title("MEG NIPS Training-Data Scaling")
    ax.set_xticks(x, tick_labels)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_curve_macro_acc.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root", default="results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only"
    )
    parser.add_argument("--subset-seed", type=int, default=1001)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="results/scaling/summary_duration_prefix_train_only_split42-46_subset1001_model42"
    )
    args = parser.parse_args()
    split_rows = collect_split_results(
        args.model_root, args.subset_seed, args.model_seed
    )
    summary_rows = aggregate_by_scale(split_rows)
    write_outputs(split_rows, summary_rows, args.output_dir)
    incomplete = [row for row in summary_rows if row["n_splits"] != 5]
    if incomplete:
        print("WARNING: some scales do not yet contain all five split seeds")
    print(f"Wrote scaling summary to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
