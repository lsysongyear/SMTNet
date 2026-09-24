import argparse
import csv
import json
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


def collect_results(results_root):
    results_root = Path(results_root)
    rows = []
    subject_values = {}
    for scale_dir in sorted(results_root.glob("scale_*")):
        runs = _completed_runs(scale_dir)
        if not runs:
            print(f"Skipping incomplete scale directory: {scale_dir}")
            continue
        run_dir = runs[-1]
        if len(runs) > 1:
            print(f"Using newest of {len(runs)} completed runs for {scale_dir.name}")

        with (run_dir / "scaling_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with (run_dir / "per_subject_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with (run_dir / "per_subject_metrics.csv").open(encoding="utf-8") as handle:
            subject_rows = list(csv.DictReader(handle))

        macro_acc = summary["metrics"]["macro_acc"]
        train_recording = manifest["train_recording"]
        val_recording = manifest["val_recording"]
        row = {
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
            "macro_acc_mean": macro_acc["mean"],
            "macro_acc_subject_std": macro_acc["std"],
            "macro_f1_mean": summary["metrics"]["macro_f1"]["mean"],
            "binary_acc_mean": summary["metrics"]["binary_acc"]["mean"],
            "run_dir": str(run_dir.resolve()),
        }
        rows.append(row)
        subject_values[manifest["scale_percent"]] = [
            float(subject_row["macro_acc"]) for subject_row in subject_rows
        ]
    return rows, subject_values


def write_outputs(rows, subject_values, output_dir):
    if not rows:
        raise RuntimeError("No completed scaling runs were found")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row["scale"])

    with (output_dir / "scaling_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "scaling_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    plot_rows = [row for row in rows if row["scale_percent"] not in {40, 60}]
    x = np.arange(len(plot_rows))
    y = np.array([row["macro_acc_mean"] for row in plot_rows])
    yerr = np.array([row["macro_acc_subject_std"] for row in plot_rows])
    tick_labels = [
        f"{row['scale_percent']}%\n({row['train_recording_hours']:.1f}h)"
        for row in plot_rows
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, label="Mean across 25 subjects")
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, label="Subject SD")
    for position, row in zip(x, plot_rows):
        ax.annotate(
            f"{100 * row['macro_acc_mean']:.2f}%",
            (position, row["macro_acc_mean"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Training-data scale and selected training duration")
    ax.set_ylabel("Test macro accuracy (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.set_title("PKU EEG Training-Duration Scaling")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_curve_macro_acc.png", dpi=180)
    plt.close(fig)

    scales = [row["scale_percent"] for row in rows]
    values = [subject_values[scale] for scale in scales]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=[f"{scale}%" for scale in scales])
    ax.set_xlabel("Training-data scale")
    ax.set_ylabel("Per-subject test macro accuracy")
    ax.set_title("Per-Subject Scaling Results")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_boxplot_macro_acc.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=(
            "results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only/split_seed_545/"
            "subset_seed_1001/model_seed_1"
        ),
    )
    parser.add_argument("--output-dir", default="results/scaling/summary_duration_prefix_train_only_seed1001_model1")
    args = parser.parse_args()
    rows, subject_values = collect_results(args.results_root)
    write_outputs(rows, subject_values, args.output_dir)
    print(f"Wrote scaling summary to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
