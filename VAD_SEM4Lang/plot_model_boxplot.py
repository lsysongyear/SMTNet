"""Plot the SEM4Lang per-subject model comparison used by the other VAD projects."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({"font.size": 11})

BASELINE_ORDER = ("cnn_lstm", "dilated_conv", "eeg_conformer", "awavenet")
PUBLIC_BASELINES = ("pnpl_cnn_tcn",)
BRAINMAGIC_ORDER = (
    "brain_magic_no_subject_attn",
    "brain_magic_no_short_conv",
    "brain_magic_no_feature_encoder",
    "brain_magic_speech_v7",
)
DISPLAY_NAMES = {
    "brain_magic_speech_v7": "MEBM",
    "brain_magic_no_subject_attn": "NoSpatialAttn",
    "brain_magic_no_short_conv": "NoMultiScaleConv",
    "brain_magic_no_feature_encoder": "NoBrainMagicBlock",
    "awavenet": "AWaveNet",
    "cnn_lstm": "CNN-LSTM",
    "dilated_conv": "Dilated\nConv",
    "eeg_conformer": "EEG\nConformer",
    "pnpl_cnn_tcn": "CNN+TCN",
}


def collect_model_data(results_dir):
    speech_results = Path(results_dir) / "speech-detection"
    if not speech_results.is_dir():
        raise FileNotFoundError(f"Missing results directory: {speech_results}")

    model_data = {}
    for model_dir in sorted(speech_results.glob("model:*-dataset:sem4lang_meg")):
        model_name = model_dir.name.split(":", 1)[1].split("-dataset:", 1)[0]
        if model_name not in BASELINE_ORDER + PUBLIC_BASELINES + BRAINMAGIC_ORDER:
            continue

        by_subject = {}
        pattern = "run5:seed1_fold0/sub-*/pretrain_eval/test_log.json"
        for log_path in sorted(model_dir.rglob(pattern)):
            subject = log_path.parents[1].name
            if subject in by_subject:
                raise RuntimeError(f"Duplicate result for {model_name}/{subject}")
            with log_path.open(encoding="utf-8") as handle:
                by_subject[subject] = float(json.load(handle)["macro_acc"])

        if by_subject:
            if len(by_subject) != 12:
                raise RuntimeError(
                    f"Expected 12 subjects for {model_name}, found {len(by_subject)}"
                )
            model_data[model_name] = [by_subject[key] for key in sorted(by_subject)]
    return model_data


def plot_model_boxplot(model_data, output_path):
    ordered_names = [
        name for name in BASELINE_ORDER + PUBLIC_BASELINES + BRAINMAGIC_ORDER if name in model_data
    ]
    missing = set(BASELINE_ORDER + BRAINMAGIC_ORDER) - set(ordered_names)
    if missing:
        raise RuntimeError(f"Missing model results: {sorted(missing)}")

    data = [model_data[name] for name in ordered_names]
    display_names = [DISPLAY_NAMES[name] for name in ordered_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(ordered_names)))

    fig, ax = plt.subplots(figsize=(14, 6))
    boxplot = ax.boxplot(
        data,
        tick_labels=display_names,
        patch_artist=True,
        widths=0.4,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 7},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.4},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    random = np.random.default_rng(42)
    for index, values in enumerate(data):
        values = np.asarray(values)
        jitter = random.normal(0, 0.06, size=len(values))
        ax.scatter(
            np.full_like(values, index + 1) + jitter,
            values,
            alpha=0.5,
            s=20,
            edgecolors="black",
            linewidths=0.3,
            facecolors=colors[index],
            zorder=3,
        )

    for index, values in enumerate(data):
        mean_value = float(np.mean(values))
        ax.annotate(
            f"{mean_value:.4f}",
            (index + 1.30, mean_value),
            ha="left",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold",
        )

    ax.set_ylabel("Macro Accuracy")
    ax.set_title(
        "Per-Subject Macro Accuracy by Model "
        "(SEM4Lang MEG, per_sub_5t5v, split_seed=5)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0.4, len(ordered_names) + 1.0)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    for name, values in zip(ordered_names, data):
        print(
            f"{name}: n={len(values)}, mean={np.mean(values):.4f}, "
            f"std={np.std(values):.4f}"
        )
    print(f"Saved to {output_path.resolve()}")


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(project_root / "results_split5_all_models"),
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "model_boxplot.png"),
    )
    args = parser.parse_args()
    plot_model_boxplot(collect_model_data(args.results_dir), args.output)


if __name__ == "__main__":
    main()
