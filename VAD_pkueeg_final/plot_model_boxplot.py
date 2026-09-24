"""
Plot per-subject macro_acc boxplot + scatter across all models (except vlaai).
BrainMagic v7 and its ablation variants are placed on the right.
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 11})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --- Collect data ---
model_data = {}

for model_dir in sorted(os.listdir(os.path.join(RESULTS_DIR, "speech-detection"))):
    if not model_dir.startswith("model:") or "vlaai" in model_dir:
        continue
    model_name = model_dir.split(":")[1].split("-dataset")[0]
    split_dir = os.path.join(RESULTS_DIR, "speech-detection", model_dir, "split:per_sub_5t5v")
    if not os.path.isdir(split_dir):
        continue
    param_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    if len(param_dirs) == 0:
        continue
    param_dir = os.path.join(split_dir, param_dirs[0])
    accs = []
    for subj in sorted(os.listdir(param_dir)):
        log_path = os.path.join(param_dir, subj, "pretrain_eval", "test_log.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            accs.append(json.load(f)["macro_acc"])
    if accs:
        model_data[model_name] = accs
        print(f"{model_name}: n={len(accs)}, mean={np.mean(accs):.4f}, std={np.std(accs):.4f}")

# --- Reorder: baselines first, BrainMagic v7 + ablations on the right ---
baseline_order = ["cnn_lstm", "dilated_conv", "eeg_conformer", "awavenet",
                  "pnpl_cnn_tcn"]
brainmagic_order = [
    "brain_magic_no_subject_attn",
    "brain_magic_no_short_conv",
    "brain_magic_no_feature_encoder",
    "brain_magic_speech_v7",
]
ordered_names = [n for n in baseline_order if n in model_data] + \
                [n for n in brainmagic_order if n in model_data]

short_names = {
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
display_names = [short_names[n] for n in ordered_names]
data = [model_data[n] for n in ordered_names]
n_models = len(ordered_names)

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

bp = ax.boxplot(data, tick_labels=display_names, patch_artist=True,
                widths=0.4, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=7),
                flierprops=dict(marker='o', markersize=4, alpha=0.4),
                medianprops=dict(color='black', linewidth=1.5))

# Color: Set2 colormap for all models
colors = plt.cm.Set2(np.linspace(0, 1, n_models))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Scatter points with jitter
np.random.seed(42)
for i, d in enumerate(data):
    jitter = np.random.normal(0, 0.06, size=len(d))
    ax.scatter(np.full_like(d, i + 1) + jitter, d, alpha=0.5, s=20,
               edgecolors='black', linewidths=0.3, facecolors=colors[i],
               zorder=3)

# Mean annotations
for i, d in enumerate(data):
    mean_val = np.mean(d)
    ax.annotate(f"{mean_val:.4f}", (i + 1.30, mean_val),
                textcoords="data", ha='left', va='center',
                fontsize=8, color='black', fontweight='bold')

ax.set_ylabel("Macro Accuracy")
ax.set_title("Per-Subject Macro Accuracy by Model (PKU EEG, per_sub_5t5v)")
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.48, 0.82)
ax.set_xlim(0.4, n_models + 1.0)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_boxplot.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
