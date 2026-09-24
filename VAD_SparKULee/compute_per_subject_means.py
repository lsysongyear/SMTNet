"""
Aggregate per-subject single-run results into cv_per_subject.json and cv_grand_average.json.

Reads test_log.json from each subject's run directory, collects metrics,
saves per-subject and grand-average results.

Usage: python compute_per_subject_means.py [--result-dir <path>]
"""

import json
import os
import sys
import numpy as np

# Default result dir — match the run_name from search-space
DEFAULT_RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results/speech-detection",
    "model:brain_magic_speech_v1-dataset:sparkulee_vad",
    "split:per_sub",
    "input_dim-64_attention_dim-128_output_dim-1_kernel_size-25_depthwise_kernel-19_num_blocks1-5_num_blocks2-12_dropout-0.01_weight-1.0_1.0_loss_type-mse_lr-0.001",
)

METRICS = [
    "test_macro_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "test_class0_precision", "test_class0_recall", "test_class0_f1",
    "test_class1_precision", "test_class1_recall", "test_class1_f1",
    "test_binary_acc", "test_loss",
    "ensemble_macro_acc", "ensemble_macro_precision", "ensemble_macro_recall", "ensemble_macro_f1",
    "ensemble_class0_precision", "ensemble_class0_recall", "ensemble_class0_f1",
    "ensemble_class1_precision", "ensemble_class1_recall", "ensemble_class1_f1",
    "ensemble_binary_acc", "ensemble_loss",
]


def load_subject_result(base, subject):
    """Load test_log.json for a single subject. Returns metrics dict or None."""
    sub_dir = os.path.join(base, f"sub-{subject}")
    if not os.path.isdir(sub_dir):
        return None

    # Find run directory (there should be exactly one)
    run_dirs = sorted(d for d in os.listdir(sub_dir)
                      if os.path.isdir(os.path.join(sub_dir, d)) and d.startswith("run"))
    if not run_dirs:
        return None

    run_dir = run_dirs[0]  # take the first/only run
    log_path = os.path.join(sub_dir, run_dir, "test_log.json")
    if not os.path.exists(log_path):
        return None

    with open(log_path) as f:
        data = json.load(f)

    # Find top1 key
    top1_keys = [k for k in data if k.startswith("top1:")]
    ensemble_keys = [k for k in data if k.startswith("ensemble")]

    result = {}
    if top1_keys:
        result.update(data[top1_keys[0]])
    if ensemble_keys:
        result.update(data[ensemble_keys[0]])

    return result


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    if not os.path.isdir(base):
        print(f"ERROR: result directory not found: {base}")
        sys.exit(1)

    # Discover subjects
    subjects = sorted(
        d.replace("sub-", "") for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("sub-")
    )

    if not subjects:
        print(f"ERROR: no subject directories found in {base}")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects in {base}")

    cv_results = {}
    missing = []
    for sub in subjects:
        r = load_subject_result(base, sub)
        if r is None:
            missing.append(sub)
            continue
        cv_results[f"sub-{sub}"] = r

    if missing:
        print(f"WARNING: {len(missing)} subjects missing results: {missing}")

    # Save per-subject
    per_subject_path = os.path.join(base, "cv_per_subject.json")
    with open(per_subject_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"Saved: {per_subject_path} ({len(cv_results)} subjects)")

    # Grand average
    grand = {}
    for m in METRICS:
        vals = [cv_results[s][m] for s in cv_results if m in cv_results[s]]
        if vals:
            grand[f"{m}_mean"] = float(np.mean(vals))
            grand[f"{m}_std"] = float(np.std(vals, ddof=1))

    grand["n_subjects"] = len(cv_results)
    grand_path = os.path.join(base, "cv_grand_average.json")
    with open(grand_path, "w") as f:
        json.dump(grand, f, indent=2)
    print(f"Saved: {grand_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Grand Average ({len(cv_results)} subjects, single run each)")
    print("=" * 60)
    for label, prefix in [("Top1", "test_"), ("Ensemble", "ensemble_")]:
        print(f"\n  {label}:")
        for m_name in ["macro_f1", "binary_acc", "macro_acc", "macro_precision", "macro_recall", "class0_f1", "class1_f1"]:
            k = f"{prefix}{m_name}_mean"
            if k in grand:
                print(f"    {m_name:20s}: {grand[k]:.4f} ± {grand.get(f'{prefix}{m_name}_std', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
