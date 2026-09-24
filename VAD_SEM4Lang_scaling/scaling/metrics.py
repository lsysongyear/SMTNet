import numpy as np


def compute_metrics(targets, predictions):
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))

    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    f1_0 = (
        2 * precision_0 * recall_0 / (precision_0 + recall_0)
        if precision_0 + recall_0
        else 0.0
    )
    f1_1 = (
        2 * precision_1 * recall_1 / (precision_1 + recall_1)
        if precision_1 + recall_1
        else 0.0
    )
    return {
        "macro_acc": float((recall_0 + recall_1) / 2),
        "macro_f1": float((f1_0 + f1_1) / 2),
        "binary_acc": float(np.mean(predictions == targets)),
        "class0_precision": float(precision_0),
        "class0_recall": float(recall_0),
        "class0_f1": float(f1_0),
        "class1_precision": float(precision_1),
        "class1_recall": float(recall_1),
        "class1_f1": float(f1_1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
