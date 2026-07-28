import math

import numpy as np


def classification_metrics(y_true, y_pred, num_classes=None):
    """Accuracy and macro-F1 for species classification.

    y_true, y_pred are integer class indices.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred))
    classes = range(num_classes) if num_classes else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "n": int(len(y_true)),
    }


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100.0
    bias = np.mean(err)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae_cm": float(mae),
        "rmse_cm": float(rmse),
        "mape_percent": float(mape),
        "bias_cm": float(bias),
        "r2": float(r2),
        "n": int(len(y_true)),
    }
