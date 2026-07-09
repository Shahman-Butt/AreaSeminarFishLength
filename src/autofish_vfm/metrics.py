import math

import numpy as np


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
