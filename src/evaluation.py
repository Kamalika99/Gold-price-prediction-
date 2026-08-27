"""
evaluation.py

Metric computation. Every metric is computed from actual predictions vs.
actual values on the held-out test set -- nothing here is hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_results_table(results: dict) -> pd.DataFrame:
    """results: {model_name: metrics_dict}"""
    return pd.DataFrame(results).T[["MAE", "RMSE", "R2"]].sort_values("RMSE")
