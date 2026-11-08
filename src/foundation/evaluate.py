"""Shared scoring and prediction export. Every model is scored by evaluate() and handed over
through save_predictions(), so the comparison table and the DM tests see identical inputs."""
from pathlib import Path

import numpy as np
import pandas as pd

from .split import TABLE_PATH, load_model_table


def evaluate(y_true, y_pred):
    """MAE, RMSE, out-of-sample R2 and QLIKE, all on the log scale (y = log realised volatility).

    R2 = 1 - SSE / sum((y_true - mean(y_true))^2), the scikit-learn definition: 0 means no better
    than always predicting the mean of the scored sample, negative means worse.

    QLIKE (Patton, 2011) compares variances. With h = exp(2*y_true) and h_hat = exp(2*y_pred),
    QLIKE = mean(h/h_hat - ln(h/h_hat) - 1) = mean(exp(z) - z - 1) with z = 2*(y_true - y_pred).
    It is 0 for a perfect forecast and punishes under-prediction (z > 0) harder than over-prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must be 1-D and the same length, got {y_true.shape} and {y_pred.shape}")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must not contain NaN or infinite values")

    err = y_pred - y_true
    z = -2 * err
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "QLIKE": float(np.mean(np.exp(z) - z - 1)),
    }


def save_predictions(model, dates, y_true, y_pred, results_dir="results", table_path=TABLE_PATH):
    """Write results/preds_<model>.csv with columns date, y_true, y_pred.

    The file must hold exactly the test-period rows of the model table, in order, with the same y,
    so every model is compared on the same days. Anything else raises instead of saving."""
    out = pd.DataFrame({"date": pd.to_datetime(np.asarray(dates)),
                        "y_true": np.asarray(y_true, dtype=float),
                        "y_pred": np.asarray(y_pred, dtype=float)})
    test = load_model_table(table_path).query("split == 'test'")

    if len(out) != len(test) or not (out["date"].to_numpy() == test["date"].to_numpy()).all():
        raise ValueError(f"predictions must cover exactly the {len(test)} test dates "
                         f"({test['date'].min().date()} to {test['date'].max().date()}), in order")
    if not np.allclose(out["y_true"], test["y"]):
        raise ValueError("y_true does not match y in the model table")
    if not np.isfinite(out["y_pred"]).all():
        raise ValueError("y_pred contains NaN or infinite values")

    path = Path(results_dir) / f"preds_{model}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
