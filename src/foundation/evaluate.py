"""Shared scoring and prediction export, used by every model."""
import numpy as np
import pandas as pd


def evaluate(y_true, y_pred):
    """MAE, RMSE and R² on the log scale. R² = 0 means no better than predicting the average."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_pred - y_true
    return {
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "R2": 1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2),
    }


def save_predictions(model_name, test, y_pred):
    """Save a model's test forecasts to results/preds_<model_name>.csv (columns: date, y_true, y_pred)."""
    out = pd.DataFrame({"date": test["date"], "y_true": test["y"], "y_pred": np.asarray(y_pred)})
    out.to_csv(f"results/preds_{model_name}.csv", index=False)
