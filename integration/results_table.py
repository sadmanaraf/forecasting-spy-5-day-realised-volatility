"""Final results table: scores every model's test predictions with the shared evaluate().

Reads results/preds_<model>.csv (columns: date, y_true, y_pred), one file per model,
prints one row per model sorted by RMSE and saves it to results/metrics.csv.
It never runs a model; it only reads the prediction files.

Run from the repo root:  python integration/results_table.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())
from src.foundation.evaluate import evaluate

files = sorted(glob.glob("results/preds_*.csv"))
if not files:
    sys.exit("No results/preds_*.csv files found. Run the models' test cells first.")

scores = {}
for path in files:
    model = os.path.basename(path).removeprefix("preds_").removesuffix(".csv")
    preds = pd.read_csv(path, parse_dates=["date"])
    print(f"{model}: {len(preds)} rows, {preds['date'].min().date()} to {preds['date'].max().date()}")
    scores[model] = evaluate(preds["y_true"], preds["y_pred"])

# One row per model, best (lowest RMSE) first
table = pd.DataFrame(scores).T.sort_values("RMSE").rename_axis("model")
print()
print(table.round(3))

table.to_csv("results/metrics.csv")
print("\nSaved results/metrics.csv")
