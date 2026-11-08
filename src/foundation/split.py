"""Shared loading and cross-validation helpers. Every slice imports these, so all models
read the same table and tune on identical folds."""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

TABLE_PATH = "data/processed/model_table.csv"   # relative to the repo root
FEATURES = [f"f{i}" for i in range(1, 11)]
TARGET = "y"


def load_model_table(path=TABLE_PATH):
    """Columns: date, y, f1..f10, split ('train' or 'test'). Embargoed rows are not in the file."""
    return pd.read_csv(path, parse_dates=["date"])


def make_cv():
    """Time-series CV for tuning, on training rows only.

    gap=5 skips 5 rows between the fitting and scoring blocks: the target at day t uses returns
    t+1..t+5, so without the gap the last fitted targets would overlap the first scored day."""
    return TimeSeriesSplit(n_splits=5, gap=5)
