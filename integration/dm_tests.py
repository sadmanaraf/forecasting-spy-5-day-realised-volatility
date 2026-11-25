"""
dm_tests.py  (Asif)

Diebold-Mariano test: is each model's forecast really better than HAR's,
or could the difference in errors just be luck?

- Reads only results/preds_<model>.csv (made by save_predictions()).
- Loss = squared error.
- Long-run variance with Newey-West, lag 4 (5-day target -> errors overlap
  for h-1 = 4 days).
- Harvey, Leybourne & Newbold (1997) small-sample correction, t-distribution.
- Holm adjustment because we test 3 models against HAR at the same time.

It never runs a model; it only reads the prediction files.

Run from the repo root:  python integration/dm_tests.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ---- settings (change these if save_predictions() uses other names) ----
PRED_DIR = "results"
BENCHMARK = "har"
DATE_COL = "date"
TRUE_COL = "y_true"
PRED_COL = "y_pred"
H = 5            # forecast horizon in days
LAG = H - 1      # Newey-West lag = 4
OUT_FILE = os.path.join(PRED_DIR, "dm_tests.csv")


def load_preds(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df[[DATE_COL, TRUE_COL, PRED_COL]].set_index(DATE_COL)


def dm_test(e_bench, e_model, lag=LAG, h=H):
    """d_t = bench loss - model loss. Positive mean d -> model is better."""
    d = e_bench ** 2 - e_model ** 2
    n = len(d)
    d_mean = d.mean()
    d_c = d - d_mean

    # Newey-West long-run variance of d (Bartlett weights)
    lrv = np.sum(d_c ** 2) / n
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        lrv += 2 * weight * np.sum(d_c[k:] * d_c[:-k]) / n

    dm = d_mean / np.sqrt(lrv / n)

    # Harvey-Leybourne-Newbold correction for small samples
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * stats.t.sf(abs(dm_hln), df=n - 1)   # two-sided
    return d_mean, dm_hln, p


def holm(pvals):
    """Holm step-down adjustment."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[i])
        adj[i] = min(1.0, running_max)
    return adj


# ---- load all prediction files ----
files = sorted(glob.glob(os.path.join(PRED_DIR, "preds_*.csv")))
if not files:
    sys.exit("No results/preds_*.csv files found. Run the models' test cells first.")
preds = {os.path.basename(f).removeprefix("preds_").removesuffix(".csv"): load_preds(f) for f in files}
print("Found:", list(preds))
if BENCHMARK not in preds:
    sys.exit("results/preds_har.csv is missing, and HAR is the benchmark.")

bench = preds.pop(BENCHMARK)
rows = []
for name, df in preds.items():
    both = bench.join(df, lsuffix="_b", rsuffix="_m", how="inner")
    # every model must be scored on exactly the same days and targets
    assert np.allclose(both[TRUE_COL + "_b"], both[TRUE_COL + "_m"]), name
    e_b = (both[TRUE_COL + "_b"] - both[PRED_COL + "_b"]).to_numpy()
    e_m = (both[TRUE_COL + "_m"] - both[PRED_COL + "_m"]).to_numpy()
    d_mean, dm, p = dm_test(e_b, e_m)
    rows.append({"model": name, "n": len(both),
                 "MSE_model": np.mean(e_m ** 2), "MSE_HAR": np.mean(e_b ** 2),
                 "mean_d": d_mean, "DM": dm, "p": p})

res = pd.DataFrame(rows)
res["p_holm"] = holm(res["p"])
res["better_than_HAR"] = np.where(
    res["p_holm"] >= 0.05, "no significant difference",
    np.where(res["DM"] > 0, "yes", "no, worse"))

pd.set_option("display.width", 120)
print(f"\nDiebold-Mariano vs HAR (squared error, lag {LAG}, HLN, Holm)")
print("DM > 0 means the model has smaller errors than HAR\n")
print(res.round(4).to_string(index=False))

res.to_csv(OUT_FILE, index=False)
print("\nSaved", OUT_FILE)
