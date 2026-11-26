# Forecasting 5-Day Realised Volatility of the S&P 500 ETF (SPY) Using Price-Derived and Macro-Financial Features

CSC1181 Group 15. Can price, volume, cross-asset and VIX features predict next-week
SPY volatility better than simple benchmarks?

- **Target** `y_t`: log realised volatility of SPY over the 5 trading days after day `t`.
- **Features**: 10, each built from data up to day `t` only (HAR memory, VIX, TLT/XLF
  cross-asset, volume, trend).
- **Models**: naive, HAR-RV (main benchmark), Elastic Net, LightGBM.
- **Split**: train 2005-03-30 to 2014-12-23 (2,452 rows, after a 5-row embargo), test 2015-01-02
  to 2017-11-03 (717 rows). The raw data cover 2005-02-25 to 2017-11-10.

## Repo layout

```
forecasting-spy-5-day-realised-volatility/
├── README.md               # how to get the data and run everything
├── requirements.txt
├── Main_Notebook.ipynb     # data, target, features, split -> model_table.csv
├── data/
│   ├── raw/                # the four downloaded files, never edited (committed)
│   └── processed/          # model_table.csv (generated, not committed)
├── src/foundation/         # evaluate.py: evaluate(), save_predictions()
├── slices/
│   ├── memory/             # Tarannum: naive, HAR-RV
│   ├── econometric/        # Asif: statistics checks, stationarity
│   ├── linear_ml/          # Victor: Elastic Net
│   └── nonlinear_ml/       # Sadman: LightGBM
├── integration/            # results_table.py (Tarannum), dm_tests.py (Asif),
│                           # errors_by_period.ipynb (shared)
└── results/                # preds_*.csv, metrics.csv, dm_tests.csv, figures/
```

## Getting the data

These four files are committed in `data/raw/`. To download them again, use exactly these names:

| File | Source |
|---|---|
| `spy.us.txt` | Kaggle, [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) (B. Marjanovic, CC0), `ETFs/` folder |
| `tlt.us.txt` | same dataset, `ETFs/` folder |
| `xlf.us.txt` | same dataset, `ETFs/` folder |
| `VIXCLS.csv` | FRED, [CBOE Volatility Index: VIX](https://fred.stlouisfed.org/series/VIXCLS) (original source Cboe). Download as CSV covering at least 2005-02-25 to 2017-11-10 |

`data/processed/` is git-ignored; step 2 of How to run rebuilds it.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebooks were run on Python 3.10 to 3.13; the pinned versions in `requirements.txt` come from Python 3.13.5.

## How to run

Run every command from the repo root.

1. **Install** (see Setup above).
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` lists the versions we ran with.

2. **Build the model table.**
   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \
     --output-dir /tmp/executed Main_Notebook.ipynb
   ```
   Writes `data/processed/model_table.csv` (2,452 train and 717 test rows), plus
   `volume_vs_target.png` and `split_cv_timeline.png`.

3. **Run the slice notebooks** (any order, after step 2).
   ```bash
   for nb in slices/memory/memory_models.ipynb slices/econometric/statistics_checks.ipynb \
             slices/linear_ml/elastic_net_model.ipynb slices/nonlinear_ml/lightgbm_model.ipynb; do
     jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \
       --output-dir /tmp/executed "$nb"
   done
   ```
   Writes the ACF figure, `results/stationarity_table.csv`, the EDA figures and
   `lgbm_fold2_crisis.png`. The run copies, with CV scores and the coefficient table, go to
   `/tmp/executed/`; the notebooks in the repo are not changed.

   **Leave `RUN_TEST = False`.** The committed `results/preds_*.csv` are the single test run
   (Wed 25 Nov); with `False` nothing touches them. `shap_lgbm.png` and
   `predicted_vs_actual.png` came from that run and are not made again.

4. **Integration** (reads `results/preds_*.csv`; the last step also reads the model table).
   ```bash
   python integration/results_table.py   # -> results/metrics.csv
   python integration/dm_tests.py        # -> results/dm_tests.csv
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 \
     --output-dir /tmp/executed integration/errors_by_period.ipynb   # -> results/figures/errors_by_period.png
   ```

## How we work

- One branch per member. Only edit your own `slices/<name>/` folder.
- `src/foundation/` changes only if all four agree; then everyone re-runs.
- Every model loads only `data/processed/model_table.csv`.
- Tuning uses cross-validation inside the training period only. The test set is used once,
  together, on Wed 25 Nov.
- Each model writes `results/preds_<model>.csv` (`date, y_true, y_pred`, test period only)
  with the shared `save_predictions()`. These files are the only hand-off between slices.
