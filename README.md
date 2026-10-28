# Forecasting 5-Day Realised Volatility of the S&P 500 ETF (SPY) Using Price-Derived and Macro-Financial Features

CSC1181 Group 15. Can price, volume, cross-asset and VIX features predict next-week
SPY volatility better than simple benchmarks?

- **Target** `y_t`: log realised volatility of SPY over the 5 trading days after day `t`.
- **Features**: 10, each built from data up to day `t` only (HAR memory, VIX, TLT/XLF
  cross-asset, volume, trend).
- **Models**: naive, expanding mean, HAR-RV (main benchmark), GARCH(1,1), Elastic Net, LightGBM.
- **Split**: train 2005-02-25 to 2014-12-31 (last 5 rows embargoed), test 2015-01-01 to 2017-11-10.

## Repo layout

```
forecasting-spy-5-day-realised-volatility/
├── README.md               # how to get the data and run everything
├── requirements.txt
├── data/
│   ├── raw/                # downloaded files, never edited (not committed)
│   └── processed/          # model_table.csv (generated, not committed)
├── src/foundation/         # frozen at v1.0: data, features, split, evaluate
├── slices/
│   ├── memory/             # Tarannum: naive, expanding mean, HAR-RV
│   ├── econometric/        # Asif: GARCH(1,1)
│   ├── linear_ml/          # Victor: Elastic Net
│   └── nonlinear_ml/       # Sadman: LightGBM
├── integration/            # results_table.py (Tarannum), dm_tests.py (Asif)
└── results/                # preds_*.csv, metrics.csv, figures/
```

## Getting the data

Put these four files in `data/raw/` with exactly these names:

| File | Source |
|---|---|
| `spy.us.txt` | Kaggle, [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) (B. Marjanovic, CC0), `ETFs/` folder |
| `tlt.us.txt` | same dataset, `ETFs/` folder |
| `xlf.us.txt` | same dataset, `ETFs/` folder |
| `VIXCLS.csv` | FRED, [CBOE Volatility Index: VIX](https://fred.stlouisfed.org/series/VIXCLS) (original source Cboe). Download as CSV covering at least 2005-02-25 to 2017-11-10 |

Raw and processed data are git-ignored, so everyone downloads their own copy.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

To be filled in once the foundation is tagged `v1.0`.

## How we work

- One branch per member. Only edit your own `slices/<name>/` folder.
- `src/foundation/` changes only if all four agree; then everyone re-runs.
- Every model loads only `data/processed/model_table.csv`.
- Tuning uses cross-validation inside the training period only. The test set is used once,
  together, on Sat 14 Nov.
- Each model writes `results/preds_<model>.csv` (`date, y_true, y_pred`, test period only)
  with the shared `save_predictions()`. These files are the only hand-off between slices.
