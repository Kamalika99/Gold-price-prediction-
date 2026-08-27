# GoldCast — Next-Day Gold Price Forecasting

A time-series machine learning project that forecasts the next trading
day's gold closing price from historical daily prices. It compares a
naive baseline, ARIMA, Random Forest, and XGBoost under a strict
leakage-free, chronologically-validated pipeline — and reports the
result honestly, including the fact that the simplest model wins.

## Table of contents

- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Feature engineering](#feature-engineering)
- [Train / validation / test split](#train--validation--test-split)
- [Models](#models)
- [Evaluation metrics](#evaluation-metrics)
- [Results](#results)
- [Findings and interpretation](#findings-and-interpretation)
- [Feature importance](#feature-importance)
- [What this project demonstrates](#what-this-project-demonstrates)
- [Known limitations](#known-limitations)
- [Suggested next steps](#suggested-next-steps)
- [Repository structure](#repository-structure)
- [Running it](#running-it)
- [License](#license)

## Problem statement

Given daily gold price history up to and including trading day *t*,
predict the closing price on trading day *t+1*. This is a one-step-ahead,
univariate-history forecasting problem — no macro or fundamental data,
just price. The goal isn't to build a trading signal; it's to build the
forecasting pipeline *correctly*: no data leakage, a split that respects
time, baselines strong enough to make "the model works" a falsifiable
claim, and metrics computed only on data the model never touched.

## Dataset

- **Source:** daily gold price history, `Date, Price, Open, High, Low, Chg%`
  columns — the standard daily-history export format used by financial
  data sites such as Investing.com. See [`data/README.md`](data/README.md)
  for the exact schema; the raw file is not committed (kept out of the
  repo to stay minimal) but is required at `data/GoldPrice.csv` to run
  the pipeline.
- **Size:** 2,531 rows, 2011-01-03 to 2020-09-11.
- **Quality:** 0 missing values, 0 duplicate rows, 0 duplicate dates
  (verified programmatically in `src/preprocessing.py`, not assumed).
- **One data quirk handled:** the raw file ships newest-first. It's
  re-sorted chronologically before anything else happens — an easy detail
  to miss that would otherwise corrupt every lag/rolling feature.

## Pipeline

```
data/GoldPrice.csv
      │
      ▼
src/preprocessing.py   load → audit → clean (sort, dedupe, fill/drop, validate)
      │
      ▼
src/features.py        lag / rolling / return / momentum features, leakage-free
      │
      ▼
src/train.py            chronological train / val / test split
      │                  ├─ Naive            (no fitting)
      │                  ├─ ARIMA(5,1,0)      (statsmodels, walk-forward)
      │                  ├─ Random Forest     (TimeSeriesSplit-tuned)
      │                  └─ XGBoost           (TimeSeriesSplit-tuned)
      ▼
src/evaluation.py      MAE / RMSE / R² on the test set only
      │
      ▼
results/                model_comparison.csv, predictions.csv, two plots
```

`notebooks/analysis.ipynb` runs the same pipeline interactively with EDA
and inline plots — it imports from `src/`, it doesn't duplicate it.

## Exploratory data analysis

(Full plots in `notebooks/analysis.ipynb`, section 3.)

- **Trend:** gold moved from ~$1,400 (early 2011) down to a low of $1,071
  (November 2015) and back up to an all-time high of $2,069 (August 2020)
  — a genuine regime range, not a stationary series.
- **Daily moves are small:** the mean absolute daily return is **under
  1% of price**. This single fact drives most of what follows — it means
  yesterday's price is already an extremely strong predictor of today's,
  which sets a high bar for any model to add value.
- **Volatility clusters:** 30-day rolling volatility of returns spikes
  around known stress periods (e.g. the March 2020 COVID crash), then
  decays — classic volatility clustering, consistent with financial
  time series generally.

## Feature engineering

`src/features.py` builds, for each row *t*, only information available
**through day *t*** — nothing from day *t+1* or later ever enters a
feature. Four families:

| Family | Features | Intuition |
|---|---|---|
| **Lag** | `price_lag_{1,2,3,5,7,14}` | Recent price levels — the most direct signal for a near-random-walk series. |
| **Rolling statistics** | `rolling_mean_{7,14,30}`, `rolling_std_{7,14,30}` | Trend (mean) and volatility (std) over recent windows. |
| **Returns** | `return_{1,3,7,14}d` | % change over a trailing window — scale-free, more stationary than raw price. |
| **Momentum** | `momentum_{5,10}` | Absolute price change over a trailing window — is the recent move accelerating? |

**Deliberately excluded: same-day `Open`/`High`/`Low`/`Chg%`.** These
correlate at ~0.98–0.99 with same-day `Price` — including them would let
a model achieve near-perfect-looking accuracy by doing same-day
arithmetic (e.g. inferring `Price` from `Open` + `Chg%`) instead of
actually forecasting the future. This is the single most important
leakage trap in this dataset, and avoiding it is the difference between
an honest pipeline and a misleadingly good-looking one.

The target is `target_next_day_price = Price.shift(-1)`. The final row
(no known next-day value) and warm-up rows without a full lookback window
are dropped: **2,501 of 2,531 rows remain, with 19 features** (including
today's own price).

## Train / validation / test split

Split **chronologically, never shuffled**: the earliest 70% of trading
days are train, the next 15% are validation, and the most recent 15% are
test.

| Split | Rows | Date range |
|---|---:|---|
| Train | 1,750 | 2011-02-14 → 2017-11-09 |
| Validation | 375 | 2017-11-10 → 2019-04-11 |
| Test | 376 | 2019-04-12 → 2020-09-10 |

A runtime assertion (`train.max(Date) < val.min(Date) < test.max(Date)`)
fails loudly if this is ever violated. Why this matters: a random 80/20
split on daily rows would let a model train on days chronologically
*after* some of the days it's tested on — for a forecasting task, that's
not a subtle bug, it's training on the future. Every metric in this repo
comes from the test split only; validation is used only for
cross-validated hyperparameter search context, never for reporting.

## Models

- **Naive** — tomorrow's price = today's price. No fitting, no
  parameters. Not a throwaway baseline: for a near-random-walk series
  it's a genuinely hard target to beat, and it's the sanity check every
  other model has to justify itself against.
- **ARIMA(5,1,0)** (`statsmodels`) — a classical autoregressive model on
  the differenced price series. Fit once on train + validation, then
  walked forward one step at a time through the test set: each true
  value is appended to the model's state (`refit=False`) before the next
  forecast is made, so no test value ever informs a prediction for an
  earlier or the same test day.
- **Random Forest** and **XGBoost** — nonlinear, non-parametric tree
  ensembles that can pick up interaction effects a linear/AR model
  can't. Hyperparameters are tuned with **`TimeSeriesSplit` cross-validation**
  (5 folds, `RandomizedSearchCV`, 12 candidates, scored on RMSE) on the
  **train** split only, then the selected hyperparameters are refit on
  train + validation before predicting the test set. `TimeSeriesSplit`
  is used instead of random K-fold for the same reason as the main
  split: a random fold would validate a model on days it could have
  effectively trained on.

## Evaluation metrics

Computed once, on the test set only, in `src/evaluation.py`:

- **MAE** (mean absolute error) — average dollar error, in the same
  units as gold price. Easiest to explain: "on average, the forecast is
  off by $X."
- **RMSE** (root mean squared error) — like MAE but penalizes large
  errors more; the primary metric used for model selection and ranking
  here.
- **R²** (coefficient of determination) — variance explained relative to
  predicting the mean. See [Findings](#findings-and-interpretation) below
  for why R² is a misleading headline metric on this particular problem.

## Results

Test set only, 376 trading days (2019-04-12 → 2020-09-10). Test-set price
ranged $1,278–$2,069 (mean ≈ $1,604).

| Model | MAE ($) | RMSE ($) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| **Naive (tomorrow = today)** | 13.43 | **19.88** | 0.9879 | 0.83 |
| ARIMA(5,1,0) | 19.67 | 28.22 | 0.9756 | 1.21 |
| XGBoost (tuned) | 28.15 | 50.69 | 0.9214 | 1.60 |
| Random Forest (tuned) | 31.72 | 55.86 | 0.9045 | 1.80 |

Full precision in [`results/model_comparison.csv`](results/model_comparison.csv);
per-day predictions in [`results/predictions.csv`](results/predictions.csv).
MAPE isn't written to the CSV (kept to the three requested metrics there)
but is reproducible from `predictions.csv` in one line of pandas, and is
included above because it directly answers "how accurate is this in
percentage terms" — see below.

## Findings and interpretation

**How accurate is it, really?** In absolute terms, all four models look
good: even the worst (Random Forest) is off by 1.8% of price on average,
and the naive baseline is within 1%. That sounds impressive until you
notice it's true almost by construction — gold's mean daily move is also
under 1%, so a model that just repeats yesterday's price will *always*
land in this range on this kind of series. **R² above 0.90 for every
model here — including the one that does zero learning — is the tell.**
On a near-random-walk series, high R² and low relative error are the
floor, not a sign of skill. The metric that actually discriminates
between these models is the *relative* ranking (which one has lower
error than naive), not the absolute value of any single model's score.

**The naive baseline wins outright**, and none of the more sophisticated
models close the gap. Two concrete reasons, both visible in
[`results/actual_vs_predicted.png`](results/actual_vs_predicted.png):

1. **The series is close to a random walk.** With daily moves this small
   relative to price, "today's price" is already close to an optimal
   one-step predictor under squared error for this kind of series. There
   is mathematically limited room for a model using price history alone
   to improve on it.
2. **The test window includes the 2020 gold rally to all-time highs.**
   Random Forest and XGBoost are tree ensembles — their predictions are
   bounded by the price range observed during training/validation
   (≈$1,071–$1,889 through April 2019). Once price broke well past that
   range in mid-2020 (up to $2,069), both models mechanically undershot the rally,
   because a tree can only output values it saw examples of at leaf
   nodes. This is visible as a clear lag in the plot during the
   July–August 2020 run-up. ARIMA, operating on *differences* rather than
   absolute level, tracks the trend more closely than the tree models
   but still trails naive overall.

Ranked by test RMSE, closest to naive first: **ARIMA > XGBoost > Random
Forest**. ARIMA operates on differences rather than absolute price level,
so it isn't bounded by a training range the way the tree models are —
consistent with it losing the least ground to naive. Between the two tree
models, XGBoost's shrinkage (learning-rate-scaled updates) lets it track
price changes slightly more smoothly than Random Forest's unweighted
tree-averaging, consistent with its smaller error.

A model that can't beat "predict no change" on a near-random-walk asset
isn't a broken pipeline — it's the expected, defensible result for this
problem, and the reasoning above is the actual deliverable: understanding
*why* a result happened is worth more than a headline number, especially
one this easy to produce by accident via leakage.

## Feature importance

[`results/feature_importance.png`](results/feature_importance.png) shows
the top features for XGBoost (the better of the two tree models on this
run, by test RMSE):

| Feature | Importance |
|---|---:|
| `price` (today's close) | 0.69 |
| `rolling_mean_7` | 0.13 |
| `price_lag_3` | 0.10 |
| `price_lag_2` | 0.03 |
| `price_lag_1` | 0.02 |
| `rolling_mean_30` | 0.01 |

`price` alone accounts for roughly two-thirds of total importance, with
the 7-day rolling mean and a 3-day lag making up most of the rest. This
is exactly consistent with the random-walk story above: the model has
correctly learned that today's price is by far the most useful predictor
of tomorrow's — it just can't use that information to extrapolate past
the range it trained on.

## What this project demonstrates

This project is a rigor exercise, not a trading strategy. What it's
meant to show:

- Recognizing and fixing **data leakage** in a time-series feature set
  (same-day OHLC correlated with the target) and in a train/test split
  (random split vs. chronological).
- Choosing a **baseline strong enough to be informative** — a naive
  model that turns out to be very hard to beat, rather than a strawman.
- Correctly reading **R² and relative-error metrics** in a domain where
  they're structurally inflated for *any* model, including one that does
  no learning.
- Using **`TimeSeriesSplit`** for cross-validated hyperparameter tuning
  instead of default K-fold, and knowing why that distinction matters.
- Diagnosing *why* a model underperforms (tree-based extrapolation
  limits) from the shape of the error, not just reporting a number.
- Reporting a negative-ish result (models lose to naive) honestly instead
  of hiding it or quietly reformulating the problem until the metric
  looks better.

## Known limitations

- Tree-based models (Random Forest, XGBoost) predicting price level
  cannot extrapolate beyond the price range seen in training — a
  structural limitation, not a bug, and the main reason they lose to
  naive here. A model that predicted the next-day *return* instead of
  price level and reconstructed price from it would likely close some of
  this gap, at the cost of extra complexity; deliberately out of scope
  for this project.
- Price-history features only — no macro variables (USD index, real
  interest rates, ETF flows), which plausibly matter more for gold's
  medium-term direction than anything derivable from price history
  alone.
- One-step-ahead forecasting only; no multi-day-ahead evaluation.
- A single chronological split, not a walk-forward backtest across
  multiple test windows — results here describe one 376-day period
  (which happens to include an unusual rally), not an average over many
  market regimes.
- No transaction-cost or tradeability analysis — even if a model beat
  naive, that says nothing about whether the edge would survive real
  trading frictions.

## Suggested next steps

1. Reformulate the ML target as next-day *return* and reconstruct price,
   to test whether it removes the tree models' extrapolation ceiling.
2. Add exogenous features (USD index, real yields, ETF holdings) that
   plausibly matter more than price history alone for medium-term
   direction.
3. Walk-forward backtest across multiple rolling test windows instead of
   one fixed split, to see whether the naive-wins result holds across
   different market regimes.
4. Backtest with realistic transaction costs to check whether any
   measured edge would survive them.

## Repository structure

```
GoldCast/
├── data/
│   └── README.md                 # dataset schema + how to obtain it
├── src/
│   ├── preprocessing.py          # load, audit, clean (every step logged)
│   ├── features.py               # leakage-free lag/rolling/return/momentum features
│   ├── models.py                 # Naive, ARIMA, Random Forest, XGBoost + tuning
│   ├── evaluation.py             # MAE / RMSE / R²
│   └── train.py                  # end-to-end pipeline, writes results/
├── notebooks/
│   └── analysis.ipynb            # EDA + walkthrough of the src/ pipeline
├── results/
│   ├── model_comparison.csv      # test-set metrics for all 4 models
│   ├── predictions.csv           # per-day actual vs. predicted, test set
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Running it

```bash
pip install -r requirements.txt
# place your own GoldPrice.csv at data/GoldPrice.csv -- see data/README.md
python src/train.py
```

This regenerates every file in `results/` from scratch (takes under a
minute on a laptop). `notebooks/analysis.ipynb` walks through the same
pipeline interactively, with EDA and inline plots.

## License

MIT — see [`LICENSE`](LICENSE).
