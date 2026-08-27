# Data

This project expects a single CSV at `data/GoldPrice.csv` with daily gold
price history. It is **not committed to the repository** (see
`.gitignore`) to keep the repo minimal — place your own copy at that path
before running `src/train.py` or the notebook.

## Expected schema

| Column | Type   | Description                                    |
|--------|--------|-------------------------------------------------|
| Date   | string | Trading date, e.g. `Sep 11, 2020`               |
| Price  | float  | Closing price (USD)                             |
| Open   | float  | Opening price (USD)                             |
| High   | float  | Session high (USD)                              |
| Low    | float  | Session low (USD)                               |
| Chg%   | float  | Day-over-day % change, as a decimal (e.g. 0.0048) |

This is the standard daily-history export format used by financial data
sites such as Investing.com. Only `Date` and `Price` are used as modeling
inputs (see `src/features.py`) — `Open`/`High`/`Low`/`Chg%` are read and
audited but deliberately excluded from the feature set because they are
same-day values that would leak information about the same day's `Price`.

## Dataset used for the results in this repo

- 2,531 rows, 2011-01-03 to 2020-09-11, no missing values or duplicate
  dates (see the audit output in `src/preprocessing.py` / `notebooks/analysis.ipynb`).
- The raw file ships newest-first; `src/preprocessing.py` re-sorts it
  chronologically before anything else happens.
