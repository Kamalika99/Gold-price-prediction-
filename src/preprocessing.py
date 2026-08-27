"""
preprocessing.py

Loads the raw GoldPrice.csv file, audits it, and produces a cleaned,
chronologically-sorted DataFrame ready for feature engineering.

Every number this module reports is computed directly from the data --
nothing here is hard-coded, and no cleaning step happens silently.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Chg%"]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected columns missing from dataset: {missing_cols}")
    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """Compute a factual audit of the raw dataset. Nothing here is assumed."""
    dates = pd.to_datetime(df["Date"])
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_dates": int(dates.duplicated().sum()),
        "min_date": str(dates.min().date()),
        "max_date": str(dates.max().date()),
        "sorted_ascending_as_loaded": bool(dates.is_monotonic_increasing),
        "rows_with_nonpositive_price": int((df["Price"] <= 0).sum()),
    }


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Deterministic cleaning pipeline. Every operation performed here is
    reported in the returned log dict -- no row is silently dropped.
    """
    log = {"rows_before": int(len(df))}
    cleaned = df.copy()

    # 1. Parse dates and sort chronologically (oldest -> newest). The raw
    #    file ships newest-first, so this step matters.
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    # 2. Duplicate dates -- keep the first occurrence.
    dup_dates = int(cleaned["Date"].duplicated().sum())
    log["duplicate_dates_removed"] = dup_dates
    if dup_dates:
        cleaned = cleaned.drop_duplicates(subset="Date", keep="first").reset_index(drop=True)

    # 3. Full-row duplicates.
    dup_rows = int(cleaned.duplicated().sum())
    log["duplicate_rows_removed"] = dup_rows
    if dup_rows:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 4. Missing values -- forward-fill (uses only past information), then
    #    drop any row that still can't be filled (e.g. missing at the start).
    missing_total = int(cleaned[REQUIRED_COLUMNS].isna().sum().sum())
    log["missing_values_found"] = missing_total
    if missing_total:
        cleaned[REQUIRED_COLUMNS[1:]] = cleaned[REQUIRED_COLUMNS[1:]].ffill()
        cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    # 5. Impossible values -- price must be positive.
    bad_price = int((cleaned["Price"] <= 0).sum())
    log["nonpositive_price_rows_removed"] = bad_price
    if bad_price:
        cleaned = cleaned.loc[cleaned["Price"] > 0].reset_index(drop=True)

    cleaned = cleaned.sort_values("Date").reset_index(drop=True)
    log["rows_after"] = int(len(cleaned))
    log["total_rows_removed"] = log["rows_before"] - log["rows_after"]
    return cleaned, log


if __name__ == "__main__":
    raw = load_raw_data("data/GoldPrice.csv")
    for k, v in audit_dataset(raw).items():
        print(f"{k}: {v}")
    _, log = clean_dataset(raw)
    print("\nCleaning log:")
    for k, v in log.items():
        print(f"{k}: {v}")
