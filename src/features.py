"""
features.py

Builds a leakage-free feature set for next-day gold price forecasting.

Task definition:
    Given information available up to and including trading day t,
    predict the closing Price on trading day t+1.

Every feature at row t uses only data from day t or earlier -- same-day
Open/High/Low/Chg% are deliberately excluded, since they are ~0.98-0.99
correlated with same-day Price and would let a model "cheat" by doing
same-day arithmetic instead of forecasting. The target at row t is
Price.shift(-1) (tomorrow's price); the last row has no known future day
and is dropped from the supervised dataset, along with warm-up rows that
don't yet have a full lookback window.

Four feature families, matching the project's methodology:
    1. Lag features       -- price N days ago
    2. Rolling statistics -- trailing mean / std (volatility)
    3. Return features    -- % change over a trailing window
    4. Momentum features  -- absolute price change over a trailing window
"""

from __future__ import annotations

import pandas as pd

LAG_WINDOWS = [1, 2, 3, 5, 7, 14]
ROLLING_WINDOWS = [7, 14, 30]
RETURN_WINDOWS = [1, 3, 7, 14]
MOMENTUM_WINDOWS = [5, 10]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Parameters
    ----------
    df : cleaned, chronologically sorted DataFrame with a Price column.

    Returns
    -------
    (feature_df, log) where feature_df has one row per usable training
    example (features at t, target = Price at t+1), and log documents how
    many rows were consumed by warm-up windows / the missing final target.
    """
    data = df.sort_values("Date").reset_index(drop=True).copy()
    log = {"rows_before_feature_engineering": int(len(data))}

    feat = pd.DataFrame(index=data.index)
    feat["Date"] = data["Date"]

    # Current-day price -- known at prediction time t.
    feat["price"] = data["Price"]

    # 1. Lag features: Price k days before t.
    for k in LAG_WINDOWS:
        feat[f"price_lag_{k}"] = data["Price"].shift(k)

    # 2. Rolling statistics: causal window ending at t (inclusive).
    for w in ROLLING_WINDOWS:
        feat[f"rolling_mean_{w}"] = data["Price"].rolling(w).mean()
        feat[f"rolling_std_{w}"] = data["Price"].rolling(w).std()

    # 3. Return features: % change over a window ending at t.
    for k in RETURN_WINDOWS:
        feat[f"return_{k}d"] = data["Price"].pct_change(k)

    # 4. Momentum features: absolute price change over a window ending at t.
    for k in MOMENTUM_WINDOWS:
        feat[f"momentum_{k}"] = data["Price"] - data["Price"].shift(k)

    # Target: next trading day's Price.
    feat["target_next_day_price"] = data["Price"].shift(-1)

    feature_cols = [c for c in feat.columns if c not in ("Date", "target_next_day_price")]

    rows_with_target = feat["target_next_day_price"].notna().sum()
    log["rows_dropped_no_target"] = int(len(feat) - rows_with_target)

    before_warmup_drop = len(feat)
    feat = feat.dropna(subset=feature_cols + ["target_next_day_price"]).reset_index(drop=True)
    log["rows_dropped_insufficient_history"] = int(before_warmup_drop - len(feat) - log["rows_dropped_no_target"])
    log["rows_after_feature_engineering"] = int(len(feat))
    log["final_feature_columns"] = feature_cols

    return feat, log


if __name__ == "__main__":
    from preprocessing import load_raw_data, clean_dataset

    raw = load_raw_data("data/GoldPrice.csv")
    cleaned, _ = clean_dataset(raw)
    feat, log = build_features(cleaned)
    print(feat.head())
    print("\nFeature engineering log:")
    for k, v in log.items():
        print(f"{k}: {v}")
