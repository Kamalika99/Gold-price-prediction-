"""
train.py

End-to-end pipeline: load -> clean -> engineer features -> chronological
train/validation/test split -> fit Naive, ARIMA, Random Forest, and XGBoost
-> evaluate on the held-out test set only -> save results.

The split is strictly chronological (no shuffling): the earliest 70% of
trading days are train, the next 15% are validation (used only for
TimeSeriesSplit-based hyperparameter tuning context), and the most recent
15% are test. Test-set metrics are the only numbers reported as "results" --
nothing is tuned or selected using the test set.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_raw_data, audit_dataset, clean_dataset
from features import build_features
from models import (
    naive_forecast, fit_predict_arima,
    build_random_forest, build_xgboost,
    rf_param_search_space, xgb_param_search_space,
    tune_with_timeseries_cv, RANDOM_STATE,
)
from evaluation import compute_metrics, build_results_table

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "GoldPrice.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # remaining 0.15 is test
ARIMA_ORDER = (5, 1, 0)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load + audit ---------------------------------------------------------
    raw = load_raw_data(DATA_PATH)
    audit = audit_dataset(raw)
    print("=== RAW DATA AUDIT ===")
    for k, v in audit.items():
        print(f"{k}: {v}")

    # 2. Clean ------------------------------------------------------------------
    cleaned, clean_log = clean_dataset(raw)
    print("\n=== CLEANING LOG ===")
    for k, v in clean_log.items():
        print(f"{k}: {v}")

    # 3. Feature engineering ------------------------------------------------------
    feat, feat_log = build_features(cleaned)
    feature_cols = feat_log["final_feature_columns"]
    print("\n=== FEATURE ENGINEERING LOG ===")
    for k, v in feat_log.items():
        print(f"{k}: {v}")
    print(f"Final feature count: {len(feature_cols)}")

    # 4. Chronological train / validation / test split ---------------------------
    n = len(feat)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))

    train_df = feat.iloc[:train_end].reset_index(drop=True)
    val_df = feat.iloc[train_end:val_end].reset_index(drop=True)
    test_df = feat.iloc[val_end:].reset_index(drop=True)
    trainval_df = feat.iloc[:val_end].reset_index(drop=True)

    assert train_df["Date"].max() < val_df["Date"].min(), "Chronological split violated (train/val)!"
    assert val_df["Date"].max() < test_df["Date"].min(), "Chronological split violated (val/test)!"

    print("\n=== SPLIT (chronological) ===")
    print(f"Train: {len(train_df):4d} rows  {train_df['Date'].min().date()} -> {train_df['Date'].max().date()}")
    print(f"Val:   {len(val_df):4d} rows  {val_df['Date'].min().date()} -> {val_df['Date'].max().date()}")
    print(f"Test:  {len(test_df):4d} rows  {test_df['Date'].min().date()} -> {test_df['Date'].max().date()}")

    X_train, y_train = train_df[feature_cols], train_df["target_next_day_price"]
    X_trainval, y_trainval = trainval_df[feature_cols], trainval_df["target_next_day_price"]
    X_test, y_test = test_df[feature_cols], test_df["target_next_day_price"]

    results = {}
    predictions = pd.DataFrame({"Date": test_df["Date"], "Actual": y_test.reset_index(drop=True)})

    # 5. Naive baseline: tomorrow's price = today's price -------------------------
    naive_preds = naive_forecast(test_df["price"])
    results["Naive"] = compute_metrics(y_test, naive_preds)
    predictions["Naive"] = naive_preds

    # 6. ARIMA: fit on train+val, walk-forward one-step-ahead through test --------
    print("\nFitting ARIMA(5,1,0) and walking forward through the test set...")
    arima_preds = fit_predict_arima(trainval_df["price"].to_numpy(), test_df["price"].to_numpy(), order=ARIMA_ORDER)
    results["ARIMA"] = compute_metrics(y_test, arima_preds)
    predictions["ARIMA"] = arima_preds

    # 7. Random Forest: tune on train (TimeSeriesSplit CV), refit on train+val ----
    print("Tuning Random Forest (TimeSeriesSplit, 5 folds, 12 candidates)...")
    _, rf_best_params, rf_best_cv_rmse = tune_with_timeseries_cv(
        build_random_forest(), rf_param_search_space(), X_train, y_train)
    rf_final = build_random_forest(**rf_best_params)
    rf_final.fit(X_trainval, y_trainval)
    rf_preds = rf_final.predict(X_test)
    results["Random Forest"] = compute_metrics(y_test, rf_preds)
    predictions["RandomForest"] = rf_preds
    print(f"Random Forest best params: {rf_best_params} | CV RMSE: {rf_best_cv_rmse:.4f}")

    # 8. XGBoost: tune on train (TimeSeriesSplit CV), refit on train+val ----------
    print("Tuning XGBoost (TimeSeriesSplit, 5 folds, 12 candidates)...")
    _, xgb_best_params, xgb_best_cv_rmse = tune_with_timeseries_cv(
        build_xgboost(), xgb_param_search_space(), X_train, y_train)
    xgb_final = build_xgboost(**xgb_best_params)
    xgb_final.fit(X_trainval, y_trainval)
    xgb_preds = xgb_final.predict(X_test)
    results["XGBoost"] = compute_metrics(y_test, xgb_preds)
    predictions["XGBoost"] = xgb_preds
    print(f"XGBoost best params: {xgb_best_params} | CV RMSE: {xgb_best_cv_rmse:.4f}")

    # 9. Results table (test set only) --------------------------------------------
    results_table = build_results_table(results)
    print("\n=== RESULTS (test set, sorted by RMSE) ===")
    print(results_table.to_string())

    # 10. Feature importance -- from whichever tree model scored lower test RMSE ---
    best_tree = "Random Forest" if results["Random Forest"]["RMSE"] <= results["XGBoost"]["RMSE"] else "XGBoost"
    best_tree_model = rf_final if best_tree == "Random Forest" else xgb_final
    importances = pd.Series(best_tree_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # 11. Save results --------------------------------------------------------------
    results_table.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    predictions.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_df["Date"], y_test.values, label="Actual", color="black", linewidth=1.6)
    ax.plot(test_df["Date"], rf_preds, label="Random Forest", linewidth=1.0, alpha=0.85)
    ax.plot(test_df["Date"], xgb_preds, label="XGBoost", linewidth=1.0, alpha=0.85)
    ax.plot(test_df["Date"], naive_preds, label="Naive", linewidth=1.0, alpha=0.6, linestyle="--")
    ax.set_title("Actual vs Predicted Next-Day Gold Price (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    importances.head(15).sort_values().plot(kind="barh", ax=ax, color="goldenrod")
    ax.set_title(f"Feature Importance -- {best_tree} (top 15)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll results written to {RESULTS_DIR}/")
    return results_table


if __name__ == "__main__":
    main()
