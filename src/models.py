"""
models.py

Naive baseline, ARIMA, Random Forest, and XGBoost for next-day gold price
forecasting. Random Forest and XGBoost are tuned with TimeSeriesSplit
cross-validation only -- never random K-fold, which would let a fold train
on days chronologically after the fold it's validated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def naive_forecast(price_today: pd.Series) -> np.ndarray:
    """Predict tomorrow's price = today's price."""
    return price_today.to_numpy()


# ---------------------------------------------------------------------------
# ARIMA -- fit once, then walk-forward one-step-ahead through the test set
# ---------------------------------------------------------------------------

def fit_predict_arima(history_prices: np.ndarray, test_prices: np.ndarray,
                       order: tuple[int, int, int] = (5, 1, 0)) -> np.ndarray:
    """
    Fit ARIMA(order) on `history_prices` (train + validation), then produce
    one-step-ahead forecasts across `test_prices` by appending each true
    observation to the model's state as it becomes "known" (walk-forward
    validation, no refitting of parameters -> fast, and never uses a test
    value to predict itself or an earlier test day).
    """
    model = ARIMA(list(history_prices), order=order).fit()
    preds = []
    for actual in test_prices:
        preds.append(float(model.forecast(steps=1)[0]))
        model = model.append([actual], refit=False)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Random Forest / XGBoost + lightweight TimeSeriesSplit tuning
# ---------------------------------------------------------------------------

def build_random_forest(**kwargs) -> RandomForestRegressor:
    params = dict(random_state=RANDOM_STATE, n_jobs=-1)
    params.update(kwargs)
    return RandomForestRegressor(**params)


def build_xgboost(**kwargs) -> XGBRegressor:
    params = dict(random_state=RANDOM_STATE, n_jobs=-1)
    params.update(kwargs)
    return XGBRegressor(**params)


def rf_param_search_space() -> dict:
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.5, 1.0],
    }


def xgb_param_search_space() -> dict:
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
    }


def tune_with_timeseries_cv(base_estimator, param_distributions: dict,
                             X: pd.DataFrame, y: pd.Series,
                             n_splits: int = 5, n_iter: int = 12):
    """
    Lightweight randomized hyperparameter search scored with TimeSeriesSplit
    cross-validation (every validation fold is strictly after the training
    fold it's scored on). Returns (best_estimator, best_params, best_cv_rmse).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        base_estimator, param_distributions, n_iter=n_iter, cv=tscv,
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_
