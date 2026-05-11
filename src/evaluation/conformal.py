"""
Conformal Prediction for credit spread forecasting.
Provides distribution-free uncertainty intervals with coverage guarantees.

Unlike heuristic intervals, conformal prediction GUARANTEES:
P(y_true in [lower, upper]) >= 1 - alpha
where alpha = 0.1 for 90% coverage.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare_supervised(
    df: pd.DataFrame,
    target: str = "ig_spread",
    lags: int = 20,
) -> tuple:
    """
    Convert time series to supervised learning format.
    Uses lagged features as predictors.
    """
    feature_cols = [
        "ig_spread", "hy_spread", "us_vix",
        "spread_1d_change", "spread_5d_change",
        "spread_21d_ma", "spread_zscore",
        "yield_curve_slope", "risk_regime",
    ]

    data = df[feature_cols].copy()

    # Create lagged features
    lagged = []
    for lag in range(1, lags + 1):
        shifted = data.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in data.columns]
        lagged.append(shifted)

    X = pd.concat(lagged, axis=1).dropna()
    y = df[target].loc[X.index]

    return X, y


def run_conformal_prediction(
    alpha: float = 0.1,
    test_size: int = 60,
    calibration_size: int = 100,
) -> dict:
    """
    Run conformal prediction with MAPIE.

    Split strategy:
    - Train: fit base model
    - Calibration: compute nonconformity scores
    - Test: generate guaranteed intervals
    """
    print("=" * 50)
    print("MarketPulse — Conformal Prediction")
    print(f"Target coverage: {(1-alpha)*100:.0f}%")
    print("=" * 50)

    df = load_data()
    X, y = prepare_supervised(df, lags=20)

    # Three-way split: train / calibration / test
    n = len(X)
    test_end = n
    test_start = n - test_size
    cal_start = test_start - calibration_size
    train_end = cal_start

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_cal = X.iloc[cal_start:test_start]
    y_cal = y.iloc[cal_start:test_start]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    print(f"\nTrain: {len(X_train)} | Calibration: {len(X_cal)} | Test: {len(X_test)}")

    # Base model: Ridge regression in a pipeline
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    # Base model: Ridge regression in a pipeline
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    # Fit base model on train
    print("\nFitting base model...")
    base_model.fit(X_train, y_train)

    # SplitConformalRegressor — uses calibration set for coverage guarantee
    print("Calibrating conformal intervals...")
    mapie = SplitConformalRegressor(
        estimator=base_model,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)

    # Generate predictions with guaranteed intervals
    print("Generating conformal intervals on test set...")
    result = mapie.predict(X_test)
    y_pred, pi = mapie.predict_interval(X_test)
    lower = pi[:, 0, 0]
    upper = pi[:, 1, 0]
    intervals = mapie.predict_interval(X_test)
    print(type(intervals))
    print(len(intervals))
    if isinstance(intervals, tuple):
        for i, v in enumerate(intervals):
            print(f"  [{i}] type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")


    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-8))) * 100
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
    interval_width = float(np.mean(upper - lower))

    print(f"\nConformal Prediction Results:")
    print(f"  MAE:            {mae:.4f}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  MAPE:           {mape:.4f}%")
    print(f"  Coverage:       {coverage:.1%} (target: {(1-alpha)*100:.0f}%)")
    print(f"  Interval width: {interval_width:.4f}")

    results = {
        "method": "Conformal Prediction (MAPIE)",
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "metrics": {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mape": round(float(mape), 4),
            "coverage": round(float(coverage), 4),
            "interval_width": round(float(interval_width), 4),
        },
        "test_dates": [str(d.date()) for d in y_test.index],
        "actual": y_test.values.tolist(),
        "predictions": y_pred.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/conformal_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Full Model Comparison")
    print("=" * 50)
    print(f"{'Model':<25} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Coverage':<10}")
    print("-" * 59)
    print(f"{'ARIMA(1,1,1)':<25} {'0.0441':<8} {'0.0604':<8} {'5.04%':<8} {'—':<10}")
    print(f"{'Prophet':<25} {'0.0538':<8} {'0.0649':<8} {'6.32%':<8} {'90%':<10}")
    print(f"{'TFT':<25} {'0.0245':<8} {'0.0303':<8} {'2.89%':<8} {'100%':<10}")
    print(f"{'Conformal (Ridge)':<25} {mae:<8.4f} {rmse:<8.4f} {mape:<8.2f}% {coverage:<10.1%}")

    return results


if __name__ == "__main__":
    run_conformal_prediction()