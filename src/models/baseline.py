"""
Classical forecasting baselines: ARIMA and Prophet.
Target: IG credit spread (BAMLC0A0CM)
"""

import warnings
import json
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


def load_spread_series(path: str = "data/processed/features.parquet") -> pd.Series:
    """Load the IG spread series for univariate forecasting."""
    df = pd.read_parquet(path)
    series = df["ig_spread"].copy()
    series.index = pd.DatetimeIndex(series.index).to_period("B").to_timestamp()
    return series


def check_stationarity(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
    }


def train_test_split_ts(
    series: pd.Series,
    test_size: int = 60,
) -> tuple[pd.Series, pd.Series]:
    """Split time series — last 60 days as test (no shuffling)."""
    return series.iloc[:-test_size], series.iloc[-test_size:]


def evaluate(actual: pd.Series, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual.values - predicted) / actual.values)) * 100
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }


def fit_arima(train: pd.Series, test: pd.Series) -> dict:
    """
    Fit ARIMA model with automatic order selection.
    Uses ARIMA(1,1,1) as default — appropriate for financial spreads.
    """
    print("Fitting ARIMA(1,1,1)...")

    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit()

    # Forecast
    forecast = fitted.forecast(steps=len(test))
    forecast = np.maximum(forecast, 0)  # spreads can't be negative

    metrics = evaluate(test, forecast)
    print(f"  MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | MAPE: {metrics['mape']}%")

    return {
        "model": "ARIMA(1,1,1)",
        "metrics": metrics,
        "forecast": forecast,
        "fitted_model": fitted,
    }


def fit_prophet(train: pd.Series, test: pd.Series) -> dict:
    """
    Fit Facebook Prophet model.
    Prophet handles regime changes and seasonality well.
    """
    print("Fitting Prophet...")

    # Prophet requires specific column names
    prophet_df = pd.DataFrame({
        "ds": train.index,
        "y": train.values,
    })

    model = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.9,
    )
    model.fit(prophet_df)

    # Forecast
    future = pd.DataFrame({"ds": test.index})
    forecast_df = model.predict(future)
    forecast = np.maximum(forecast_df["yhat"].values, 0)

    # Prophet uncertainty intervals
    lower = forecast_df["yhat_lower"].values
    upper = forecast_df["yhat_upper"].values

    metrics = evaluate(test, forecast)
    print(f"  MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | MAPE: {metrics['mape']}%")

    return {
        "model": "Prophet",
        "metrics": metrics,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fitted_model": model,
    }


def run_baselines() -> dict:
    """Run all baseline models and log to MLflow."""
    mlflow.set_experiment("marketpulse-forecasting")

    print("=" * 50)
    print("MarketPulse — Classical Baseline Models")
    print("=" * 50)

    series = load_spread_series()
    train, test = train_test_split_ts(series, test_size=60)

    print(f"\nTrain: {len(train)} days | Test: {len(test)} days")
    print(f"Train range: {train.index[0].date()} to {train.index[-1].date()}")
    print(f"Test range: {test.index[0].date()} to {test.index[-1].date()}")

    # Stationarity check
    stat = check_stationarity(train)
    print(f"\nStationarity (ADF): p={stat['p_value']} | "
          f"{'Stationary' if stat['is_stationary'] else 'Non-stationary'}")

    results = {}

    # ARIMA
    with mlflow.start_run(run_name="arima"):
        mlflow.log_param("model", "ARIMA(1,1,1)")
        mlflow.log_param("test_size", len(test))
        arima_result = fit_arima(train, test)
        mlflow.log_metrics(arima_result["metrics"])
        results["arima"] = arima_result

    # Prophet
    with mlflow.start_run(run_name="prophet"):
        mlflow.log_param("model", "Prophet")
        mlflow.log_param("test_size", len(test))
        prophet_result = fit_prophet(train, test)
        mlflow.log_metrics(prophet_result["metrics"])
        results["prophet"] = prophet_result

    # Save results
    output = {
        "train_size": len(train),
        "test_size": len(test),
        "arima": arima_result["metrics"],
        "prophet": prophet_result["metrics"],
        "test_dates": [str(d.date()) for d in test.index],
        "test_actual": test.values.tolist(),
        "arima_forecast": arima_result["forecast"].tolist(),
        "prophet_forecast": prophet_result["forecast"].tolist(),
        "prophet_lower": prophet_result["lower"].tolist(),
        "prophet_upper": prophet_result["upper"].tolist(),
    }

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 50)
    print("Baseline Results Summary")
    print("=" * 50)
    print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 45)
    for name, result in results.items():
        m = result["metrics"]
        print(f"{result['model']:<15} {m['mae']:<10} {m['rmse']:<10} {m['mape']:<10}")

    return results


if __name__ == "__main__":
    run_baselines()