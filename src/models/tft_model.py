"""
Temporal Fusion Transformer for credit spread forecasting.
Uses pytorch-forecasting with multi-variate features.
"""

import json
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightning.pytorch.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")


def load_features(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})
    df["time_idx"] = range(len(df))
    df["group"] = "ig_spread"  # single time series group
    df["risk_regime"] = df["risk_regime"].astype(str)  # categorical
    return df


def build_datasets(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 20,
    test_size: int = 60,
) -> tuple:
    """Build TFT train/val/test datasets."""

    # Continuous features
    time_varying_known_reals = [
        "time_idx",
    ]
    time_varying_unknown_reals = [
        "ig_spread",
        "hy_spread",
        "uk_10y_gilt",
        "us_10y_treasury",
        "us_vix",
        "lloyds_price",
        "barclays_price",
        "spread_1d_change",
        "spread_5d_change",
        "spread_21d_ma",
        "spread_zscore",
        "yield_curve_slope",
    ]
    time_varying_known_categoricals = ["risk_regime"]

    training_cutoff = df["time_idx"].max() - test_size

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="ig_spread",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= training_cutoff + max_prediction_length],
        predict=True,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=32, num_workers=0
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=32, num_workers=0
    )

    return training, validation, train_loader, val_loader, training_cutoff


def train_tft(
    training: TimeSeriesDataSet,
    train_loader,
    val_loader,
    max_epochs: int = 30,
) -> tuple:
    """Train the TFT model."""

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    print(f"TFT parameters: {sum(p.numel() for p in tft.parameters()):,}")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    checkpoint = ModelCheckpoint(
        dirpath="data/processed/tft_checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        logger=TensorBoardLogger(
            save_dir="data/processed/tft_logs",
            name="tft",
        ),
    )

    trainer.fit(tft, train_loader, val_loader)

    # Load best model
    best_model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint.best_model_path
    )

    return best_model, trainer


def evaluate_tft(
    model: TemporalFusionTransformer,
    val_loader,
    df: pd.DataFrame,
    training_cutoff: int,
    max_prediction_length: int = 20,
) -> dict:
    """Generate predictions and evaluate."""

    predictions = model.predict(val_loader, return_y=True)

    # Output shape: [batch, horizon] — point predictions
    pred_values = predictions.output.flatten().numpy()
    actual = predictions.y[0].flatten().numpy()

    # Align lengths
    min_len = min(len(actual), len(pred_values))
    actual = actual[:min_len]
    pred_values = pred_values[:min_len]

    mae = mean_absolute_error(actual, pred_values)
    rmse = np.sqrt(mean_squared_error(actual, pred_values))
    mape = np.mean(np.abs((actual - pred_values) / (actual + 1e-8))) * 100

    # Generate uncertainty intervals via quantile prediction
    quantile_preds = model.predict(
        val_loader,
        mode="quantiles",
        return_y=False,
    )
    # quantile_preds shape: [batch, horizon, n_quantiles]
    lower = quantile_preds[:, :, 0].flatten().numpy()[:min_len]
    upper = quantile_preds[:, :, -1].flatten().numpy()[:min_len]
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "coverage_90pct": round(coverage, 4),
    }

    print(f"  MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | "
          f"MAPE: {metrics['mape']}% | Coverage: {metrics['coverage_90pct']:.1%}")

    return {
        "metrics": metrics,
        "predictions": pred_values.tolist(),
        "actual": actual.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }


def run_tft() -> dict:
    """Full TFT training and evaluation pipeline."""
    mlflow.set_experiment("marketpulse-forecasting")

    print("=" * 50)
    print("MarketPulse — Temporal Fusion Transformer")
    print("=" * 50)

    df = load_features()
    print(f"Dataset: {len(df)} timesteps, {len(df.columns)} features")

    training, validation, train_loader, val_loader, cutoff = build_datasets(df)
    print(f"Training cutoff: time_idx={cutoff}")

    with mlflow.start_run(run_name="tft"):
        mlflow.log_param("model", "TemporalFusionTransformer")
        mlflow.log_param("hidden_size", 32)
        mlflow.log_param("max_encoder_length", 60)
        mlflow.log_param("max_prediction_length", 20)

        print("\nTraining TFT...")
        model, trainer = train_tft(training, train_loader, val_loader)

        print("\nEvaluating TFT...")
        results = evaluate_tft(model, val_loader, df, cutoff)

        mlflow.log_metrics(results["metrics"])

        # Save results
        with open("data/processed/tft_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("TFT Results")
    print("=" * 50)
    m = results["metrics"]
    print(f"MAE:      {m['mae']} (ARIMA baseline: 0.0441)")
    print(f"RMSE:     {m['rmse']} (ARIMA baseline: 0.0604)")
    print(f"MAPE:     {m['mape']}% (ARIMA baseline: 5.04%)")
    print(f"Coverage: {m['coverage_90pct']:.1%} (target: ~90%)")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_tft()