"""
SHAP explainability for MarketPulse credit spread forecasting.
Answers: which macro features drive credit spread predictions most?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare_supervised(
    df: pd.DataFrame,
    target: str = "ig_spread",
    lags: int = 20,
) -> tuple:
    """Same supervised format as conformal module."""
    feature_cols = [
        "ig_spread", "hy_spread", "us_vix",
        "spread_1d_change", "spread_5d_change",
        "spread_21d_ma", "spread_zscore",
        "yield_curve_slope", "risk_regime",
    ]

    data = df[feature_cols].copy()

    lagged = []
    for lag in range(1, lags + 1):
        shifted = data.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in data.columns]
        lagged.append(shifted)

    X = pd.concat(lagged, axis=1).dropna()
    y = df[target].loc[X.index]

    return X, y


def get_feature_groups(feature_names: list) -> dict:
    """
    Group lagged features by their base name.
    Returns mean absolute SHAP per feature group.
    """
    groups = {}
    for name in feature_names:
        base = name.rsplit("_lag", 1)[0]
        if base not in groups:
            groups[base] = []
        groups[base].append(name)
    return groups


def run_shap_analysis(
    test_size: int = 60,
    output_dir: str = "data/processed/shap",
) -> None:
    """
    Run SHAP analysis on the Ridge conformal model.
    Generates:
    1. Global feature importance (bar chart)
    2. SHAP summary plot (beeswarm)
    3. Feature group importance (grouped by macro factor)
    4. JSON results for dashboard integration
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("MarketPulse — SHAP Explainability")
    print("=" * 50)

    df = load_data()
    X, y = prepare_supervised(df, lags=20)

    # Same split as conformal
    n = len(X)
    X_train = X.iloc[:n - test_size - 100]
    y_train = y.iloc[:n - test_size - 100]
    X_test = X.iloc[n - test_size:]
    y_test = y.iloc[n - test_size:]

    # Fit Ridge model
    print("\nFitting Ridge model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)

    # SHAP explainer — use LinearExplainer for Ridge
    print("Computing SHAP values...")
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.LinearExplainer(
        ridge,
        X_train_scaled,
        feature_names=X.columns.tolist(),
    )
    shap_values = explainer(X_test_scaled)

    # --- Plot 1: Global feature importance (top 20) ---
    print("Generating global importance plot...")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color="#00FF7F",
        alpha=0.8,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        "Top 20 Features Driving Credit Spread Predictions",
        fontsize=14, fontweight="bold",
    )
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/shap_importance.png")

    # --- Plot 2: SHAP summary beeswarm ---
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary — Credit Spread Drivers", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/shap_summary.png")

    # --- Plot 3: Feature group importance ---
    print("Generating feature group importance...")
    groups = get_feature_groups(X.columns.tolist())
    group_importance = {}
    for group, features in groups.items():
        indices = [X.columns.tolist().index(f) for f in features]
        group_importance[group] = float(
            np.abs(shap_values.values[:, indices]).mean()
        )

    group_df = pd.DataFrame([
        {"feature_group": k, "importance": v}
        for k, v in group_importance.items()
    ]).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#00FF7F" if i == 0 else "#7B68EE" if i < 3 else "#444"
              for i in range(len(group_df))]
    ax.barh(
        group_df["feature_group"][::-1],
        group_df["importance"][::-1],
        color=colors[::-1],
        alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        "Macro Factor Importance for Credit Spread Forecasting",
        fontsize=14, fontweight="bold",
    )
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_groups.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/shap_groups.png")

    # --- Save JSON for dashboard ---
    results = {
        "top_features": importance_df.to_dict(orient="records"),
        "feature_groups": group_df.to_dict(orient="records"),
        "n_test_samples": len(X_test),
    }
    with open(f"{output_dir}/shap_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Print summary ---
    print("\n" + "=" * 50)
    print("Top 10 Credit Spread Drivers")
    print("=" * 50)
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:<35} {bar} {row['importance']:.4f}")

    print("\nMacro Factor Group Importance:")
    print("=" * 50)
    for _, row in group_df.iterrows():
        bar = "█" * int(row["importance"] / group_df["importance"].max() * 30)
        print(f"  {row['feature_group']:<25} {bar} {row['importance']:.4f}")


if __name__ == "__main__":
    run_shap_analysis()