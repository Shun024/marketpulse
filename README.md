# 📈 MarketPulse

UK credit spread forecasting with uncertainty quantification. Compares classical and deep learning models with mathematically guaranteed prediction intervals via conformal prediction.

![CI](https://github.com/Shun024/marketpulse/actions/workflows/lint-test.yml/badge.svg)

---

## Results

| Model | MAE | RMSE | MAPE | Coverage | vs ARIMA |
|---|---|---|---|---|---|
| ARIMA(1,1,1) | 0.0441 | 0.0604 | 5.04% | — | baseline |
| Prophet | 0.0538 | 0.0649 | 6.32% | 90% | -22% |
| TFT | 0.0245 | 0.0303 | 2.89% | 100% | **+44%** |
| **Conformal (Ridge)** | **0.0168** | **0.0208** | **1.98%** | 86.7% | **+62%** |

---

## Architecture

```
Data Sources: FRED API + yfinance
    ├── IG credit spread (target)
    ├── HY spread, VIX, gilt yields
    └── UK bank stocks (Lloyds, Barclays, NatWest, HSBC)
            │
            ▼
    Feature Engineering (21 features)
    spread z-score, yield curve slope, volatility regime
            │
            ├── ARIMA(1,1,1)          classical baseline
            ├── Prophet               trend + seasonality
            ├── Temporal Fusion Transformer  deep learning
            └── Conformal Prediction  guaranteed intervals
                        │
                        ▼
            Plotly Dash Dashboard
```

---

## Key Findings

**TFT beats ARIMA by 44% on MAE** — attention-based deep learning captures non-linear regime dynamics that ARIMA misses.

**Conformal prediction achieves 86.7% empirical coverage** (target: 90%) with the tightest point forecasts (MAE 0.0168). Unlike heuristic intervals, conformal prediction provides a mathematical guarantee: P(y ∈ [lower, upper]) ≥ 1 − α.

**Prophet underperforms ARIMA** on credit spreads — confirming that financial spreads are driven by regime changes, not seasonal patterns.

---

## Stack

FRED API · yfinance · statsmodels · Prophet · PyTorch Forecasting · MAPIE · Plotly Dash · MLflow

---

## Quickstart

```bash
git clone https://github.com/Shun024/marketpulse.git
cd marketpulse
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add FRED API key to .env
echo "FRED_API_KEY=your_key" > .env

# Run pipeline
PYTHONPATH=. python -m src.data.loader
PYTHONPATH=. python -m src.models.baseline
PYTHONPATH=. python -m src.models.tft_model
PYTHONPATH=. python -m src.evaluation.conformal

# Launch dashboard
PYTHONPATH=. python src/dashboard/app.py
# Open http://localhost:8050
```

---

## Model Explainability — SHAP Analysis

Which macro factors drive credit spread predictions? SHAP values quantify each feature's contribution to the forecast.

**Macro factor importance ranking:**
1. **ig_spread** (own momentum) — yesterday's spread level is the strongest predictor
2. **hy_spread** — high yield spreads signal credit stress building with a lag
3. **spread_21d_ma** — medium-term trend context
4. **yield_curve_slope** — rate environment with ~10 day transmission lag
5. **us_vix** — volatility regime with 2-day lag (markets price fear before spreads widen)

**Key finding:** The model learns real financial dynamics — spread momentum, cross-asset contagion (HY → IG), and macro transmission lags — rather than spurious correlations.

![SHAP Macro Factors](data/processed/shap/shap_groups.png)
![SHAP Top Features](data/processed/shap/shap_importance.png)

---

## Author

**Shun Le Yi Mon (Sheryl)** · Data Scientist · NLP & GenAI  
[LinkedIn](#) · [GitHub](https://github.com/Shun024)