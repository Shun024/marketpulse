"""
Data loader for UK financial time-series.
Sources: FRED API, Bank of England, yfinance.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def load_fred_series() -> pd.DataFrame:
    """
    Load UK-relevant credit spread proxies from FRED.
    Uses ICE BofA indices as credit spread proxies.
    """
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    series = {
        # Credit spreads
        "ig_spread": "BAMLC0A0CM",           # US IG OAS — primary target
        "hy_spread": "BAMLH0A0HYM2",         # US HY OAS — risk proxy
        "euro_ig_spread": "BAMLHE00EHY0EY",  # Euro IG (may not exist, handled)
        # Macro context
        "uk_10y_gilt": "IRLTLT01GBM156N",    # UK 10Y gilt yield
        "us_10y_treasury": "DGS10",          # US 10Y treasury
        "us_vix": "VIXCLS",                  # VIX
        "us_cpi": "CPIAUCSL",               # US CPI (monthly, forward-filled)
    }

    dfs = {}
    for name, ticker in series.items():
        print(f"Loading {name} ({ticker})...")
        try:
            s = fred.get_series(ticker, observation_start="2010-01-01")
            dfs[name] = s
        except Exception as e:
            print(f"  Warning: {e}")

    df = pd.DataFrame(dfs)
    df.index = pd.to_datetime(df.index)
    df = df.resample("B").last().ffill()  # business day frequency
    return df


def load_uk_bank_data() -> pd.DataFrame:
    """
    Load UK bank stock prices as credit quality proxies.
    """
    tickers = {
        "lloyds": "LLOY.L",
        "barclays": "BARC.L",
        "natwest": "NWG.L",
        "hsbc": "HSBA.L",
    }

    dfs = {}
    for name, ticker in tickers.items():
        print(f"Loading {name} ({ticker})...")
        try:
            data = yf.download(ticker, start="2010-01-01", progress=False)
            dfs[name + "_price"] = data["Close"].squeeze()
            dfs[name + "_vol"] = data["Close"].squeeze().pct_change().rolling(21).std() * np.sqrt(252)
        except Exception as e:
            print(f"  Warning: {e}")

    df = pd.DataFrame(dfs)
    df.index = pd.to_datetime(df.index)
    df = df.resample("B").last().ffill()
    return df


def build_feature_matrix() -> pd.DataFrame:
    print("Loading FRED data...")
    fred_df = load_fred_series()

    print("\nLoading UK bank data...")
    bank_df = load_uk_bank_data()

    df = fred_df.join(bank_df, how="inner")
    df = df.dropna(subset=["ig_spread"])  # changed from uk_ig_spread

    # Feature engineering
    df["spread_1d_change"] = df["ig_spread"].diff()
    df["spread_5d_change"] = df["ig_spread"].diff(5)
    df["spread_21d_ma"] = df["ig_spread"].rolling(21).mean()
    df["spread_21d_std"] = df["ig_spread"].rolling(21).std()
    df["spread_zscore"] = (
        (df["ig_spread"] - df["spread_21d_ma"]) / df["spread_21d_std"]
    )
    df["yield_curve_slope"] = df["us_10y_treasury"] - df["uk_10y_gilt"]
    df["risk_regime"] = (df["us_vix"] > 25).astype(int)

    df = df.dropna()

    print(f"\nFeature matrix: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Features: {list(df.columns)}")

    return df


def save_data(df: pd.DataFrame, path: str = "data/processed/features.parquet") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Saved to {path}")


def load_data(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build_feature_matrix()
    save_data(df)
    print("\nSample:")
    print(df.tail())