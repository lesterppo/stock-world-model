#!/usr/bin/env python3
"""
Multi-ETF Data Pipeline — Download and fuse data for SPY, TLT, GLD, QQQ.

Produces a joint dataset where each ETF gets its own technical features,
all share the same macro features, and the target is a portfolio return.

Usage:
    python etf_data.py                          # Download SPY, TLT, GLD, QQQ
    python etf_data.py --tickers SPY,TLT,GLD     # Custom basket
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TICKER_NAMES = {
    "SPY": "S&P 500",
    "TLT": "20+ Yr Treasury",
    "GLD": "Gold",
    "QQQ": "Nasdaq 100",
}


def download_stock(ticker: str, start: str = "2010-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[~data.index.duplicated()]
    data.index = pd.to_datetime(data.index).normalize()
    data.index.name = "Date"
    return data[["Open", "High", "Low", "Close", "Volume"]]


def build_macro(start: str = "2010-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Build macro indicators (FRED fallback)."""
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    rng = np.random.default_rng(42)

    us10y = np.zeros(n)
    us2y = np.zeros(n)
    vix = np.zeros(n)

    for i, d in enumerate(dates):
        year = d.year
        if year <= 2015:
            us10y[i] = 2.2 + rng.normal(0, 0.05)
            us2y[i] = 0.4 + rng.normal(0, 0.03)
            vix[i] = max(10, 18 + rng.normal(0, 3))
        elif year <= 2019:
            us10y[i] = 2.5 + rng.normal(0, 0.04)
            us2y[i] = 2.0 + rng.normal(0, 0.04)
            vix[i] = max(9, 15 + rng.normal(0, 2))
        elif year == 2020:
            if d.month <= 2:
                us10y[i] = 1.5 + rng.normal(0, 0.05)
                us2y[i] = 1.2 + rng.normal(0, 0.04)
                vix[i] = max(12, 18 + rng.normal(0, 3))
            elif d.month <= 4:
                us10y[i] = max(0.5, 0.7 + rng.normal(0, 0.1))
                us2y[i] = max(0.1, 0.2 + rng.normal(0, 0.05))
                vix[i] = max(25, 55 + rng.normal(0, 15))
            else:
                us10y[i] = 0.9 + rng.normal(0, 0.04)
                us2y[i] = 0.2 + rng.normal(0, 0.03)
                vix[i] = max(15, 28 + rng.normal(0, 4))
        elif year == 2021:
            us10y[i] = 1.5 + rng.normal(0, 0.04)
            us2y[i] = 0.3 + rng.normal(0, 0.03)
            vix[i] = max(12, 20 + rng.normal(0, 3))
        elif year == 2022:
            us10y[i] = 3.0 + 0.005 * (d.day_of_year / 365) + rng.normal(0, 0.06)
            us2y[i] = 3.5 + 0.005 * (d.day_of_year / 365) + rng.normal(0, 0.06)
            vix[i] = max(15, 25 + rng.normal(0, 4))
        else:
            us10y[i] = 4.0 + rng.normal(0, 0.05)
            us2y[i] = 4.3 + rng.normal(0, 0.05)
            vix[i] = max(12, 18 + rng.normal(0, 3))

    macro = pd.DataFrame({
        "US10Y": us10y, "US2Y": us2y, "VIX": vix,
    }, index=dates)
    macro["Yield_Spread"] = macro["US10Y"] - macro["US2Y"]
    macro["VIX_1w_Change"] = macro["VIX"].pct_change(periods=5).fillna(0)
    macro["US10Y_Volatility"] = macro["US10Y"].rolling(window=10).std().fillna(0)
    return macro


def main():
    p = argparse.ArgumentParser(description="Multi-ETF Data Pipeline")
    p.add_argument("--tickers", default="SPY,TLT,GLD",
                   help="Comma-separated tickers")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2024-12-31")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    print(f"Downloading: {tickers}")

    # Macro
    print("[1/3] Building macro indicators...")
    macro = build_macro(args.start, args.end)

    # Individual ETFs
    etf_data = {}
    for ticker in tickers:
        print(f"[2/3] Downloading {ticker} ({TICKER_NAMES.get(ticker, '?')})...")
        try:
            etf_data[ticker] = download_stock(ticker, args.start, args.end)
            print(f"  {ticker}: {len(etf_data[ticker])} days")
        except Exception as e:
            print(f"  WARNING: {ticker} failed: {e}")

    if not etf_data:
        print("ERROR: No ETF data downloaded")
        return

    # Fuse: align all ETFs on common trading days
    print("[3/3] Fusing multi-ETF dataset...")
    # Start with the first ETF's index, inner-join others
    common_idx = etf_data[tickers[0]].index
    for ticker in tickers[1:]:
        common_idx = common_idx.intersection(etf_data[ticker].index)

    fused = pd.DataFrame(index=common_idx)

    # Add macro features (forward-filled to trading days)
    for col in macro.columns:
        fused[col] = np.nan
        for date in macro.index:
            if date in fused.index:
                fused.loc[date, col] = macro.loc[date, col]
    fused = fused.ffill().bfill()

    # Add ETF-specific features with prefix
    for ticker in tickers:
        etf = etf_data[ticker].loc[common_idx]
        for col in ["Open", "Close", "Volume"]:
            fused[f"{ticker}_{col}"] = etf[col]
        # Returns
        fused[f"{ticker}_Return"] = etf["Close"].pct_change()
        fused[f"{ticker}_Next_Return"] = etf["Close"].pct_change().shift(-1)

    # Synthetic fundamental features per ETF
    for ticker in tickers:
        fused[f"{ticker}_ROE"] = 0.15
        fused[f"{ticker}_Debt_Ratio"] = 0.30

    # Portfolio return target: equal-weighted
    fused["Portfolio_Next_Return"] = 0.0
    for ticker in tickers:
        fused["Portfolio_Next_Return"] += fused[f"{ticker}_Next_Return"]
    fused["Portfolio_Next_Return"] /= len(tickers)

    # Clean
    fused = fused.dropna()
    fused = fused.sort_index()

    # Save
    tag = "_".join(tickers)
    out_path = DATA_DIR / f"multi_{tag}_fused.csv"
    fused.to_csv(out_path)
    print(f"\nSaved: {out_path} ({len(fused)} rows, {len(fused.columns)} columns)")
    print(f"Date range: {fused.index[0].date()} — {fused.index[-1].date()}")
    print(f"Columns: {list(fused.columns)}")

    # Quick stats
    print(f"\nPortfolio stats:")
    port_ret = fused["Portfolio_Next_Return"]
    print(f"  Mean daily: {port_ret.mean():.6f} ({port_ret.mean()*252:.2%} ann)")
    print(f"  Std daily:  {port_ret.std():.6f} ({port_ret.std()*np.sqrt(252):.2%} ann)")
    print(f"  Sharpe:     {port_ret.mean()/port_ret.std()*np.sqrt(252):.4f}")

    # Individual correlations
    print(f"\nCross-asset correlations:")
    ret_cols = [f"{t}_Return" for t in tickers]
    corr = fused[ret_cols].corr()
    print(corr.to_string())


if __name__ == "__main__":
    main()
