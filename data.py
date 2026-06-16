"""
Stock World Model — Data Pipeline

1. FRED macro data download (no API key needed)
2. Mock stock data generator (for testing)
3. PIT (Point-in-Time) asymmetric temporal alignment
4. Phase 1 Dataset: sliding-window pairs for self-supervised world model training
5. Phase 2 Dataset: trajectory-level slices for controller training
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# 1. FRED Macro Data Pipeline
# ══════════════════════════════════════════════════════════════════════════════


def download_macro_data(
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame:
    """
    Download macro indicators from FRED (no API key needed).
    Returns DataFrame with: US10Y, US2Y, VIX, plus engineered features.
    """
    tickers = {
        "US10Y": "DGS10",
        "US2Y": "DGS2",
        "VIX": "VIXCLS",
    }
    df_list = []
    for name, ticker in tickers.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df[ticker] = pd.to_numeric(df[ticker], errors="coerce")
        df = df.rename(columns={ticker: name})
        df_list.append(df)

    macro_df = pd.concat(df_list, axis=1)
    macro_df = macro_df.loc[start_date:end_date]
    return macro_df


def clean_and_engineer_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values (async holidays) + feature engineering.
    """
    df = df.ffill().bfill()

    # Yield spread: 10Y - 2Y
    df["Yield_Spread"] = df["US10Y"] - df["US2Y"]

    # VIX 1-week change (5 trading days)
    df["VIX_1w_Change"] = df["VIX"].pct_change(periods=5).fillna(0)

    # 10Y volatility (10-day rolling std)
    df["US10Y_Volatility"] = df["US10Y"].rolling(window=10).std().fillna(0)

    # Ensure sorted
    df = df.sort_index()
    return df


def build_macro_feature_matrix(
    start_date: str = "2019-12-01",
    end_date: str = "2023-01-01",
) -> pd.DataFrame:
    """One-shot: download + clean + engineer. Use this in production."""
    raw = download_macro_data(start_date=start_date, end_date=end_date)
    return clean_and_engineer_macro(raw)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mock Stock Data
# ══════════════════════════════════════════════════════════════════════════════


def generate_mock_stock_data(
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate mock daily OHLCV + quarterly financial data for testing.
    Returns (daily_df, fund_df).
    """
    rng = np.random.default_rng(seed)
    daily_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(daily_dates)

    # Simulate a price series with random walk + trend
    price = 200.0
    prices = []
    for _ in range(n_days):
        price *= (1.0 + rng.normal(0.0005, 0.02))  # ~0.05% daily drift, 2% vol
        prices.append(price)

    daily_df = pd.DataFrame(
        {
            "Open": [p * rng.uniform(0.99, 1.01) for p in prices],
            "Close": prices,
            "High": [p * rng.uniform(1.00, 1.03) for p in prices],
            "Low": [p * rng.uniform(0.97, 1.00) for p in prices],
            "Volume": rng.uniform(1e6, 5e7, n_days),
        },
        index=daily_dates,
    )
    daily_df.index.name = "Date"

    # Mock quarterly financials with PIT announcement dates
    announcement_dates = [
        pd.Timestamp("2022-02-16"),
        pd.Timestamp("2022-05-25"),
        pd.Timestamp("2022-08-24"),
        pd.Timestamp("2022-11-16"),
    ]
    fund_df = pd.DataFrame(
        {
            "Announcement_Date": announcement_dates,
            "Fiscal_Quarter": ["2021Q4", "2022Q1", "2022Q2", "2022Q3"],
            "ROE": [0.25, 0.22, 0.18, 0.20],
            "Debt_Ratio": [0.45, 0.43, 0.48, 0.46],
            "Earnings_Surprise": [0.08, -0.03, -0.05, 0.12],
        }
    ).set_index("Announcement_Date")

    return daily_df, fund_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIT Asymmetric Alignment
# ══════════════════════════════════════════════════════════════════════════════


def align_asymmetric_pipeline(
    daily_df: pd.DataFrame,
    fund_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily price data with sparse quarterly filings using PIT logic.

    - State track (ROE, Debt_Ratio): forward-filled, represents continuous gravity field.
    - Shock track (is_earnings_day, Earnings_Surprise): only active on announcement day,
      zero otherwise. Feeds M-Dynamics micro action stream.
    """
    merged = daily_df.join(fund_df, how="left")

    # Shock track
    merged["is_earnings_day"] = np.where(merged["Fiscal_Quarter"].notna(), 1, 0)
    merged["Earnings_Surprise"] = merged["Earnings_Surprise"].fillna(0.0)

    # State track (forward-fill then backward-fill for pre-first-earnings days)
    stable_features = ["ROE", "Debt_Ratio"]
    merged[stable_features] = merged[stable_features].ffill().bfill()

    merged = merged.sort_index()
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 4. Full Fusion: stock + macro
# ══════════════════════════════════════════════════════════════════════════════


def fuse_stock_and_macro(
    stock_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join stock data with macro indicators. Macro data is forward-filled
    to cover stock trading days (handles async holidays).
    """
    # Ensure both have DatetimeIndex
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        raise TypeError("stock_df must have DatetimeIndex")
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        raise TypeError("macro_df must have DatetimeIndex")

    merged = stock_df.join(macro_df, how="left")
    # Forward-fill macro for stock trading days when bond/Vix market was closed
    macro_cols = macro_df.columns.tolist()
    merged[macro_cols] = merged[macro_cols].ffill().bfill()
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. PyTorch Datasets
# ══════════════════════════════════════════════════════════════════════════════


# Feature column definitions
V_FEATURE_COLS = ["Open", "Close", "Volume", "ROE", "Debt_Ratio"]
# Split into: tech features (daily, seq) + fund features (scalar, current day)
TECH_FEATURE_COLS = ["Open", "Close", "Volume"]
FUND_FEATURE_COLS = ["ROE", "Debt_Ratio"]

MACRO_ACTION_COLS = ["US10Y", "Yield_Spread", "VIX", "VIX_1w_Change", "US10Y_Volatility"]
MICRO_ACTION_COLS = ["is_earnings_day", "Earnings_Surprise"]


class Phase1Dataset(Dataset):
    """
    Phase 1: Self-supervised world model training.

    Returns pairs needed for transition + reward prediction losses:
        tech_seq_t:    [L, tech_dim]  — past L days of technical data
        fund_vec_t:    [fund_dim]     — fundamental state at day t
        action_macro_t: [macro_dim]   — macro environment at day t
        action_micro_t: [micro_dim]   — micro shock at day t
        reward_t1:      scalar         — next-day return (ground truth)
        tech_seq_t1:   [L, tech_dim]  — past L days ending at t+1 (for true z_{t+1})
        fund_vec_t1:   [fund_dim]     — fundamental state at day t+1
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        lookback_window: int = 60,
    ):
        super().__init__()
        self.lookback = lookback_window
        df = dataframe.sort_index().copy()

        # Compute next-day return
        df["Next_Day_Return"] = df["Close"].pct_change().shift(-1).fillna(0.0)

        # Extract numpy arrays for fast indexing
        self.tech_feats = df[TECH_FEATURE_COLS].values.astype(np.float32)
        self.fund_feats = df[FUND_FEATURE_COLS].values.astype(np.float32)
        self.macro_acts = df[MACRO_ACTION_COLS].values.astype(np.float32)
        self.micro_acts = df[MICRO_ACTION_COLS].values.astype(np.float32)
        self.returns = df["Next_Day_Return"].values.astype(np.float32)

        # Valid samples: need lookback days of history + 1 day for next state
        self.total_samples = len(df) - self.lookback - 1

    def __len__(self):
        return max(0, self.total_samples)

    def __getitem__(self, idx: int):
        # t is the last day of the "current" history window
        t = idx + self.lookback - 1  # 0-indexed

        # History window ending at day t
        hist_start = idx  # = t - lookback + 1
        hist_end = t + 1  # exclusive

        # Current state: last day of history window
        tech_seq_t = self.tech_feats[hist_start:hist_end]    # [L, tech_dim]
        fund_vec_t = self.fund_feats[t]                       # [fund_dim]
        macro_t = self.macro_acts[t]                          # [macro_dim]
        micro_t = self.micro_acts[t]                          # [micro_dim]

        # Next day (t+1): for true z_{t+1} encoding and reward
        reward = self.returns[t]                              # scalar

        # History window ending at t+1 (shifted by 1)
        hist_start_t1 = idx + 1
        hist_end_t1 = t + 2
        tech_seq_t1 = self.tech_feats[hist_start_t1:hist_end_t1]  # [L, tech_dim]
        fund_vec_t1 = self.fund_feats[t + 1]                        # [fund_dim]

        return (
            torch.tensor(tech_seq_t, dtype=torch.float32),
            torch.tensor(fund_vec_t, dtype=torch.float32),
            torch.tensor(macro_t, dtype=torch.float32),
            torch.tensor(micro_t, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(tech_seq_t1, dtype=torch.float32),
            torch.tensor(fund_vec_t1, dtype=torch.float32),
        )


class Phase2Dataset(Dataset):
    """
    Phase 2: Trajectory-level slices for controller training.

    Returns:
        x_history:  [L, v_features]  — past L days for V-Encoder
        x_actions:  [K, a_features]  — future K days of actions for M-Dynamics rollout
        y_trajectory: [K]             — future K days of true returns for loss
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        lookback_window: int = 60,
        future_horizon: int = 20,
    ):
        super().__init__()
        self.lookback = lookback_window
        self.horizon = future_horizon
        df = dataframe.sort_index().copy()

        # Next-day return
        df["Next_Day_Return"] = df["Close"].pct_change().shift(-1).fillna(0.0)

        # All V-encoder features
        v_feature_cols = TECH_FEATURE_COLS + FUND_FEATURE_COLS
        self.v_features = df[v_feature_cols].values.astype(np.float32)
        self.actions = df[MACRO_ACTION_COLS + MICRO_ACTION_COLS].values.astype(np.float32)
        self.returns = df["Next_Day_Return"].values.astype(np.float32)
        self.total_samples = len(df) - self.lookback - self.horizon

    def __len__(self):
        return max(0, self.total_samples)

    def __getitem__(self, idx: int):
        hist_start = idx
        hist_end = idx + self.lookback
        future_start = hist_end
        future_end = hist_end + self.horizon

        x_history = self.v_features[hist_start:hist_end]       # [L, V_feats]
        x_actions = self.actions[future_start:future_end]       # [K, A_feats]
        y_traj = self.returns[future_start:future_end]          # [K]

        return (
            torch.tensor(x_history, dtype=torch.float32),
            torch.tensor(x_actions, dtype=torch.float32),
            torch.tensor(y_traj, dtype=torch.float32),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Mock Full Pipeline (for testing without real data)
# ══════════════════════════════════════════════════════════════════════════════


def build_mock_fused_df(
    n_days: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a fully fused mock DataFrame with stock + macro columns.
    Useful for testing the training loop end-to-end.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="B")

    price = 200.0
    closes = []
    for _ in range(n_days):
        price *= (1.0 + rng.normal(0.0005, 0.02))
        closes.append(price)

    df = pd.DataFrame(
        {
            # Technical
            "Open": [p * rng.uniform(0.99, 1.01) for p in closes],
            "Close": closes,
            "Volume": rng.uniform(1e6, 5e7, n_days),
            # Fundamental (simulated quarterly updates via random walk every 60 days)
            "ROE": np.repeat(rng.uniform(0.10, 0.30, n_days // 60 + 1), 60)[:n_days],
            "Debt_Ratio": np.repeat(rng.uniform(0.30, 0.60, n_days // 60 + 1), 60)[:n_days],
            # Macro
            "US10Y": rng.uniform(1.5, 4.5, n_days),
            "Yield_Spread": rng.uniform(-0.5, 0.5, n_days),
            "VIX": rng.uniform(15, 40, n_days),
            "VIX_1w_Change": rng.uniform(-0.1, 0.1, n_days),
            "US10Y_Volatility": rng.uniform(0.05, 0.20, n_days),
            # Micro
            "is_earnings_day": rng.choice([0, 1], n_days, p=[0.95, 0.05]),
            "Earnings_Surprise": rng.uniform(-0.10, 0.10, n_days),
        },
        index=dates,
    )
    return df


def build_multi_regime_df(
    n_days: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build mock data with explicit market regime transitions for
    contrastive learning validation.

    Regimes:
      - bull: positive drift, low VIX, steep yield curve
      - bear: negative drift, high VIX, inverted/flat curve
      - sideways: near-zero drift, moderate VIX

    Returns:
        df: fused DataFrame with regime-varying macro
        regimes: list of regime labels per day (for validation)
    """
    rng = np.random.default_rng(seed)

    # Define regime blocks
    regime_blocks = [
        ("bull", 100, {"drift": +0.002, "vol": 0.012, "us10y": 2.0, "spread": +1.0, "vix": 16}),
        ("sideways", 80, {"drift": +0.0001, "vol": 0.015, "us10y": 2.5, "spread": +0.3, "vix": 22}),
        ("bear", 90, {"drift": -0.002, "vol": 0.025, "us10y": 1.2, "spread": -0.3, "vix": 35}),
        ("sideways", 70, {"drift": -0.0001, "vol": 0.018, "us10y": 2.8, "spread": +0.1, "vix": 25}),
        ("bull", 100, {"drift": +0.0015, "vol": 0.014, "us10y": 3.5, "spread": +0.8, "vix": 18}),
        ("bear", 60, {"drift": -0.003, "vol": 0.030, "us10y": 0.8, "spread": -0.5, "vix": 45}),
    ]

    total = sum(b[1] for b in regime_blocks)
    if total < n_days:
        regime_blocks.append(
            ("sideways", n_days - total,
             {"drift": 0.0, "vol": 0.016, "us10y": 2.5, "spread": +0.2, "vix": 20})
        )
    elif total > n_days:
        # Trim last block
        excess = total - n_days
        last_name, last_n, last_params = regime_blocks[-1]
        regime_blocks[-1] = (last_name, max(10, last_n - excess), last_params)

    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="B")
    regimes = []
    day_idx = 0

    opens_list, closes_list, volumes_list = [], [], []
    roe_list, debt_list = [], []
    us10y_list, spread_list, vix_list = [], [], []
    vix_change_list, us10y_vol_list = [], []
    earn_day_list, earn_surprise_list = [], []

    price = 200.0
    roe = 0.20
    debt = 0.45

    for regime_name, n, params in regime_blocks:
        for _ in range(n):
            if day_idx >= n_days:
                break

            drift = params["drift"]
            vol = params["vol"]
            daily_ret = rng.normal(drift, vol)
            price *= (1.0 + daily_ret)

            opens_list.append(price * rng.uniform(0.99, 1.01))
            closes_list.append(price)
            volumes_list.append(rng.uniform(1e6, 5e7))

            # Slow fundamental changes
            if day_idx % 60 == 0 and day_idx > 0:
                roe += rng.normal(0.0, 0.02)
                debt += rng.normal(0.0, 0.02)

            roe_list.append(roe)
            debt_list.append(debt)

            # Macro with regime-specific mean + noise
            us10y_list.append(params["us10y"] + rng.normal(0, 0.1))
            spread_list.append(params["spread"] + rng.normal(0, 0.05))
            vix_list.append(max(10, params["vix"] + rng.normal(0, 2)))
            vix_change_list.append(rng.normal(0, 0.03))
            us10y_vol_list.append(abs(rng.normal(0.05, 0.02)))

            earn_day_list.append(1 if rng.random() < 0.05 else 0)
            earn_surprise_list.append(rng.normal(0, 0.05) if earn_day_list[-1] else 0.0)

            regimes.append(regime_name)
            day_idx += 1

    df = pd.DataFrame(
        {
            "Open": opens_list,
            "Close": closes_list,
            "Volume": volumes_list,
            "ROE": roe_list,
            "Debt_Ratio": debt_list,
            "US10Y": us10y_list,
            "Yield_Spread": spread_list,
            "VIX": vix_list,
            "VIX_1w_Change": vix_change_list,
            "US10Y_Volatility": us10y_vol_list,
            "is_earnings_day": earn_day_list,
            "Earnings_Surprise": earn_surprise_list,
        },
        index=dates[:n_days],
    )
    return df, regimes
