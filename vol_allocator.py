#!/usr/bin/env python3
"""
Volatility Targeting + Trend Following — Production Allocator.

Fixes two bugs caught by Gemini Pro:
  1. Calmar now uses ANNUALIZED return (not cumulative)
  2. Vol targeting now targets the equity sleeve directly, not diluted by normalization

Combined TrendVol strategy (CTA/Managed Futures pattern):
  - Trend filter: only eligible if Close > 200-day SMA
  - Vol sizing: position = target_vol / realized_vol for each eligible asset
  - Remainder goes to TLT (safe haven), never cash-drag
  - Monthly rebalancing with 5% tolerance band (no daily noise-trading)

Benchmarks: Equal Weight, Risk Parity, 60/40, SPY-only, Trend-only, Vol-only

Usage:
    python vol_allocator.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ── CORRECTED METRICS ───────────────────────────────────────────────────────

def annualized_return(daily_rets):
    """Geometric annualized return from daily returns."""
    years = len(daily_rets) / 252
    total = np.prod(1 + daily_rets)
    return total ** (1 / years) - 1 if years > 0 else 0.0

def sharpe(r): return r.mean() / (r.std() + 1e-8) * np.sqrt(252)

def _dd(r):
    c = np.cumprod(1 + r); p = np.maximum.accumulate(c)
    return float(np.min((c - p) / p))

def calmar(r):
    """CORRECTED: Annualized return / Max Drawdown."""
    ann_ret = annualized_return(r)
    d = _dd(r)
    return ann_ret / abs(d) if d != 0 else 0.0

def sortino(r):
    downside = r[r < 0]
    if len(downside) < 2: return 0.0
    return r.mean() / (downside.std() + 1e-8) * np.sqrt(252)

def cum_return(r): return np.prod(1 + r) - 1


# ── VOLATILITY TARGETING ALLOCATOR ──────────────────────────────────────────

class VolTargetAllocator:
    """
    Pure vol targeting: size each asset by target_vol / realized_vol.
    Equity assets (SPY, DBC) get their vol-targeted allocation.
    Safe assets (TLT, GLD) split the remainder.

    No trend filter — always invested. Just varies sizing.
    """
    def __init__(
        self,
        target_vol: float = 0.15,
        vol_lookback: int = 63,         # 3-month realized vol
        equity_indices: list = [0, 3],  # SPY, DBC
        safe_indices: list = [1, 2],    # TLT, GLD
        max_single: float = 1.0,
        tc_bps: float = 2.0,
    ):
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.equity_indices = equity_indices
        self.safe_indices = safe_indices
        self.max_single = max_single
        self.tc_bps = tc_bps

    def allocate(self, returns: np.ndarray) -> np.ndarray:
        N, A = returns.shape
        positions = np.zeros((N, A))
        tc_rate = self.tc_bps / 10000.0

        for t in range(N):
            if t < self.vol_lookback:
                positions[t] = np.ones(A) / A
                continue

            rets_window = returns[t - self.vol_lookback:t]
            vols = rets_window.std(axis=0) * np.sqrt(252) + 1e-8

            # Equity sleeve: each gets target_vol / realized_vol, capped
            eq_w = np.zeros(A)
            eq_used = 0.0
            for i in self.equity_indices:
                w = min(self.max_single, self.target_vol / vols[i])
                eq_w[i] = w
                eq_used += w

            # If equity allocation exceeds 1.0, scale down
            if eq_used > 1.0:
                eq_w /= eq_used
                positions[t] = eq_w
                continue

            # Remainder goes to safe assets, proportionally by inverse vol
            remaining = 1.0 - eq_used
            safe_inv_vol = np.array([1.0 / vols[i] for i in self.safe_indices])
            safe_sum = safe_inv_vol.sum()
            if safe_sum > 0:
                for j, i in enumerate(self.safe_indices):
                    eq_w[i] = remaining * safe_inv_vol[j] / safe_sum

            positions[t] = eq_w

        return positions

    def backtest(self, returns: np.ndarray) -> dict:
        positions = self.allocate(returns)
        N, A = returns.shape
        tc_rate = self.tc_bps / 10000.0
        net_rets = np.zeros(N)
        prev_pos = np.ones(A) / A

        for t in range(N):
            gross = (positions[t] * returns[t]).sum()
            turnover = np.abs(positions[t] - prev_pos).sum()
            cost = tc_rate * turnover
            net_rets[t] = gross - cost
            prev_pos = positions[t]

        return {
            "net_rets": net_rets,
            "positions": positions,
            "calmar": calmar(net_rets),
            "sharpe": sharpe(net_rets),
            "sortino": sortino(net_rets),
            "cum_ret": cum_return(net_rets),
            "max_dd": _dd(net_rets),
            "ann_ret": annualized_return(net_rets),
            "ann_vol": net_rets.std() * np.sqrt(252),
            "turnover": np.abs(np.diff(positions, axis=0)).mean() if N > 1 else 0,
        }


# ── TREND FOLLOWING ALLOCATOR ───────────────────────────────────────────────

class TrendAllocator:
    """
    Pure trend: only hold assets above their 200-day SMA.
    Equal-weight across eligible assets. If none eligible → TLT.
    """
    def __init__(self, sma_lookback: int = 200, safe_idx: int = 1, tc_bps: float = 2.0):
        self.sma_lookback = sma_lookback
        self.safe_idx = safe_idx
        self.tc_bps = tc_bps

    def backtest(self, prices: np.ndarray, returns: np.ndarray) -> dict:
        N, A = prices.shape
        tc_rate = self.tc_bps / 10000.0
        positions = np.zeros((N, A))
        net_rets = np.zeros(N)
        prev_pos = np.ones(A) / A

        for t in range(N):
            if t < self.sma_lookback:
                positions[t] = prev_pos
                net_rets[t] = (prev_pos * returns[t]).sum()
                continue

            sma = prices[t - self.sma_lookback:t].mean(axis=0)
            eligible = np.where(prices[t] > sma)[0]

            if len(eligible) == 0:
                target_w = np.zeros(A)
                target_w[self.safe_idx] = 1.0
            else:
                target_w = np.zeros(A)
                target_w[eligible] = 1.0 / len(eligible)

            gross = (target_w * returns[t]).sum()
            turnover = np.abs(target_w - prev_pos).sum()
            cost = tc_rate * turnover
            net_rets[t] = gross - cost
            positions[t] = target_w
            prev_pos = target_w

        return {
            "net_rets": net_rets, "positions": positions,
            "calmar": calmar(net_rets), "sharpe": sharpe(net_rets),
            "sortino": sortino(net_rets), "cum_ret": cum_return(net_rets),
            "max_dd": _dd(net_rets), "ann_ret": annualized_return(net_rets),
            "ann_vol": net_rets.std() * np.sqrt(252),
            "turnover": np.abs(np.diff(positions, axis=0)).mean() if N > 1 else 0,
        }


# ── COMBINED TREND + VOL ALLOCATOR ──────────────────────────────────────────

class TrendVolAllocator:
    """
    Combined CTA/managed futures approach:
      1. Trend filter: asset must be above 200-day SMA
      2. Vol sizing: position = target_vol / realized_vol for eligible assets
      3. Normalize eligible weights to sum ≤ 1.0
      4. Remainder → TLT (or equal-weight safe if TLT not eligible)
      5. Monthly rebalancing with 5% tolerance band
    """
    def __init__(
        self,
        target_vol: float = 0.15,
        vol_lookback: int = 63,
        sma_lookback: int = 200,
        safe_idx: int = 1,            # TLT
        rebalance_freq: int = 21,     # monthly (~21 trading days)
        tolerance_band: float = 0.05, # 5% drift before rebalance
        max_single: float = 1.0,
        tc_bps: float = 5.0,
    ):
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.sma_lookback = sma_lookback
        self.safe_idx = safe_idx
        self.rebalance_freq = rebalance_freq
        self.tolerance_band = tolerance_band
        self.max_single = max_single
        self.tc_bps = tc_bps

    def backtest(self, prices: np.ndarray, returns: np.ndarray) -> dict:
        N, A = prices.shape
        tc_rate = self.tc_bps / 10000.0
        positions = np.zeros((N, A))
        net_rets = np.zeros(N)
        prev_pos = np.ones(A) / A
        days_since_rebalance = 0

        for t in range(N):
            # Use previous day's position by default
            current_pos = prev_pos.copy()

            # Check if rebalance is needed
            need_rebalance = False

            if t >= max(self.sma_lookback, self.vol_lookback):
                if days_since_rebalance >= self.rebalance_freq:
                    need_rebalance = True
                elif self.tolerance_band > 0:
                    # Check drift
                    drift = np.abs(current_pos - prev_pos).max()
                    # Actually check vs the theoretical target
                    # (simplified: rebalance if enough days passed)

                if days_since_rebalance >= self.rebalance_freq:
                    # Compute new target weights
                    target_w = self._compute_target(prices, returns, t)
                    # Apply tolerance band
                    drift = np.abs(target_w - prev_pos).sum()
                    if drift > self.tolerance_band * 2:
                        # Execute trade
                        gross = (target_w * returns[t]).sum()
                        turnover = np.abs(target_w - prev_pos).sum()
                        cost = tc_rate * turnover
                        net_rets[t] = gross - cost
                        current_pos = target_w
                        prev_pos = target_w
                        days_since_rebalance = 0
                    else:
                        # Skip — just hold
                        net_rets[t] = (current_pos * returns[t]).sum()
                        days_since_rebalance += 1
                else:
                    net_rets[t] = (current_pos * returns[t]).sum()
                    days_since_rebalance += 1
            else:
                # Not enough history — equal weight
                current_pos = np.ones(A) / A
                net_rets[t] = (current_pos * returns[t]).sum()
                prev_pos = current_pos
                days_since_rebalance += 1

            positions[t] = current_pos

        return {
            "net_rets": net_rets, "positions": positions,
            "calmar": calmar(net_rets), "sharpe": sharpe(net_rets),
            "sortino": sortino(net_rets), "cum_ret": cum_return(net_rets),
            "max_dd": _dd(net_rets), "ann_ret": annualized_return(net_rets),
            "ann_vol": net_rets.std() * np.sqrt(252),
            "turnover": np.abs(np.diff(positions, axis=0)).mean() if N > 1 else 0,
        }

    def _compute_target(self, prices, returns, t):
        A = returns.shape[1]

        # Trend filter
        sma = prices[t - self.sma_lookback:t].mean(axis=0)
        above_sma = prices[t] > sma

        # Realized vol
        rets_window = returns[t - self.vol_lookback:t]
        vols = rets_window.std(axis=0) * np.sqrt(252) + 1e-8

        # Eligible assets (above SMA, not safe haven)
        eligible = []
        for i in range(A):
            if i == self.safe_idx:
                continue  # TLT handled separately
            if above_sma[i]:
                eligible.append(i)

        target = np.zeros(A)

        if len(eligible) == 0:
            # Nothing trending — all in TLT
            target[self.safe_idx] = 1.0
            return target

        # Vol-size each eligible asset
        total_w = 0.0
        for i in eligible:
            w = min(self.max_single, self.target_vol / vols[i])
            target[i] = w
            total_w += w

        if total_w > 1.0:
            # Scale down
            target /= total_w
        else:
            # Remainder to TLT
            target[self.safe_idx] = 1.0 - total_w

        return target


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)
    lookback = 60

    n = len(df)
    n_out = n - lookback
    rets_all = np.zeros((n_out, n_assets), dtype=np.float32)
    prices_all = np.zeros((n_out, n_assets), dtype=np.float32)
    for i, t in enumerate(tickers):
        rets_all[:, i] = df[f'{t}_Next_Return'].values[lookback:]
        prices_all[:, i] = df[f'{t}_Close'].values[lookback:]

    # ── Walk-forward ────────────────────────────────────────────────────────
    print(f"{'='*110}")
    print("WALK-FORWARD: Trend+Vol vs Benchmarks (monthly rebalance, TC=5bp, CORRECTED Calmar)")
    print(f"{'='*110}")
    print(f"  {'Year':<6} {'TrendVol':>10} {'VolTgt':>10} {'Trend':>10} "
          f"{'EW':>10} {'RP':>10} {'60/40':>10} {'SPY':>10} {'Best':>10}")

    for year in range(2016, 2025):
        test_start = df.index.searchsorted(pd.Timestamp(f"{year}-01-02"))
        test_end = min(df.index.searchsorted(pd.Timestamp(f"{year}-12-31")), len(df) - 1)
        test_start_lb = test_start - lookback
        test_end_lb = test_end - lookback
        if test_end_lb - test_start_lb < 50: continue

        R_te = rets_all[test_start_lb:test_end_lb]
        P_te = prices_all[test_start_lb:test_end_lb]

        # Trend+Vol combined
        tv = TrendVolAllocator(target_vol=0.15, tc_bps=5.0)
        tv_res = tv.backtest(P_te, R_te)

        # Vol only
        vt = VolTargetAllocator(target_vol=0.15, tc_bps=2.0)
        vt_res = vt.backtest(R_te)

        # Trend only
        tf = TrendAllocator(sma_lookback=200, safe_idx=1, tc_bps=2.0)
        tf_res = tf.backtest(P_te, R_te)

        # Benchmarks
        ew = R_te.mean(axis=1)
        vols = np.array([np.nanstd(R_te[:, i]) * np.sqrt(252) for i in range(n_assets)])
        rp_w = (1.0 / (vols + 1e-8)) / np.sum(1.0 / (vols + 1e-8))
        rp = (R_te * rp_w).sum(axis=1)
        bw = np.zeros(n_assets); bw[0] = 0.6; bw[1] = 0.4
        bench = (R_te * bw).sum(axis=1)
        spy = R_te[:, 0]

        scores = [
            ("TrendVol", tv_res['calmar']),
            ("VolTgt", vt_res['calmar']),
            ("Trend", tf_res['calmar']),
            ("EW", calmar(ew)),
            ("RP", calmar(rp)),
            ("60/40", calmar(bench)),
            ("SPY", calmar(spy)),
        ]
        best = max(scores, key=lambda x: x[1])[0]

        print(f"  {year:<6} {tv_res['calmar']:>+10.3f} {vt_res['calmar']:>+10.3f} "
              f"{tf_res['calmar']:>+10.3f} {calmar(ew):>+10.3f} {calmar(rp):>+10.3f} "
              f"{calmar(bench):>+10.3f} {calmar(spy):>+10.3f} {best:>10}")

    # ── Full-sample summary ─────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("FULL-SAMPLE SUMMARY (all data 2000-2024, CORRECTED Calmar = AnnRet / MaxDD)")
    print(f"{'='*100}")
    print(f"  {'Strategy':<30} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'Calmar':>8} {'MaxDD':>10} {'CumRet':>10} {'T/O':>8}")

    tv = TrendVolAllocator(target_vol=0.15, tc_bps=5.0)
    tv_res = tv.backtest(prices_all, rets_all)

    vt = VolTargetAllocator(target_vol=0.15, tc_bps=2.0)
    vt_res = vt.backtest(rets_all)

    tf = TrendAllocator(sma_lookback=200, safe_idx=1, tc_bps=2.0)
    tf_res = tf.backtest(prices_all, rets_all)

    ew = rets_all.mean(axis=1)
    vols = np.array([np.nanstd(rets_all[:, i]) * np.sqrt(252) for i in range(n_assets)])
    rp_w = (1.0 / (vols + 1e-8)) / np.sum(1.0 / (vols + 1e-8))
    rp = (rets_all * rp_w).sum(axis=1)
    bw = np.zeros(n_assets); bw[0] = 0.6; bw[1] = 0.4
    bench = (rets_all * bw).sum(axis=1)
    spy = rets_all[:, 0]

    for name, res in [
        ("Trend+Vol (combined)", tv_res),
        ("Vol Target", vt_res),
        ("Trend Follow", tf_res),
    ]:
        print(f"  {name:<30} {res['ann_ret']:>+7.1%} {res['ann_vol']:>7.1%} "
              f"{res['sharpe']:>+8.3f} {res['calmar']:>+8.3f} "
              f"{res['max_dd']:>10.2%} {res['cum_ret']:>+10.2%} "
              f"{res['turnover']:>7.4f}")

    for name, rets in [
        ("Equal Weight", ew),
        ("Risk Parity", rp),
        ("60/40 SPY/TLT", bench),
        ("SPY Only", spy),
    ]:
        c = calmar(rets); s = sharpe(rets); so = sortino(rets)
        ann_ret = annualized_return(rets); ann_vol = rets.std() * np.sqrt(252)
        cum = cum_return(rets); dd = _dd(rets)
        print(f"  {name:<30} {ann_ret:>+7.1%} {ann_vol:>7.1%} {s:>+8.3f} "
              f"{c:>+8.3f} {dd:>10.2%} {cum:>+10.2%} {'--':>8}")

    # ── Position summary ────────────────────────────────────────────────────
    print(f"\n  Position Ranges (Trend+Vol):")
    tv_pos = tv_res['positions']
    for i, t in enumerate(tickers):
        print(f"    {t}: [{tv_pos[:,i].min():.0%}, {tv_pos[:,i].max():.0%}], "
              f"mean={tv_pos[:,i].mean():.0%}, "
              f">0 on {(tv_pos[:,i] > 0.01).sum()}/{len(tv_pos)} days")

    # ── Drawdown comparison ─────────────────────────────────────────────────
    print(f"\n  Worst Drawdowns:")
    for name, rets in [
        ("Trend+Vol", tv_res['net_rets']),
        ("SPY Only", spy),
        ("60/40", bench),
    ]:
        dd = _dd(rets)
        c = np.cumprod(1 + rets)
        peak_idx = np.argmax(c)
        trough_idx = peak_idx + np.argmin(c[peak_idx:] / c[peak_idx])
        print(f"    {name:<15}: {dd:.1%} (peak day {peak_idx}, trough day {trough_idx})")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    np.savez("checkpoints/trendvol.npz",
             net_rets=tv_res['net_rets'], positions=tv_res['positions'],
             calmar=tv_res['calmar'], sharpe=tv_res['sharpe'])


if __name__ == "__main__":
    main()
