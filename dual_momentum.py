#!/usr/bin/env python3
"""
Dual Momentum Allocator — Gemini Pro's Final Prescription.

Gary Antonacci-style Relative + Absolute Momentum with cash-gate.
Monthly rebalancing, 12-month lookback, 4-asset universe (SPY/TLT/GLD/DBC).

Rules (run last trading day of each month):
  Step 1 — Absolute Momentum Gate:
    If SPY 12-month excess return > 0 → Risk-On (Step 2)
    Else → Risk-Off (Step 3)

  Step 2 — Risk-On (Relative Momentum):
    Compare 12-month returns of SPY, GLD, DBC
    Select top 2, allocate 50/50

  Step 3 — Risk-Off (Defensive):
    If TLT 12-month return > 0 → 100% TLT
    Else → 100% Cash (0% to all risky assets)

This is NOT designed to beat SPY in bull markets. It's designed to survive
2000-2003, 2007-2009, and 2022 — the regimes where SPY gets destroyed.

Usage:
    python dual_momentum.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ── CORRECTED METRICS ───────────────────────────────────────────────────────

def annualized_return(daily_rets):
    years = len(daily_rets) / 252
    total = np.prod(1 + daily_rets)
    return total ** (1 / years) - 1 if years > 0 else 0.0

def sharpe(r): return r.mean() / (r.std() + 1e-8) * np.sqrt(252)

def _dd(r):
    c = np.cumprod(1 + r); p = np.maximum.accumulate(c)
    return float(np.min((c - p) / p))

def calmar(r):
    ann_ret = annualized_return(r)
    d = _dd(r)
    return ann_ret / abs(d) if d != 0 else 0.0

def sortino(r):
    downside = r[r < 0]
    if len(downside) < 2: return 0.0
    return r.mean() / (downside.std() + 1e-8) * np.sqrt(252)

def cum_return(r): return np.prod(1 + r) - 1


class DualMomentum:
    """
    Dual Momentum across SPY/TLT/GLD/DBC.

    Parameters:
      mom_lookback: trailing window in trading days (252 = 12 months)
      tc_bps: transaction cost in basis points
      cash_return: daily risk-free rate (default: 0)
    """
    def __init__(
        self,
        mom_lookback: int = 252,
        tc_bps: float = 5.0,
        cash_return: float = 0.0,
    ):
        self.mom_lookback = mom_lookback
        self.tc_bps = tc_bps
        self.cash_return = cash_return

    def backtest(self, returns: np.ndarray, dates: pd.DatetimeIndex) -> dict:
        """
        Args:
            returns: [N, A] daily returns (SPY=0, TLT=1, GLD=2, DBC=3)
            dates: [N] DatetimeIndex for detecting month-end
        """
        N, A = returns.shape
        tc_rate = self.tc_bps / 10000.0
        positions = np.zeros((N, A))
        net_rets = np.zeros(N)
        prev_pos = np.ones(A) / A

        # Track last rebalance month
        last_month = -1

        for t in range(N):
            # Default: hold previous position
            target_w = prev_pos.copy()

            # Check if we should rebalance (month-end)
            current_month = dates[t].month
            is_month_end = False
            if t < N - 1:
                next_month = dates[t + 1].month
                is_month_end = (current_month != next_month)
            else:
                is_month_end = True  # last day → rebalance

            # Also check: is this the last trading day of the month?
            # (Simple: month changes tomorrow, or it's year-end)
            if not is_month_end and t < N - 1:
                if dates[t + 1].year != dates[t].year:
                    is_month_end = True

            if is_month_end and t >= self.mom_lookback:
                # Compute 12-month trailing returns
                window_rets = returns[t - self.mom_lookback:t]
                # Cumulative return over window for each asset
                cum_rets = np.prod(1 + window_rets, axis=0) - 1

                spy_ret = cum_rets[0]
                tlt_ret = cum_rets[1]
                gld_ret = cum_rets[2]
                dbc_ret = cum_rets[3]

                target_w = np.zeros(A)

                # Step 1: Absolute Momentum Gate (SPY vs Cash)
                if spy_ret > self.cash_return:
                    # Step 2: Risk-On — Top 2 of SPY, GLD, DBC
                    offensive = {0: spy_ret, 2: gld_ret, 3: dbc_ret}
                    ranked = sorted(offensive.items(), key=lambda x: x[1], reverse=True)
                    target_w[ranked[0][0]] = 0.5
                    target_w[ranked[1][0]] = 0.5
                else:
                    # Step 3: Risk-Off — TLT if positive, else Cash
                    if tlt_ret > self.cash_return:
                        target_w[1] = 1.0  # 100% TLT
                    else:
                        # 100% Cash — zero position (or short-term bills)
                        pass  # target_w stays all zeros

            # Transaction cost
            gross = (target_w * returns[t]).sum()
            turnover = np.abs(target_w - prev_pos).sum()
            cost = tc_rate * turnover
            net_rets[t] = gross - cost
            positions[t] = target_w
            prev_pos = target_w

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
    dates = df.index[lookback:]

    # ── Full-sample backtest ─────────────────────────────────────────────────
    dm = DualMomentum(mom_lookback=252, tc_bps=5.0)

    print("Running Dual Momentum backtest...")
    result = dm.backtest(rets_all, dates)

    ew = rets_all.mean(axis=1)
    vols = np.array([np.nanstd(rets_all[:, i]) * np.sqrt(252) for i in range(n_assets)])
    rp_w = (1.0 / (vols + 1e-8)) / np.sum(1.0 / (vols + 1e-8))
    rp = (rets_all * rp_w).sum(axis=1)
    bw = np.zeros(n_assets); bw[0] = 0.6; bw[1] = 0.4
    bench = (rets_all * bw).sum(axis=1)
    spy = rets_all[:, 0]

    print(f"\n{'='*100}")
    print("DUAL MOMENTUM — Full-Sample 2000-2024 (monthly rebalance, TC=5bp)")
    print(f"{'='*100}")
    print(f"  {'Strategy':<25} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'Calmar':>8} {'Sortino':>8} {'MaxDD':>10} {'CumRet':>10} {'T/O':>8}")

    for name, res in [
        ("Dual Momentum", result),
    ]:
        print(f"  {name:<25} {res['ann_ret']:>+7.1%} {res['ann_vol']:>7.1%} "
              f"{res['sharpe']:>+8.3f} {res['calmar']:>+8.3f} "
              f"{res['sortino']:>+8.3f} {res['max_dd']:>10.2%} "
              f"{res['cum_ret']:>+10.2%} {res['turnover']:>7.4f}")

    for name, rets in [
        ("Equal Weight", ew),
        ("Risk Parity", rp),
        ("60/40 SPY/TLT", bench),
        ("SPY Only", spy),
    ]:
        c = calmar(rets); s = sharpe(rets); so = sortino(rets)
        ann_ret = annualized_return(rets); ann_vol = rets.std() * np.sqrt(252)
        cum = cum_return(rets); dd = _dd(rets)
        print(f"  {name:<25} {ann_ret:>+7.1%} {ann_vol:>7.1%} {s:>+8.3f} "
              f"{c:>+8.3f} {so:>+8.3f} {dd:>10.2%} {cum:>+10.2%} {'--':>8}")

    # ── Key regime periods ───────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("REGIME-SPECIFIC PERFORMANCE (where Dual Momentum should shine)")
    print(f"{'='*100}")
    print(f"  {'Period':<20} {'DM AnnRet':>10} {'SPY AnnRet':>10} "
          f"{'DM MaxDD':>10} {'SPY MaxDD':>10} {'DM beats?':>10}")

    for label, start_str, end_str in [
        ("Dot-Com Bust", "2000-03-24", "2002-10-09"),
        ("GFC 2007-2009", "2007-10-09", "2009-03-09"),
        ("2022 Bear", "2022-01-03", "2022-12-31"),
        ("Lost Decade", "2000-01-03", "2009-12-31"),
        ("Post-GFC Bull", "2009-03-10", "2024-12-31"),
    ]:
        try:
            start_idx = df.index.searchsorted(pd.Timestamp(start_str))
            end_idx = df.index.searchsorted(pd.Timestamp(end_str))
        except:
            # Approximate
            start_idx = max(0, df.index.searchsorted(pd.Timestamp(start_str)) - 1)
            end_idx = min(len(df) - 1, df.index.searchsorted(pd.Timestamp(end_str)))

        start_lb = max(0, start_idx - lookback)
        end_lb = max(0, end_idx - lookback)

        if end_lb - start_lb < 20:
            print(f"  {label:<20} {'(too few days)':>60}")
            continue

        dm_slice = result['net_rets'][start_lb:end_lb]
        spy_slice = spy[start_lb:end_lb]

        dm_ann = annualized_return(dm_slice)
        spy_ann = annualized_return(spy_slice)
        dm_dd = _dd(dm_slice)
        spy_dd = _dd(spy_slice)
        beats = "YES" if dm_ann > spy_ann else "no"

        print(f"  {label:<20} {dm_ann:>+10.1%} {spy_ann:>+10.1%} "
              f"{dm_dd:>10.2%} {spy_dd:>10.2%} {beats:>10}")

    # ── Position summary ─────────────────────────────────────────────────────
    print(f"\n  Position Summary:")
    pos = result['positions']
    for i, t in enumerate(tickers):
        nonzero = (pos[:, i] > 0.01).sum()
        print(f"    {t}: held {nonzero}/{len(pos)} days ({nonzero/len(pos):.0%}), "
              f"mean allocation {pos[:, i].mean():.0%}")

    cash_days = (pos.sum(axis=1) < 0.01).sum()
    print(f"    Cash: {cash_days}/{len(pos)} days ({cash_days/len(pos):.0%})")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("WALK-FORWARD (monthly rebalance, TC=5bp)")
    print(f"{'='*100}")
    print(f"  {'Year':<6} {'DualMom':>10} {'SPY':>10} {'60/40':>10} {'EW':>10} {'DM>SPY?':>8}")

    for year in range(2016, 2025):
        test_start = df.index.searchsorted(pd.Timestamp(f"{year}-01-02"))
        test_end = min(df.index.searchsorted(pd.Timestamp(f"{year}-12-31")), len(df) - 1)
        test_start_lb = test_start - lookback
        test_end_lb = test_end - lookback
        if test_end_lb - test_start_lb < 50: continue

        R_te = rets_all[test_start_lb:test_end_lb]
        D_te = dates[test_start_lb:test_end_lb]

        dm_year = DualMomentum(mom_lookback=252, tc_bps=5.0)
        dm_res = dm_year.backtest(R_te, D_te)

        spy_te = R_te[:, 0]
        ew_te = R_te.mean(axis=1)
        bw_te = np.zeros(n_assets); bw_te[0] = 0.6; bw_te[1] = 0.4
        bench_te = (R_te * bw_te).sum(axis=1)

        dm_c = dm_res['calmar']
        spy_c = calmar(spy_te)
        bench_c = calmar(bench_te)
        ew_c = calmar(ew_te)
        beats = "YES" if dm_c > spy_c else "no"

        print(f"  {year:<6} {dm_c:>+10.3f} {spy_c:>+10.3f} "
              f"{bench_c:>+10.3f} {ew_c:>+10.3f} {beats:>8}")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    np.savez("checkpoints/dual_momentum.npz",
             net_rets=result['net_rets'], positions=result['positions'],
             calmar=result['calmar'], sharpe=result['sharpe'])


if __name__ == "__main__":
    main()
