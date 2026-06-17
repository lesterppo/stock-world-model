#!/usr/bin/env python3
"""
Leveraged Dual Momentum — Tiered Leverage on SPY Only.

Gemini Pro's blueprint:
  SPY 12-month return > +10%  → 1.5x levered (high conviction bull)
  SPY 12-month return 0%–10%  → 1.0x (weak/flat bull)
  SPY 12-month return < 0%    → 0x leverage, 100% Cash or TLT

Key safeguards:
  - Leverage applies ONLY to SPY, never GLD/DBC (no long-term risk premium)
  - Tiered (not binary) avoids whip-saw destruction at the 0% boundary
  - Margin cost: SOFR + 1% ≈ 5.5% annual, charged daily on borrowed amount
  - Monthly rebalancing, TC=5bp

Compares: SPY-only, Dual Momentum (unlevered), Levered DM (tiered)

Usage:
    python levered_dm.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ── METRICS ──────────────────────────────────────────────────────────────────

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


class LeveredDualMomentum:
    """
    Tiered-leverage Dual Momentum with SPY-only leverage.

    Monthly rebalancing. Only borrows against SPY positions.
    GLD and DBC always held at 1.0x (no leverage on non-productive assets).
    """
    def __init__(
        self,
        mom_lookback: int = 252,
        tc_bps: float = 5.0,
        margin_rate: float = 0.055,     # SOFR + 1% annual
        high_leverage: float = 1.5,      # SPY 12m > +10%
        mid_leverage: float = 1.0,       # SPY 12m 0%–10%
        high_conviction_threshold: float = 0.10,
    ):
        self.mom_lookback = mom_lookback
        self.tc_bps = tc_bps
        self.margin_rate_daily = margin_rate / 252
        self.high_leverage = high_leverage
        self.mid_leverage = mid_leverage
        self.high_threshold = high_conviction_threshold

    def _is_month_end(self, dates, t):
        """Check if day t is the last trading day of its month."""
        if t >= len(dates) - 1:
            return True
        return (dates[t].month != dates[t + 1].month or
                dates[t].year != dates[t + 1].year)

    def _compute_spy_momentum(self, returns, t):
        """12-month SPY return."""
        if t < self.mom_lookback:
            return 0.0
        return np.prod(1 + returns[t - self.mom_lookback:t, 0]) - 1

    def backtest(self, returns: np.ndarray, dates: pd.DatetimeIndex) -> dict:
        """
        Args:
            returns: [N, A] daily returns (SPY=0, TLT=1, GLD=2, DBC=3)
            dates:   [N] DatetimeIndex
        """
        N, A = returns.shape
        tc_rate = self.tc_bps / 10000.0

        positions = np.zeros((N, A))       # gross positions (may sum > 1 for leverage)
        leverage_history = np.zeros(N)      # total leverage used each day
        net_rets = np.zeros(N)
        margin_costs = np.zeros(N)
        prev_pos = np.ones(A) / A
        current_spy_leverage = 1.0

        for t in range(N):
            if t < self.mom_lookback or not self._is_month_end(dates, t):
                # Hold previous position
                target_w = prev_pos.copy()
                spy_lev = current_spy_leverage
            else:
                # ── Rebalance: compute momentum signals ─────────────────────
                window_rets = returns[t - self.mom_lookback:t]
                cum_rets = np.prod(1 + window_rets, axis=0) - 1

                spy_ret = cum_rets[0]
                tlt_ret = cum_rets[1]
                gld_ret = cum_rets[2]
                dbc_ret = cum_rets[3]

                target_w = np.zeros(A)

                # Step 1: Determine SPY leverage tier
                if spy_ret > self.high_threshold:
                    spy_lev = self.high_leverage      # 1.5x
                elif spy_ret > 0:
                    spy_lev = self.mid_leverage        # 1.0x
                else:
                    spy_lev = 0.0                      # Risk-Off

                # Step 2: Allocate
                if spy_lev > 0:
                    # Risk-On: top 2 of SPY, GLD, DBC by 12-month return
                    offensive = {0: spy_ret, 2: gld_ret, 3: dbc_ret}
                    ranked = sorted(offensive.items(), key=lambda x: x[1], reverse=True)

                    # SPY gets leverage if it's in top 2
                    for rank, (asset_idx, _) in enumerate(ranked[:2]):
                        if asset_idx == 0:  # SPY
                            target_w[0] = 0.5 * spy_lev  # levered
                        else:
                            target_w[asset_idx] = 0.5     # unlevered
                else:
                    # Risk-Off: TLT if trending, else Cash
                    if tlt_ret > 0:
                        target_w[1] = 1.0
                    # else: all zeros = 100% cash

            # ── Margin cost ──────────────────────────────────────────────
            total_gross = target_w.sum()
            borrowed = max(0, total_gross - 1.0)
            margin_cost = borrowed * self.margin_rate_daily

            # ── Net return = gross return - margin cost - transaction cost ─
            gross_ret = (target_w * returns[t]).sum()
            turnover = np.abs(target_w - prev_pos).sum()
            tc_cost = tc_rate * turnover
            net_rets[t] = gross_ret - margin_cost - tc_cost

            positions[t] = target_w
            leverage_history[t] = total_gross
            margin_costs[t] = margin_cost
            prev_pos = target_w
            current_spy_leverage = spy_lev

        return {
            "net_rets": net_rets,
            "positions": positions,
            "leverage_history": leverage_history,
            "margin_costs": margin_costs,
            "calmar": calmar(net_rets),
            "sharpe": sharpe(net_rets),
            "sortino": sortino(net_rets),
            "cum_ret": cum_return(net_rets),
            "max_dd": _dd(net_rets),
            "ann_ret": annualized_return(net_rets),
            "ann_vol": net_rets.std() * np.sqrt(252),
            "turnover": np.abs(np.diff(positions, axis=0)).mean() if N > 1 else 0,
        }


# ── Unlevered Dual Momentum (from dual_momentum.py) ─────────────────────────

class DualMomentum:
    """Same as dual_momentum.py — unlevered, for comparison."""
    def __init__(self, mom_lookback=252, tc_bps=5.0, cash_return=0.0):
        self.mom_lookback = mom_lookback
        self.tc_bps = tc_bps
        self.cash_return = cash_return

    def backtest(self, returns, dates):
        N, A = returns.shape
        tc_rate = self.tc_bps / 10000.0
        positions = np.zeros((N, A))
        net_rets = np.zeros(N)
        prev_pos = np.ones(A) / A

        for t in range(N):
            target_w = prev_pos.copy()
            is_month_end = False
            if t >= N - 1:
                is_month_end = True
            elif dates[t].month != dates[t + 1].month or dates[t].year != dates[t + 1].year:
                is_month_end = True

            if is_month_end and t >= self.mom_lookback:
                cum_rets = np.prod(1 + returns[t - self.mom_lookback:t], axis=0) - 1
                spy_ret, tlt_ret, gld_ret, dbc_ret = cum_rets[0], cum_rets[1], cum_rets[2], cum_rets[3]
                target_w = np.zeros(A)
                if spy_ret > self.cash_return:
                    offensive = {0: spy_ret, 2: gld_ret, 3: dbc_ret}
                    ranked = sorted(offensive.items(), key=lambda x: x[1], reverse=True)
                    target_w[ranked[0][0]] = 0.5
                    target_w[ranked[1][0]] = 0.5
                else:
                    if tlt_ret > self.cash_return:
                        target_w[1] = 1.0

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


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)
    lookback = 60

    n = len(df)
    n_out = n - lookback
    rets_all = np.zeros((n_out, n_assets), dtype=np.float32)
    for i, t in enumerate(tickers):
        rets_all[:, i] = df[f'{t}_Next_Return'].values[lookback:]
    dates = df.index[lookback:]

    # ── All strategies ───────────────────────────────────────────────────────
    print("Running strategies...")

    levered = LeveredDualMomentum(mom_lookback=252, tc_bps=5.0, margin_rate=0.055,
                                   high_leverage=1.5, mid_leverage=1.0)
    ldm_res = levered.backtest(rets_all, dates)

    unlevered = DualMomentum(mom_lookback=252, tc_bps=5.0)
    dm_res = unlevered.backtest(rets_all, dates)

    spy = rets_all[:, 0]
    ew = rets_all.mean(axis=1)
    vols = np.array([np.nanstd(rets_all[:, i]) * np.sqrt(252) for i in range(n_assets)])
    rp_w = (1.0 / (vols + 1e-8)) / np.sum(1.0 / (vols + 1e-8))
    rp = (rets_all * rp_w).sum(axis=1)
    bw = np.zeros(n_assets); bw[0] = 0.6; bw[1] = 0.4
    bench = (rets_all * bw).sum(axis=1)

    # ── Full-sample ──────────────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print("LEVERAGED DUAL MOMENTUM — Full-Sample 2000-2024")
    print(f"  (Tiered: 1.5x when SPY 12m > +10%, 1.0x when 0–10%, 0x when <0%)")
    print(f"  (Margin cost: 5.5%/yr on borrowed, TC=5bp, monthly rebalance)")
    print(f"{'='*105}")
    print(f"  {'Strategy':<28} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'Calmar':>8} {'Sortino':>8} {'MaxDD':>10} {'CumRet':>10} {'T/O':>8}")

    for name, res in [
        ("Levered DM (tiered 0/1.0/1.5x)", ldm_res),
        ("Dual Momentum (unlevered)", dm_res),
    ]:
        print(f"  {name:<28} {res['ann_ret']:>+7.1%} {res['ann_vol']:>7.1%} "
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
        print(f"  {name:<28} {ann_ret:>+7.1%} {ann_vol:>7.1%} {s:>+8.3f} "
              f"{c:>+8.3f} {so:>+8.3f} {dd:>10.2%} {cum:>+10.2%} {'--':>8}")

    # ── Leverage stats ───────────────────────────────────────────────────────
    lev_hist = ldm_res['leverage_history']
    margin_total = ldm_res['margin_costs'].sum()
    print(f"\n  Leverage Stats (Levered DM):")
    print(f"    1.5x days: {(lev_hist > 1.01).sum()}/{len(lev_hist)} ({(lev_hist > 1.01).mean():.0%})")
    print(f"    1.0x days: {((lev_hist > 0.99) & (lev_hist <= 1.01)).sum()}/{len(lev_hist)}")
    print(f"    0x (cash) days: {(lev_hist < 0.01).sum()}/{len(lev_hist)} ({(lev_hist < 0.01).mean():.0%})")
    print(f"    Average gross exposure: {lev_hist.mean():.2f}x")
    print(f"    Total margin cost: {margin_total:.2%} of final portfolio")

    # ── Regime performance ───────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print("REGIME BREAKDOWN")
    print(f"{'='*105}")
    print(f"  {'Period':<22} {'LevDM AnnRet':>12} {'DM AnnRet':>12} "
          f"{'SPY AnnRet':>12} {'LevDM MaxDD':>12} {'SPY MaxDD':>12}")

    for label, start_str, end_str in [
        ("2022 Bear", "2022-01-03", "2022-12-31"),
        ("GFC 2007-2009", "2007-10-09", "2009-03-09"),
        ("COVID Crash", "2020-02-19", "2020-03-23"),
        ("Post-GFC Bull", "2009-03-10", "2024-12-31"),
    ]:
        try:
            si = df.index.searchsorted(pd.Timestamp(start_str))
            ei = df.index.searchsorted(pd.Timestamp(end_str))
        except:
            si = max(0, df.index.searchsorted(pd.Timestamp(start_str)) - 1)
            ei = min(len(df) - 1, df.index.searchsorted(pd.Timestamp(end_str)))
        sl = max(0, si - lookback)
        el = max(0, ei - lookback)
        if el - sl < 20:
            print(f"  {label:<22} {'(data unavailable)':>60}")
            continue

        ldm_a = annualized_return(ldm_res['net_rets'][sl:el])
        dm_a = annualized_return(dm_res['net_rets'][sl:el])
        spy_a = annualized_return(spy[sl:el])
        ldm_dd = _dd(ldm_res['net_rets'][sl:el])
        spy_dd = _dd(spy[sl:el])
        print(f"  {label:<22} {ldm_a:>+12.1%} {dm_a:>+12.1%} "
              f"{spy_a:>+12.1%} {ldm_dd:>12.2%} {spy_dd:>12.2%}")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print("WALK-FORWARD (monthly, TC=5bp, margin=5.5%)")
    print(f"{'='*105}")
    print(f"  {'Year':<6} {'LevDM':>10} {'UnlevDM':>10} {'SPY':>10} {'60/40':>10} {'Lev>SPY?':>9} {'Lev>DM?':>9}")

    for year in range(2016, 2025):
        ts = df.index.searchsorted(pd.Timestamp(f"{year}-01-02"))
        te = min(df.index.searchsorted(pd.Timestamp(f"{year}-12-31")), len(df) - 1)
        tsl = ts - lookback; tel = te - lookback
        if tel - tsl < 50: continue

        R_te = rets_all[tsl:tel]
        D_te = dates[tsl:tel]

        ldm_y = LeveredDualMomentum(mom_lookback=252, tc_bps=5.0, margin_rate=0.055,
                                     high_leverage=1.5, mid_leverage=1.0)
        ldm_yr = ldm_y.backtest(R_te, D_te)

        dm_y = DualMomentum(mom_lookback=252, tc_bps=5.0)
        dm_yr = dm_y.backtest(R_te, D_te)

        spy_te = R_te[:, 0]
        bw_te = np.zeros(n_assets); bw_te[0] = 0.6; bw_te[1] = 0.4
        bench_te = (R_te * bw_te).sum(axis=1)

        ldm_c = ldm_yr['calmar']
        dm_c = dm_yr['calmar']
        spy_c = calmar(spy_te)
        bench_c = calmar(bench_te)
        beats_spy = "YES" if ldm_c > spy_c else "no"
        beats_dm = "YES" if ldm_c > dm_c else "no"

        print(f"  {year:<6} {ldm_c:>+10.3f} {dm_c:>+10.3f} "
              f"{spy_c:>+10.3f} {bench_c:>+10.3f} {beats_spy:>9} {beats_dm:>9}")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path("checkpoints").mkdir(exist_ok=True)
    np.savez("checkpoints/levered_dm.npz",
             net_rets=ldm_res['net_rets'], positions=ldm_res['positions'],
             leverage=ldm_res['leverage_history'],
             calmar=ldm_res['calmar'], sharpe=ldm_res['sharpe'])


if __name__ == "__main__":
    main()
