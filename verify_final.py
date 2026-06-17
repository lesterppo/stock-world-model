#!/usr/bin/env python3
"""Comprehensive walk-forward: soft-blend regime allocator vs benchmarks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from macro_allocator import MacroRegimeAllocator, extract_macro_features, sharpe, _dd, calmar

def main():
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)
    lookback = 60
    n_regimes = 6
    tc_bps = 5.0

    features = extract_macro_features(df, lookback)
    rets_all = np.zeros((len(features), n_assets), dtype=np.float32)
    for i, t in enumerate(tickers):
        rets_all[:, i] = df[f'{t}_Next_Return'].values[lookback:]

    n_initial = 252 * 5

    print(f"{'Year':<6} {'Soft Calmar':>12} {'Hard Calmar':>12} {'EW Calmar':>12} {'RP Calmar':>12} {'Soft>EW?':>9} {'N_Reg':>6}")
    print("-" * 80)

    for year in range(2016, 2025):
        train_end = df.index.searchsorted(pd.Timestamp(f"{year-1}-12-31")) - 1
        test_start = df.index.searchsorted(pd.Timestamp(f"{year}-01-02"))
        test_end = min(df.index.searchsorted(pd.Timestamp(f"{year}-12-31")), len(df) - 1)
        if train_end < 0: train_end = 0
        train_end_lb = train_end - lookback
        test_start_lb = test_start - lookback
        test_end_lb = test_end - lookback
        if test_end_lb - test_start_lb < 50: continue

        F_tr, R_tr = features[:train_end_lb], rets_all[:train_end_lb]
        F_te, R_te = features[test_start_lb:test_end_lb], rets_all[test_start_lb:test_end_lb]

        # Soft-blend allocator
        alloc_soft = MacroRegimeAllocator(
            n_regimes=n_regimes, tc_bps=tc_bps, slippage_bps=1.0,
            n_ensemble=3, velocity_cap=0.3, soft_assign=True, soft_temperature=1.0,
        )
        alloc_soft.fit(F_tr, R_tr)
        _, net_soft = alloc_soft.predict_positions(F_te, R_te)
        sc = calmar(net_soft)

        # Hard assignment allocator
        alloc_hard = MacroRegimeAllocator(
            n_regimes=n_regimes, tc_bps=tc_bps, slippage_bps=1.0,
            n_ensemble=3, velocity_cap=0.3, soft_assign=False,
        )
        alloc_hard.fit(F_tr, R_tr)
        _, net_hard = alloc_hard.predict_positions(F_te, R_te)
        hc = calmar(net_hard)

        # Equal weight
        ew = R_te.mean(axis=1)
        ec = calmar(ew)

        # Risk parity
        vols = np.array([np.nanstd(R_te[:, i]) * np.sqrt(252) for i in range(n_assets)])
        rp_w = (1.0 / (vols + 1e-8)) / np.sum(1.0 / (vols + 1e-8))
        rp = (R_te * rp_w).sum(axis=1)
        rc = calmar(rp)

        beats = "YES" if sc > ec else "no"
        # Count distinct labels
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        F_tr_s = scaler.fit_transform(F_tr)
        km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        km.fit(F_tr_s)
        F_te_s = scaler.transform(F_te)
        n_reg = len(set(km.predict(F_te_s)))

        print(f"{year:<6} {sc:>+12.3f} {hc:>+12.3f} {ec:>+12.3f} {rc:>+12.3f} {beats:>9} {n_reg:>6}")

    # ── Full-sample comparison 2022-2024 ─────────────────────────────────
    print(f"\n{'='*70}")
    print("FULL-SAMPLE 2022-2024 (train 2000-2021)")
    print(f"{'='*70}")

    train_end = df.index.searchsorted(pd.Timestamp("2021-12-31")) - 1
    test_start = df.index.searchsorted(pd.Timestamp("2022-01-02"))
    train_end_lb = train_end - lookback
    test_start_lb = test_start - lookback

    F_tr, R_tr = features[:train_end_lb], rets_all[:train_end_lb]
    F_te, R_te = features[test_start_lb:], rets_all[test_start_lb:]

    alloc = MacroRegimeAllocator(
        n_regimes=n_regimes, tc_bps=6.0, slippage_bps=0.0,
        n_ensemble=5, velocity_cap=0.3, soft_assign=True, soft_temperature=1.0,
    )
    alloc.fit(F_tr, R_tr)
    positions, net_macro = alloc.predict_positions(F_te, R_te)

    ew = R_te.mean(axis=1)
    vols = np.array([np.nanstd(R_te[:, i]) * np.sqrt(252) for i in range(n_assets)])
    rp_w = (1.0 / (vols + 1e-8)) / np.sum(1.0 / (vols + 1e-8))
    rp = (R_te * rp_w).sum(axis=1)
    bw = np.zeros(n_assets); bw[0] = 0.6; bw[1] = 0.4
    bench = (R_te * bw).sum(axis=1)
    spy = R_te[:, 0]

    print(f"  {'Strategy':<20} {'Calmar':>8} {'Sharpe':>8} {'CumRet':>10} {'MaxDD':>10}")
    for name, rets in [("Macro (soft-blend)", net_macro), ("Equal Weight", ew),
                        ("Risk Parity", rp), ("60/40 SPY/TLT", bench), ("SPY only", spy)]:
        c = calmar(rets); s = sharpe(rets)
        cum = np.prod(1 + rets) - 1; dd = _dd(rets)
        print(f"  {name:<20} {c:>+8.3f} {s:>+8.3f} {cum:>+10.2%} {dd:>10.2%}")

    # Position diversity
    print(f"\n  Position range: SPY [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"TLT [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")

if __name__ == "__main__":
    main()
