#!/usr/bin/env python3
"""
Phase 7: Regime-Switching Risk Premium — Full Backtest

Components:
  1. Cluster RSSM h_t on training data → 6 regimes
  2. Fit RegimeTransitionModel (smoothed transition + Kelly sizing)
  3. Walk-forward on test data: update beliefs, size positions, compute returns
  4. Generative vol surface from RSSM imagination
  5. Compare vs B&H, meta-controller, bandit, equal-weight

Usage:
    python train_regime_switching.py
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM
from regime_switching import (
    RegimeTransitionModel,
    GenerativeVolSurface,
    extract_rssm_with_internal_state,
)


def _max_drawdown(returns):
    cum = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def sharpe(returns):
    return returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--rssm-ckpt", default="checkpoints/SPY_rssm.pt")
    p.add_argument("--n-regimes", type=int, default=6)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--smooth-alpha", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--vol-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load data ───────────────────────────────────────────────────────────
    df = pd.read_csv(f"data/{args.ticker}_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]
    print(f"Data: {len(df)} total, {len(train_df)} train, {len(test_df)} test")

    # ── Extract states (needed for clustering and for walk-forward) ─────────
    print("\n[1/4] Extracting RSSM states...")
    h_full, z_full, rets_full, dates_full = extract_rssm_with_internal_state(
        args.rssm_ckpt, df, args.lookback, args.device
    )
    n_train = len(train_df) - args.lookback
    h_train = h_full[:n_train]
    z_train = z_full[:n_train]
    h_test = h_full[n_train:]
    z_test = z_full[n_train:]
    rets_train = rets_full[:n_train]
    rets_test = rets_full[n_train:]
    dates_test = dates_full[n_train:]
    print(f"  Train states: {len(h_train)}, Test states: {len(h_test)}")

    # ── Cluster training states ─────────────────────────────────────────────
    print(f"\n[2/4] Clustering RSSM states (K={args.n_regimes})...")
    scaler = StandardScaler()
    h_train_scaled = scaler.fit_transform(h_train)
    kmeans = KMeans(n_clusters=args.n_regimes, random_state=args.seed, n_init=10)
    train_labels = kmeans.fit_predict(h_train_scaled)

    # Profile regimes
    print(f"\n{'Regime':<8} {'Days':>6} {'AnnRet':>9} {'AnnVol':>9} {'Sharpe':>8} {'MaxDD':>8}")
    for c in range(args.n_regimes):
        mask = train_labels == c
        if mask.sum() < 5:
            continue
        r = rets_train[mask]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sh = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = _max_drawdown(r)
        print(f"  #{c:<7} {mask.sum():>6} {ann_ret:>+9.1%} {ann_vol:>9.1%} {sh:>+8.2f} {dd:>8.1%}")

    # Transition matrix
    print(f"\n  Transition stay rates:")
    trans = np.zeros((args.n_regimes, args.n_regimes))
    for t in range(len(train_labels) - 1):
        trans[train_labels[t], train_labels[t + 1]] += 1
    trans_prob = trans / trans.sum(axis=1, keepdims=True)
    for c in range(args.n_regimes):
        print(f"    #{c}: {trans_prob[c,c]:.1%} → #{np.argmax(trans_prob[c]):.0f} ({trans_prob[c].max():.1%})" 
              if np.argmax(trans_prob[c]) != c else
              f"    #{c}: {trans_prob[c,c]:.1%} stay")

    # ── Fit RegimeTransitionModel ───────────────────────────────────────────
    print(f"\n[3/4] Fitting RegimeTransitionModel (α={args.smooth_alpha})...")
    model = RegimeTransitionModel(
        n_regimes=args.n_regimes,
        smooth_alpha=args.smooth_alpha,
    )
    model.fit(train_labels, rets_train)

    # ── Walk-forward backtest on test data ──────────────────────────────────
    print(f"\n[4/4] Running walk-forward backtest (2022-2024)...")

    h_test_scaled = scaler.transform(h_test)
    test_labels = kmeans.predict(h_test_scaled)

    # Strategy positions
    positions = np.zeros(len(test_labels))
    daily_rets = np.zeros(len(test_labels))
    prev_position = 1.0  # start fully invested

    # Track realized vol for exposure cap
    realized_vol = 0.15  # initial estimate
    vol_lookback = 21
    ret_window = []

    for t in range(len(test_labels)):
        # Update regime belief
        model.update(test_labels[t])

        # Kelly position from belief
        kelly_raw = model.kelly_position(rf=0.02)

        # Conservative: cap position based on current realized vol
        # Target: 15% annual vol for the position
        vol_cap = 0.15 / (realized_vol + 1e-6)  # max position for 15% target vol
        target_pos = np.clip(kelly_raw, 0.0, min(1.5, vol_cap))

        # Crisis circuit breaker (regime #4 in training was a weird 34% vol outlier)
        if model.crisis_probability() > 0.03:
            target_pos *= 0.3  # reduce to 30%

        # Smooth position changes
        position = 0.6 * target_pos + 0.4 * prev_position
        position = np.clip(position, 0.0, 1.5)

        positions[t] = position
        daily_rets[t] = position * rets_test[t]
        prev_position = position

        # Update realized vol estimate
        ret_window.append(rets_test[t])
        if len(ret_window) > vol_lookback:
            ret_window.pop(0)
        if len(ret_window) >= 5:
            realized_vol = np.std(ret_window) * np.sqrt(252)

    # ── Results ─────────────────────────────────────────────────────────────
    rs_cum = np.prod(1 + daily_rets) - 1
    rs_sharpe = sharpe(daily_rets)
    rs_dd = _max_drawdown(daily_rets)

    bh_cum = np.prod(1 + rets_test) - 1
    bh_sharpe = sharpe(rets_test)
    bh_dd = _max_drawdown(rets_test)

    # Meta-controller comparison
    mc_positions = np.ones(len(test_labels))
    sorted_c = sorted(model.regime_stats.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    best_id = sorted_c[0][0]
    worst_id = sorted_c[-1][0]
    mc_rets = np.zeros(len(test_labels))
    for t, label in enumerate(test_labels):
        if label == best_id:
            mc_positions[t] = 1.0
        elif label == worst_id:
            mc_positions[t] = 0.25
        else:
            mc_positions[t] = 0.60
        mc_rets[t] = mc_positions[t] * rets_test[t]

    mc_cum = np.prod(1 + mc_rets) - 1
    mc_sharpe = sharpe(mc_rets)
    mc_dd = _max_drawdown(mc_rets)

    # ── Regime distribution on test ─────────────────────────────────────────
    unique, counts = np.unique(test_labels, return_counts=True)
    regime_dist = dict(zip(unique, counts))
    cash_days = (positions < 0.1).sum()
    crisis_days = (positions < 0.05).sum()

    print(f"\n{'='*70}")
    print(f"STRATEGY COMPARISON — SPY 2022-2024 ({len(test_labels)} days)")
    print(f"{'='*70}")
    print(f"  {'':<28} {'RegimeSwitch':>12} {'Meta-Ctrl':>12} {'B&H':>12}")
    print(f"  {'Cumulative Return':<28} {rs_cum:>+12.2%} {mc_cum:>+12.2%} {bh_cum:>+12.2%}")
    print(f"  {'Ann. Sharpe':<28} {rs_sharpe:>+12.3f} {mc_sharpe:>+12.3f} {bh_sharpe:>+12.3f}")
    print(f"  {'Max Drawdown':<28} {rs_dd:>12.2%} {mc_dd:>12.2%} {bh_dd:>12.2%}")
    print(f"  {'Mean Position':<28} {positions.mean():>12.2%} {mc_positions.mean():>12.0%} {'100%':>12}")
    print(f"  {'Cash Days':<28} {cash_days:>12} {'N/A':>12} {'0':>12}")

    # Calmar ratio
    rs_calmar = rs_cum / abs(rs_dd) if rs_dd != 0 else 0
    mc_calmar = mc_cum / abs(mc_dd) if mc_dd != 0 else 0
    bh_calmar = bh_cum / abs(bh_dd) if bh_dd != 0 else 0
    print(f"  {'Calmar Ratio':<28} {rs_calmar:>12.3f} {mc_calmar:>12.3f} {bh_calmar:>12.3f}")

    # ── Regime-switching specific stats ─────────────────────────────────────
    print(f"\n  Regime distribution on test: { {k: f'{v}' for k,v in sorted(regime_dist.items())} }")
    print(f"  Crisis circuit breaker triggered: {crisis_days} days")

    # Position stats per regime
    for c in range(args.n_regimes):
        mask = test_labels == c
        if mask.sum() > 0:
            print(f"  Regime #{c}: mean_pos={positions[mask].mean():.3f}, "
                  f"n={mask.sum()}, ann_ret={(positions[mask]*rets_test[mask]).mean()*252:+.2%}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    improvements = []
    if rs_sharpe > bh_sharpe + 0.05:
        improvements.append(f"Sharpe {rs_sharpe:.3f} > B&H {bh_sharpe:.3f}")
    if rs_dd > bh_dd:  # less negative = better
        improvements.append(f"MaxDD {rs_dd:.2%} better than B&H {bh_dd:.2%}")
    if rs_sharpe > mc_sharpe + 0.02:
        improvements.append(f"Outperforms meta-controller in Sharpe")

    if improvements:
        for imp in improvements:
            print(f"  ✓ {imp}")
    else:
        print(f"  ~ No significant improvement over benchmarks")

    if rs_sharpe > bh_sharpe:
        print(f"  ~ Marginal Sharpe edge ({rs_sharpe - bh_sharpe:+.3f})")
    else:
        print(f"  ✗ Underperforms B&H in Sharpe ({rs_sharpe - bh_sharpe:+.3f})")

    # Kelly insight
    avg_kelly = positions.mean()
    if avg_kelly < 0.3:
        print(f"  ⚠ Strategy is very conservative (avg pos {avg_kelly:.0%}) — regime model is risk-averse")
    elif avg_kelly > 1.0:
        print(f"  ⚠ Strategy uses leverage (avg pos {avg_kelly:.0%}) — check risk controls")

    # ── Generative Vol Surface (optional, slow) ─────────────────────────────
    if args.vol_samples > 0:
        print(f"\n[Bonus] Generating RSSM volatility surface ({args.vol_samples} trajectories × 30 days)...")
        ckpt = torch.load(args.rssm_ckpt, map_location='cpu', weights_only=False)
        cfg = ckpt.get("config", {})
        encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128))
        rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128),
                    cfg.get("latent_dim", 32))
        encoder.load_state_dict(ckpt["encoder_state"])
        rssm.load_state_dict(ckpt["rssm_state"])
        encoder.eval(); rssm.eval()

        vol_surface = GenerativeVolSurface(
            rssm, encoder,
            horizon=30,
            n_trajectories=args.vol_samples,
            device=args.device,
        )

        # Use last day of test data for vol forecast
        tech = torch.tensor(test_df[['Open', 'Close', 'Volume']].values, dtype=torch.float32)
        fund = torch.tensor(test_df[['ROE', 'Debt_Ratio']].values, dtype=torch.float32)
        acts = torch.tensor(test_df[['US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                                     'US10Y_Volatility', 'is_earnings_day',
                                     'Earnings_Surprise']].values, dtype=torch.float32)

        t_last = len(test_df) - 1
        tech_seq = tech[t_last - args.lookback:t_last]
        fund_t = fund[t_last]
        action_t = acts[t_last]
        h_last = torch.tensor(h_test[-1])
        z_last = torch.tensor(z_test[-1])

        forecast = vol_surface.forecast_vol(
            tech_seq, fund_t, action_t, h_last, z_last,
            n_samples=args.vol_samples,
        )

        vix_last = test_df['VIX'].values[-1]
        model_vol = forecast["implied_vol_30d"]
        signal = vol_surface.vol_arbitrage_signal(model_vol, vix_last / 100.0)

        print(f"  RSSM-implied 30-day vol: {model_vol:.2%}")
        print(f"  VIX (market vol):        {vix_last:.1f}%")
        print(f"  Vol ratio:               {model_vol / (vix_last/100):.3f}")
        print(f"  Signal:                  {signal}")
        print(f"  Trajectory range:        [{forecast['vol_percentiles']['p10']:+.3%}, "
              f"{forecast['vol_percentiles']['p90']:+.3%}]")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    np.savez("checkpoints/regime_switching_results.npz",
             positions=positions, daily_rets=daily_rets,
             rs_sharpe=rs_sharpe, rs_cum=rs_cum, rs_dd=rs_dd,
             test_labels=test_labels)


if __name__ == "__main__":
    main()
