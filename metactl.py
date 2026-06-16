#!/usr/bin/env python3
"""
Market Weather Meta-Controller — latent regime detection + risk management.

Steps:
  1. Extract RSSM deterministic states h_t from real SPY data
  2. Cluster h_t with K-Means → discover latent market regimes
  3. Profile each regime (return, vol, max drawdown)
  4. Backtest a rule-based meta-controller: regime-aware position sizing

Usage:
    python metactl.py --ticker SPY --checkpoint checkpoints/SPY_rssm.pt --n-clusters 4
"""

import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM


def extract_states(checkpoint_path, df, lookback=60, device="cpu"):
    """Walk through SPY data, encode each day through RSSM, return h_t states."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128)).to(device)
    rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128),
                cfg.get("latent_dim", 32)).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    rssm.load_state_dict(ckpt["rssm_state"])
    encoder.eval(); rssm.eval()

    tech = torch.tensor(df[['Open','Close','Volume']].values, dtype=torch.float32)
    fund = torch.tensor(df[['ROE','Debt_Ratio']].values, dtype=torch.float32)
    acts = torch.tensor(df[['US10Y','Yield_Spread','VIX','VIX_1w_Change',
                           'US10Y_Volatility','is_earnings_day',
                           'Earnings_Surprise']].values, dtype=torch.float32)

    n = len(df)
    h_states = np.zeros((n - lookback, cfg.get("hidden_dim", 128)), dtype=np.float32)
    z_vars = np.zeros(n - lookback, dtype=np.float32)  # aleatoric uncertainty
    dates = df.index[lookback:]

    h_t, z_t = rssm.initial_state(1, torch.device(device))
    with torch.no_grad():
        for t in range(lookback, n):
            idx = t - lookback
            tw = tech[t-lookback:t].unsqueeze(0).to(device)
            fw = fund[t].unsqueeze(0).to(device)
            e_t = encoder(tw, fw)
            a_prev = acts[t].unsqueeze(0).to(device)
            out = rssm.observe_step(h_t, z_t, a_prev, e_t)
            h_t, z_t = out["h_t"], out["z_t"]
            h_states[idx] = h_t.cpu().numpy().squeeze(0)
            z_vars[idx] = torch.exp(out["post_logvar"]).mean().item()  # avg aleatoric var

    return h_states, z_vars, dates


def cluster_and_profile(h_states, z_vars, returns, dates, n_clusters=4, seed=42):
    """Cluster latent states, compute per-cluster market statistics."""
    # Standardize
    scaler = StandardScaler()
    h_scaled = scaler.fit_transform(h_states)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(h_scaled)

    # Profile each cluster
    profiles = {}
    for c in range(n_clusters):
        mask = labels == c
        n_days = mask.sum()
        if n_days < 10:
            continue
        rets = returns[mask]
        vol = rets.std() * np.sqrt(252)
        ann_ret = rets.mean() * 252
        sharpe = ann_ret / vol if vol > 0 else 0
        max_dd = _max_drawdown(rets)
        avg_var = z_vars[mask].mean()
        profiles[c] = {
            "n_days": n_days,
            "ann_return": ann_ret,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_aleatoric_var": avg_var,
        }

    # Sort by Sharpe
    sorted_clusters = sorted(profiles.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    return labels, profiles, sorted_clusters, scaler, kmeans


def _max_drawdown(returns):
    """Max drawdown from return series (array or Series)."""
    cum = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def backtest_metactl(
    labels, profiles, sorted_clusters, z_vars, returns, dates,
    var_threshold_pct=90,  # top N% variance = uncertainty signal
):
    """
    Meta-controller backtest:
      - Best regime (highest Sharpe): 100% long SPY
      - Worst regime (lowest Sharpe): 25% long (reduced exposure)
      - Middle regimes: 50-75% long
      - Uncertainty signal (top var_threshold_pct variance): 0% (cash)
    """
    best_id = sorted_clusters[0][0]
    worst_id = sorted_clusters[-1][0]

    # Determine variance threshold for uncertainty signal
    var_threshold = np.percentile(z_vars, var_threshold_pct)

    positions = np.ones(len(returns))  # default: fully long
    daily_rets = np.zeros(len(returns))

    for i, (label, var, ret) in enumerate(zip(labels, z_vars, returns)):
        # Epistemic-style veto: high aleatoric variance → cash
        if var > var_threshold:
            positions[i] = 0.0
        elif label == best_id:
            positions[i] = 1.0
        elif label == worst_id:
            positions[i] = 0.25
        else:
            positions[i] = 0.60

        daily_rets[i] = positions[i] * ret

    return positions, daily_rets


def main():
    p = argparse.ArgumentParser(description="Market Weather Meta-Controller")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--checkpoint", default="checkpoints/SPY_rssm.pt")
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--var-threshold-pct", type=float, default=90)
    args = p.parse_args()

    # Load data
    df = pd.read_csv(f"data/{args.ticker}_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]
    returns_test = test_df['Next_Day_Return'].values
    dates_test = test_df.index

    print(f"Data: {len(train_df)} train, {len(test_df)} test days")

    # ── Step 1: Extract latent states from TRAIN data (for clustering) ──────
    print(f"\n[1/4] Extracting RSSM latent states from training data...")
    h_train, z_train, dates_train = extract_states(args.checkpoint, train_df, args.lookback)

    # ── Step 2: Cluster & Profile ────────────────────────────────────────────
    print(f"[2/4] Clustering latent states (K={args.n_clusters})...")
    train_rets = train_df['Next_Day_Return'].values[args.lookback:]
    labels_train, profiles, sorted_clusters, scaler, kmeans = cluster_and_profile(
        h_train, z_train, train_rets, dates_train, args.n_clusters
    )

    print(f"\n{'='*70}")
    print(f"LATENT MARKET REGIMES (discovered from 2010-2021 training data)")
    print(f"{'='*70}")
    print(f"{'Cluster':<8} {'Days':>6} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>8} {'Max DD':>8} {'Avg Var':>8}")
    for c_id, p in sorted_clusters:
        label = f"#{c_id}"
        if c_id == sorted_clusters[0][0]:
            label += " ★"  # best
        print(f"{label:<8} {p['n_days']:>6} {p['ann_return']:>+8.1%} {p['ann_vol']:>8.1%} "
              f"{p['sharpe']:>+8.2f} {p['max_drawdown']:>8.1%} {p['avg_aleatoric_var']:>8.4f}")

    # ── Step 3: Apply to TEST data ───────────────────────────────────────────
    print(f"\n[3/4] Extracting latent states from TEST data...")
    h_test, z_test, _ = extract_states(args.checkpoint, test_df, args.lookback)
    h_test_scaled = scaler.transform(h_test)
    labels_test = kmeans.predict(h_test_scaled)

    # Trim returns to match (first lookback days have no state)
    returns_test_trimmed = returns_test[args.lookback:]
    dates_test_trimmed = dates_test[args.lookback:]

    # ── Step 4: Meta-Controller Backtest ─────────────────────────────────────
    print(f"[4/4] Running Meta-Controller backtest...")
    positions, daily_rets = backtest_metactl(
        labels_test, profiles, sorted_clusters, z_test,
        returns_test_trimmed, dates_test_trimmed, args.var_threshold_pct,
    )

    # Metrics
    mc_cum = (1 + daily_rets).cumprod()
    bh_cum = (1 + returns_test_trimmed).cumprod()
    mc_sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252)
    bh_sharpe = returns_test_trimmed.mean() / (returns_test_trimmed.std() + 1e-8) * np.sqrt(252)
    mc_dd = _max_drawdown(daily_rets)
    bh_dd = _max_drawdown(returns_test_trimmed)
    mc_total = mc_cum[-1] - 1
    bh_total = bh_cum[-1] - 1

    # Regime distribution on test
    unique, counts = np.unique(labels_test, return_counts=True)
    regime_dist = dict(zip(unique, counts))

    cash_days = (positions == 0).sum()
    total_days = len(positions)

    print(f"\n{'='*70}")
    print(f"META-CONTROLLER BACKTEST (Test Set {dates_test_trimmed[0].date()} — {dates_test_trimmed[-1].date()})")
    print(f"{'='*70}")
    print(f"  Regime distribution: { {k: f'{v} days' for k,v in regime_dist.items()} }")
    print(f"  Circuit breaker (cash): {cash_days}/{total_days} days ({cash_days/total_days:.1%})")
    print()
    print(f"  {'':<25} {'Meta-Controller':>15} {'Buy & Hold':>15}")
    print(f"  {'Cumulative Return':<25} {mc_total:>+15.2%} {bh_total:>+15.2%}")
    print(f"  {'Ann. Sharpe':<25} {mc_sharpe:>+15.3f} {bh_sharpe:>+15.3f}")
    print(f"  {'Max Drawdown':<25} {mc_dd:>15.2%} {bh_dd:>15.2%}")
    print(f"  {'Mean Position':<25} {positions.mean():>15.2%} {'N/A':>15}")

    improvement = mc_sharpe - bh_sharpe
    if improvement > 0.05:
        print(f"\n  ✓ Meta-Controller improves Sharpe by {improvement:+.3f}")
    elif improvement > 0:
        print(f"\n  ~ Marginal improvement: {improvement:+.3f}")
    else:
        print(f"\n  ✗ No improvement: {improvement:+.3f}")

    if mc_dd > bh_dd:
        print(f"  ✗ Drawdown worse: {mc_dd:.2%} vs {bh_dd:.2%}")
    else:
        print(f"  ✓ Drawdown reduced by {bh_dd - mc_dd:.2%}")

    print(f"\nProfiles saved in memory. Use --n-clusters to adjust granularity.")


if __name__ == "__main__":
    main()
