#!/usr/bin/env python3
"""
Phase 6: Honest Backtest — Contextual Bandit vs Meta-Controller vs Buy & Hold

Runs all three strategies on the SAME test set (2022-2024) and compares.

Usage:
    python backtest_bandit.py --ticker SPY --rssm-ckpt checkpoints/SPY_rssm.pt --bandit-ckpt checkpoints/bandit_policy.pt
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
from contextual_bandit import BanditPolicy, extract_rssm_states_with_context


def _max_drawdown(returns):
    cum = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def sharpe(returns):
    return returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)


def evaluate_bandit(policy, h_test, ret_ctx_test, rets_test, smooth=0.7):
    """Deterministic walk-forward evaluation for bandit."""
    device = next(policy.parameters()).device
    positions = np.zeros(len(h_test))
    daily_rets = np.zeros(len(h_test))
    prev_w = 0.5  # start at neutral

    with torch.no_grad():
        for t in range(len(h_test)):
            h = torch.tensor(h_test[t:t + 1], device=device)
            ctx = torch.tensor(ret_ctx_test[t:t + 1], device=device)
            w = policy.get_action_deterministic(h, ctx).item()
            w_smooth = smooth * w + (1 - smooth) * prev_w
            positions[t] = w_smooth
            daily_rets[t] = w_smooth * rets_test[t]
            prev_w = w_smooth

    return positions, daily_rets


def evaluate_metactl(
    checkpoint_path, test_df, lookback=60,
    n_clusters=6, var_threshold_pct=90,
):
    """
    Re-implement meta-controller evaluation in-process.
    Returns positions and daily returns.
    """
    device = "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128)).to(device)
    rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128),
                cfg.get("latent_dim", 32)).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    rssm.load_state_dict(ckpt["rssm_state"])
    encoder.eval(); rssm.eval()

    # Extract states for test data
    tech = torch.tensor(test_df[['Open', 'Close', 'Volume']].values, dtype=torch.float32)
    fund = torch.tensor(test_df[['ROE', 'Debt_Ratio']].values, dtype=torch.float32)
    acts = torch.tensor(test_df[['US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                                'US10Y_Volatility', 'is_earnings_day',
                                'Earnings_Surprise']].values, dtype=torch.float32)
    rets_test = test_df['Next_Day_Return'].values

    n = len(test_df)
    h_states = np.zeros((n - lookback, cfg.get("hidden_dim", 128)), dtype=np.float32)

    h_t, z_t = rssm.initial_state(1, torch.device(device))
    with torch.no_grad():
        for t in range(lookback, n):
            idx = t - lookback
            tw = tech[t - lookback:t].unsqueeze(0).to(device)
            fw = fund[t].unsqueeze(0).to(device)
            e_t = encoder(tw, fw)
            a_prev = acts[t].unsqueeze(0).to(device)
            out = rssm.observe_step(h_t, z_t, a_prev, e_t)
            h_t, z_t = out["h_t"], out["z_t"]
            h_states[idx] = h_t.cpu().numpy().squeeze(0)

    # For fair comparison, use the same clustering trained on train data
    # We load the train data to fit clustering
    train_df = pd.read_csv("data/SPY_fused.csv", index_col=0, parse_dates=True)[:'2021-12-31']

    # Extract train states for clustering
    tech_tr = torch.tensor(train_df[['Open', 'Close', 'Volume']].values, dtype=torch.float32)
    fund_tr = torch.tensor(train_df[['ROE', 'Debt_Ratio']].values, dtype=torch.float32)
    acts_tr = torch.tensor(train_df[['US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                                     'US10Y_Volatility', 'is_earnings_day',
                                     'Earnings_Surprise']].values, dtype=torch.float32)
    n_tr = len(train_df)
    h_train = np.zeros((n_tr - lookback, cfg.get("hidden_dim", 128)), dtype=np.float32)

    h_t, z_t = rssm.initial_state(1, torch.device(device))
    with torch.no_grad():
        for t in range(lookback, n_tr):
            idx = t - lookback
            tw = tech_tr[t - lookback:t].unsqueeze(0).to(device)
            fw = fund_tr[t].unsqueeze(0).to(device)
            e_t = encoder(tw, fw)
            a_prev = acts_tr[t].unsqueeze(0).to(device)
            out = rssm.observe_step(h_t, z_t, a_prev, e_t)
            h_t, z_t = out["h_t"], out["z_t"]
            h_train[idx] = h_t.cpu().numpy().squeeze(0)

    # Cluster on train
    scaler = StandardScaler()
    h_train_scaled = scaler.fit_transform(h_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_train = kmeans.fit_predict(h_train_scaled)

    # Profile clusters
    train_rets = train_df['Next_Day_Return'].values[lookback:]
    profiles = {}
    for c in range(n_clusters):
        mask = labels_train == c
        if mask.sum() < 10:
            continue
        rets_c = train_rets[mask]
        profiles[c] = {
            "sharpe": rets_c.mean() / (rets_c.std() + 1e-8) * np.sqrt(252),
        }

    sorted_c = sorted(profiles.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    best_id = sorted_c[0][0]
    worst_id = sorted_c[-1][0]

    # Predict on test
    h_test_scaled = scaler.transform(h_states)
    labels_test = kmeans.predict(h_test_scaled)

    # Meta-controller rules
    rets_trimmed = rets_test[lookback:]
    positions = np.ones(len(rets_trimmed))
    daily_rets = np.zeros(len(rets_trimmed))

    for i, label in enumerate(labels_test):
        if label == best_id:
            positions[i] = 1.0
        elif label == worst_id:
            positions[i] = 0.25
        else:
            positions[i] = 0.60
        daily_rets[i] = positions[i] * rets_trimmed[i]

    return positions, daily_rets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--rssm-ckpt", default="checkpoints/SPY_rssm.pt")
    p.add_argument("--bandit-ckpt", default="checkpoints/bandit_policy.pt")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # ── Load data ───────────────────────────────────────────────────────────
    df = pd.read_csv(f"data/{args.ticker}_fused.csv", index_col=0, parse_dates=True)
    test_df = df['2022-01-01':]
    print(f"Test data: {len(test_df)} days ({test_df.index[0].date()} — {test_df.index[-1].date()})")

    # ── Extract test states ─────────────────────────────────────────────────
    h_test, ret_ctx_test, rets_test, dates_test = extract_rssm_states_with_context(
        args.rssm_ckpt, test_df, args.lookback, args.device
    )
    rets_test_raw = test_df['Next_Day_Return'].values
    rets_trimmed = rets_test_raw[args.lookback:]

    # ── Strategy 1: Buy & Hold ──────────────────────────────────────────────
    bh_rets = rets_trimmed
    bh_cum = np.prod(1 + bh_rets) - 1
    bh_sharpe = sharpe(bh_rets)
    bh_dd = _max_drawdown(bh_rets)

    # ── Strategy 2: Meta-Controller ─────────────────────────────────────────
    try:
        mc_positions, mc_rets = evaluate_metactl(
            args.rssm_ckpt, test_df, args.lookback, args.n_clusters,
        )
        mc_cum = np.prod(1 + mc_rets) - 1
        mc_sharpe = sharpe(mc_rets)
        mc_dd = _max_drawdown(mc_rets)
        mc_mean_pos = float(np.mean(mc_positions))
    except Exception as e:
        print(f"  Meta-controller evaluation failed: {e}")
        mc_cum, mc_sharpe, mc_dd, mc_mean_pos = 0, 0, 0, 0

    # ── Strategy 3: Contextual Bandit ───────────────────────────────────────
    try:
        policy = BanditPolicy(state_dim=128).to(device)
        ckpt = torch.load(args.bandit_ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state"])
        policy.eval()

        for smooth in [0.5, 0.7, 0.9]:
            _, bandit_rets = evaluate_bandit(policy, h_test, ret_ctx_test, rets_test, smooth=smooth)
            b_sharpe = sharpe(np.array(bandit_rets))
            print(f"  Bandit (smooth={smooth}): Sharpe={b_sharpe:+.4f}")

        # Use best smoothing from validation (default 0.7)
        bandit_positions, bandit_rets = evaluate_bandit(policy, h_test, ret_ctx_test, rets_test, smooth=0.7)
        bandit_cum = np.prod(1 + bandit_rets) - 1
        bandit_sharpe = sharpe(np.array(bandit_rets))
        bandit_dd = _max_drawdown(bandit_rets)
        bandit_mean_pos = float(np.mean(bandit_positions))
        bandit_cash_days = int((bandit_positions < 0.1).sum())
    except FileNotFoundError:
        print(f"  Bandit checkpoint not found: {args.bandit_ckpt}")
        bandit_cum = bandit_sharpe = bandit_dd = bandit_mean_pos = 0
        bandit_cash_days = 0

    # ── Comparison Table ────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"STRATEGY COMPARISON — SPY 2022-2024 ({len(rets_trimmed)} days)")
    print(f"{'='*75}")
    print(f"  {'':<25} {'Buy & Hold':>15} {'Meta-Ctrl':>15} {'Bandit':>15}")
    print(f"  {'Cumulative Return':<25} {bh_cum:>+15.2%} {mc_cum:>+15.2%} {bandit_cum:>+15.2%}")
    print(f"  {'Ann. Sharpe':<25} {bh_sharpe:>+15.3f} {mc_sharpe:>+15.3f} {bandit_sharpe:>+15.3f}")
    print(f"  {'Max Drawdown':<25} {bh_dd:>15.2%} {mc_dd:>15.2%} {bandit_dd:>15.2%}")
    print(f"  {'Mean Position':<25} {'100%':>15} {mc_mean_pos:>15.0%} {bandit_mean_pos:>15.0%}")
    print(f"  {'Cash Days (<10%)':<25} {'0':>15} {'N/A':>15} {bandit_cash_days:>15}")

    # ── Risk-Adjusted Metric ────────────────────────────────────────────────
    # Calmar ratio: return / |max_dd|
    bh_calmar = bh_cum / abs(bh_dd) if bh_dd != 0 else 0
    mc_calmar = mc_cum / abs(mc_dd) if mc_dd != 0 else 0
    bandit_calmar = bandit_cum / abs(bandit_dd) if bandit_dd != 0 else 0

    print(f"  {'Calmar Ratio':<25} {bh_calmar:>15.3f} {mc_calmar:>15.3f} {bandit_calmar:>15.3f}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("VERDICT")
    print(f"{'='*75}")

    winner_sharpe = max(bh_sharpe, mc_sharpe, bandit_sharpe)
    if bandit_sharpe == winner_sharpe:
        print(f"  ✓ Contextual Bandit has the highest Sharpe ({bandit_sharpe:+.4f})")
    elif mc_sharpe == winner_sharpe:
        print(f"  ~ Meta-Controller has the highest Sharpe ({mc_sharpe:+.4f})")
    else:
        print(f"  ~ Buy & Hold has the highest Sharpe ({bh_sharpe:+.4f})")

    winner_calmar = max(bh_calmar, mc_calmar, bandit_calmar)
    if bandit_calmar == winner_calmar:
        print(f"  ✓ Contextual Bandit has the highest Calmar ratio ({bandit_calmar:+.3f})")
    elif mc_calmar == winner_calmar:
        print(f"  ✓ Meta-Controller has the highest Calmar ratio ({mc_calmar:+.3f})")

    if bandit_dd < min(bh_dd, mc_dd):
        print(f"  ✓ Contextual Bandit has the lowest Max DD ({bandit_dd:.2%})")

    # Bandit-specific insights
    if bandit_mean_pos > 0.8:
        print(f"  ⚠ Bandit stays mostly invested ({bandit_mean_pos:.0%}) — little market timing signal")
    elif bandit_mean_pos < 0.3:
        print(f"  ⚠ Bandit stays mostly in cash ({bandit_mean_pos:.0%}) — too conservative")

    if bandit_sharpe > bh_sharpe + 0.1:
        print(f"  ✓ Bandit meaningfully improves Sharpe over B&H (+{bandit_sharpe - bh_sharpe:+.3f})")
    elif bandit_sharpe > bh_sharpe:
        print(f"  ~ Marginal Sharpe improvement (+{bandit_sharpe - bh_sharpe:+.3f})")
    else:
        print(f"  ✗ Bandit underperforms B&H in Sharpe ({bandit_sharpe - bh_sharpe:+.3f})")


if __name__ == "__main__":
    main()
