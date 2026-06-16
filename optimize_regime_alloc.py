#!/usr/bin/env python3
"""
Phase 7b: Pure Regime Allocation Optimization

Accept that regime detection is the only thing that works.
Optimize per-regime allocations via walk-forward cross-validation.
No Kelly, no belief smoothing, no circuit breakers — just:
  "In regime X, allocate W_X based on what worked historically."

Plus: generative vol surface vs VIX comparison over time.

Usage:
    python optimize_regime_alloc.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM


def _max_drawdown(returns):
    cum = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def sharpe(returns):
    return returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)


def optimize_allocations(labels, returns, n_regimes, metric="sharpe"):
    """
    Find optimal per-regime allocations via grid search.
    Each regime gets a weight ∈ {0.0, 0.25, 0.50, 0.75, 1.00}.
    """
    weights_grid = [0.0, 0.25, 0.50, 0.75, 1.00]
    best_score = -float("inf")
    best_weights = None

    # For 6 regimes with 5 options each = 5^6 = 15,625 combinations
    # Too many for grid search. Use greedy instead.
    weights = np.ones(n_regimes) * 0.5  # start at neutral

    for _ in range(20):  # 20 iterations of coordinate ascent
        improved = False
        for c in range(n_regimes):
            best_w = weights[c]
            best_local = -float("inf")
            for w in weights_grid:
                test_weights = weights.copy()
                test_weights[c] = w
                # Compute strategy returns with these weights
                strat_rets = np.array([test_weights[label] * ret for label, ret in zip(labels, returns)])
                if metric == "sharpe":
                    score = sharpe(strat_rets)
                elif metric == "calmar":
                    cum = np.prod(1 + strat_rets) - 1
                    dd = _max_drawdown(strat_rets)
                    score = cum / abs(dd) if dd != 0 else 0
                else:
                    score = np.prod(1 + strat_rets) - 1  # total return

                if score > best_local:
                    best_local = score
                    best_w = w

            if best_w != weights[c]:
                weights[c] = best_w
                improved = True

        if not improved:
            break

    return weights


def main():
    # ── Load data & states ──────────────────────────────────────────────────
    df = pd.read_csv("data/SPY_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]

    ckpt = torch.load("checkpoints/SPY_rssm.pt", map_location='cpu', weights_only=False)
    cfg = ckpt.get("config", {})
    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128))
    rssm_obj = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128), cfg.get("latent_dim", 32))
    encoder.load_state_dict(ckpt["encoder_state"])
    rssm_obj.load_state_dict(ckpt["rssm_state"])
    encoder.eval(); rssm_obj.eval()

    lookback = 60
    n_regimes = 6

    # Extract states for full dataset
    def extract_for_df(data_df):
        tech = torch.tensor(data_df[['Open','Close','Volume']].values, dtype=torch.float32)
        fund = torch.tensor(data_df[['ROE','Debt_Ratio']].values, dtype=torch.float32)
        acts = torch.tensor(data_df[['US10Y','Yield_Spread','VIX','VIX_1w_Change',
                                     'US10Y_Volatility','is_earnings_day',
                                     'Earnings_Surprise']].values, dtype=torch.float32)
        n = len(data_df)
        h = np.zeros((n - lookback, 128), dtype=np.float32)
        h_t, z_t = rssm_obj.initial_state(1, torch.device('cpu'))
        with torch.no_grad():
            for t in range(lookback, n):
                tw = tech[t-lookback:t].unsqueeze(0)
                fw = fund[t].unsqueeze(0)
                e_t = encoder(tw, fw)
                a_prev = acts[t].unsqueeze(0)
                out = rssm_obj.observe_step(h_t, z_t, a_prev, e_t)
                h_t, z_t = out["h_t"], out["z_t"]
                h[t-lookback] = h_t.cpu().numpy().squeeze(0)
        return h, data_df['Next_Day_Return'].values[lookback:], data_df.index[lookback:]

    h_train, rets_train, _ = extract_for_df(train_df)
    h_test, rets_test, dates_test = extract_for_df(test_df)

    print(f"Train: {len(h_train)} states, Test: {len(h_test)} states")

    # ── Walk-forward CV for optimal allocations ─────────────────────────────
    print(f"\n{'='*60}")
    print("WALK-FORWARD REGIME ALLOCATION OPTIMIZATION")
    print(f"{'='*60}")

    # Split train into sub-training and validation chronologically
    n_train = len(h_train)
    folds = 5
    val_size = n_train // (folds * 2)
    min_train = n_train // 2

    fold_weights = []
    fold_results = []

    for fold in range(folds):
        train_end = min_train + fold * (n_train - min_train) // folds
        val_start = train_end
        val_end = min(val_start + val_size, n_train - 1)

        # Fit K-Means on sub-train
        scaler = StandardScaler()
        h_sub_scaled = scaler.fit_transform(h_train[:train_end])
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(h_sub_scaled)

        # Optimize allocations on sub-train
        sub_rets = rets_train[:train_end]
        weights = optimize_allocations(sub_labels, sub_rets, n_regimes, metric="sharpe")

        # Test on validation
        h_val_scaled = scaler.transform(h_train[val_start:val_end])
        val_labels = kmeans.predict(h_val_scaled)
        val_rets_raw = rets_train[val_start:val_end]
        val_strat = np.array([weights[label] for label in val_labels])
        val_rets = val_strat * val_rets_raw

        val_cum = np.prod(1 + val_rets) - 1
        val_sh = sharpe(val_rets)
        val_dd = _max_drawdown(val_rets)

        fold_weights.append(weights)
        fold_results.append({"cum": val_cum, "sharpe": val_sh, "max_dd": val_dd})

        print(f"  Fold {fold+1}: weights={ {c: f'{w:.2f}' for c,w in enumerate(weights)} } | "
              f"Val Sharpe={val_sh:+.3f} CumRet={val_cum:+.2%}")

    # ── Consensus weights (median across folds) ─────────────────────────────
    consensus = np.median(fold_weights, axis=0)
    print(f"\n  Consensus weights: { {c: f'{w:.2f}' for c,w in enumerate(consensus)} }")
    print(f"  CV Sharpe: {np.mean([r['sharpe'] for r in fold_results]):+.3f} ± "
          f"{np.std([r['sharpe'] for r in fold_results]):.3f}")

    # ── Final: cluster full train, apply consensus to test ──────────────────
    scaler_full = StandardScaler()
    h_train_scaled = scaler_full.fit_transform(h_train)
    kmeans_full = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    train_labels = kmeans_full.fit_predict(h_train_scaled)

    # Get optimized weights on full training set
    opt_weights = optimize_allocations(train_labels, rets_train, n_regimes, metric="sharpe")

    # Apply to test
    h_test_scaled = scaler_full.transform(h_test)
    test_labels = kmeans_full.predict(h_test_scaled)

    positions = np.array([opt_weights[label] for label in test_labels])
    strat_rets = positions * rets_test

    strat_cum = np.prod(1 + strat_rets) - 1
    strat_sh = sharpe(strat_rets)
    strat_dd = _max_drawdown(strat_rets)

    bh_cum = np.prod(1 + rets_test) - 1
    bh_sh = sharpe(rets_test)
    bh_dd = _max_drawdown(rets_test)

    # Original meta-controller weights
    mc_weights = [0.60, 0.60, 0.60, 1.0, 0.60, 0.25]  # hand-tuned
    mc_positions = np.array([mc_weights[label] for label in test_labels])
    mc_rets = mc_positions * rets_test
    mc_cum = np.prod(1 + mc_rets) - 1
    mc_sh = sharpe(mc_rets)
    mc_dd = _max_drawdown(mc_rets)

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL STRATEGY COMPARISON — SPY 2022-2024")
    print(f"{'='*70}")
    print(f"  {'':<28} {'Optimized':>12} {'Meta-Ctrl':>12} {'B&H':>12}")
    print(f"  {'Cumulative Return':<28} {strat_cum:>+12.2%} {mc_cum:>+12.2%} {bh_cum:>+12.2%}")
    print(f"  {'Ann. Sharpe':<28} {strat_sh:>+12.3f} {mc_sh:>+12.3f} {bh_sh:>+12.3f}")
    print(f"  {'Max Drawdown':<28} {strat_dd:>12.2%} {mc_dd:>12.2%} {bh_dd:>12.2%}")
    print(f"  {'Mean Position':<28} {positions.mean():>12.2%} {mc_positions.mean():>12.0%} {'100%':>12}")

    strat_calmar = strat_cum / abs(strat_dd) if strat_dd != 0 else 0
    mc_calmar = mc_cum / abs(mc_dd) if mc_dd != 0 else 0
    bh_calmar = bh_cum / abs(bh_dd) if bh_dd != 0 else 0
    print(f"  {'Calmar Ratio':<28} {strat_calmar:>12.3f} {mc_calmar:>12.3f} {bh_calmar:>12.3f}")

    print(f"\n  Optimized weights: { {c: f'{w:.2f}' for c,w in enumerate(opt_weights)} }")
    print(f"  Meta-ctrl weights: { {i: f'{w:.2f}' for i,w in enumerate(mc_weights)} }")

    # Per-regime performance
    print(f"\n  Regime distribution & performance:")
    for c in range(n_regimes):
        mask = test_labels == c
        if mask.sum() > 0:
            bh_regime = np.prod(1 + rets_test[mask]) - 1
            strat_regime = np.prod(1 + strat_rets[mask]) - 1
            print(f"    #{c}: n={mask.sum():>3}, weight={opt_weights[c]:.2f}, "
                  f"B&H={bh_regime:>+7.2%}, Strat={strat_regime:>+7.2%}")

    # ── VIX vs Regime Regime ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VIX vs REGIME — Does regime predict volatility?")
    print(f"{'='*60}")
    for c in range(n_regimes):
        mask = test_labels == c
        if mask.sum() > 0:
            vix_c = test_df['VIX'].values[lookback:][mask]
            print(f"  Regime #{c}: VIX mean={vix_c.mean():.1f}, std={vix_c.std():.1f}, "
                  f"min={vix_c.min():.1f}, max={vix_c.max():.1f}")


if __name__ == "__main__":
    main()
