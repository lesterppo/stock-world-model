#!/usr/bin/env python3
"""
Phase 8b: Production Backtest — K-Means + All Production Guards

Per Gemini Pro review, all fixes applied:
  1. ✅ PIT-aligned fundamentals (45-day delay, fixed in data pipeline)
  2. ✅ Transaction costs (5bp/trade + 1bp slippage)
  3. ✅ K-Means hard clustering (what worked in Phase 7)
  4. ✅ OOD distance monitor (Euclidean to nearest centroid)
  5. ✅ Transition velocity cap
  6. ✅ Coordinate ascent optimization (what produced Sharpe 1.7-1.9)

NOT using GMM soft clustering — it destroyed regime differentiation.

Usage:
    python backtest_production.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM
from allocator import ProductionRegimeAllocator


def _max_drawdown(returns):
    cum = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def sharpe(returns):
    return returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)


def extract_states(df, checkpoint_path, lookback=60, device="cpu"):
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
    h = np.zeros((n - lookback, 128), dtype=np.float32)
    rets = df['Next_Day_Return'].values[lookback:]

    h_t, z_t = rssm.initial_state(1, torch.device(device))
    with torch.no_grad():
        for t in range(lookback, n):
            tw = tech[t-lookback:t].unsqueeze(0).to(device)
            fw = fund[t].unsqueeze(0).to(device)
            e_t = encoder(tw, fw)
            a_prev = acts[t].unsqueeze(0).to(device)
            out = rssm.observe_step(h_t, z_t, a_prev, e_t)
            h_t, z_t = out["h_t"], out["z_t"]
            h[t-lookback] = h_t.cpu().numpy().squeeze(0)
    return h, rets, df.index[lookback:]


def main():
    # ── Load data (with PIT fix) ────────────────────────────────────────────
    df = pd.read_csv("data/SPY_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]

    # ── Extract states ──────────────────────────────────────────────────────
    print("Extracting RSSM states (PIT-aligned fundamentals)...")
    h_train, rets_train, _ = extract_states(train_df, "checkpoints/SPY_rssm.pt")
    h_test, rets_test, dates_test = extract_states(test_df, "checkpoints/SPY_rssm.pt")
    print(f"  Train: {len(h_train)}, Test: {len(h_test)}")

    # ── Stability test across seeds ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STABILITY TEST — K-Means + Production Guards across seeds")
    print(f"{'='*70}")
    print(f"  {'Seed':<6} {'Weights':<40} {'Sharpe':>8} {'CumRet':>10} {'MaxDD':>10} {'OOD%':>7}")

    results = []
    for seed in [0, 1, 17, 42, 99, 123, 999]:
        allocator = ProductionRegimeAllocator(
            n_regimes=6,
            max_position=1.5,
            velocity_cap=0.15,
            ood_percentile=99.0,
            tc_bps=5.0,
            slippage_bps=1.0,
        )
        # Override K-Means seed
        allocator.kmeans = None  # force fresh fit
        allocator.scaler = None

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        allocator.scaler = StandardScaler()
        h_scaled = allocator.scaler.fit_transform(h_train)
        allocator.kmeans = KMeans(n_clusters=6, random_state=seed, n_init=10)
        labels = allocator.kmeans.fit_predict(h_scaled)
        allocator.centroids = allocator.kmeans.cluster_centers_

        # Fit stats and weights
        for c in range(6):
            mask = labels == c
            if mask.sum() < 5:
                allocator.regime_stats[c] = {"ann_ret": 0.0, "ann_vol": 0.2, "sharpe": 0.0}
                continue
            r = rets_train[mask]
            allocator.regime_stats[c] = {
                "ann_ret": float(r.mean() * 252),
                "ann_vol": float(r.std() * np.sqrt(252)),
                "sharpe": float(r.mean() / (r.std() + 1e-8) * np.sqrt(252)),
            }

        # Coordinate ascent (same as Phase 7)
        allocator.regime_weights = np.ones(6) * 0.5
        for _ in range(20):
            improved = False
            for c in range(6):
                mask = labels == c
                if mask.sum() < 5: continue
                best_w = allocator.regime_weights[c]; best_s = -float("inf")
                for w in [0.0, 0.25, 0.50, 0.75, 1.00]:
                    tw = allocator.regime_weights.copy(); tw[c] = w
                    sr = np.array([tw[l] * r for l, r in zip(labels, rets_train)])
                    score = sr.mean() / (sr.std() + 1e-8) * np.sqrt(252)
                    if score > best_s: best_s = score; best_w = w
                if best_w != allocator.regime_weights[c]:
                    allocator.regime_weights[c] = best_w; improved = True
            if not improved: break

        # OOD threshold
        dists = allocator._centroid_distance(h_scaled)
        allocator.ood_threshold = float(np.percentile(dists, 99.0))

        # Predict on test
        positions, test_labels, diag = allocator.predict_positions_batch(h_test)
        net_rets = allocator.compute_returns(positions, rets_test)

        strat_sh = sharpe(net_rets)
        strat_cum = np.prod(1 + net_rets) - 1
        strat_dd = _max_drawdown(net_rets)
        ood_pct = (diag["ood_distances"] > allocator.ood_threshold).mean() * 100

        ws = "[" + ",".join(f"{w:.0%}" for w in allocator.regime_weights) + "]"
        results.append({
            "seed": seed, "sharpe": strat_sh, "cum": strat_cum, "dd": strat_dd,
            "ood_pct": ood_pct, "weights": allocator.regime_weights.copy(),
            "mean_pos": positions.mean(),
        })
        print(f"  {seed:>6} {ws:<40} {strat_sh:>+8.4f} {strat_cum:>+10.2%} {strat_dd:>10.2%} {ood_pct:>6.1f}%")

    # ── Summary ─────────────────────────────────────────────────────────────
    sharpes = [r["sharpe"] for r in results]
    cums = [r["cum"] for r in results]
    dds = [r["dd"] for r in results]
    positions_mean = [r["mean_pos"] for r in results]

    bh_sh = sharpe(rets_test)
    bh_cum = np.prod(1 + rets_test) - 1
    bh_dd = _max_drawdown(rets_test)

    print(f"  {'─'*70}")
    print(f"  {'MEAN':<6} {'':<40} {np.mean(sharpes):>+8.4f} {np.mean(cums):>+10.2%} {np.mean(dds):>10.2%}")
    print(f"  {'STD':<6} {'':<40} {np.std(sharpes):>8.4f} {np.std(cums):>10.2%} {np.std(dds):>10.2%}")
    print(f"  {'B&H':<6} {'[100%,100%,100%,100%,100%,100%]':<40} {bh_sh:>+8.4f} {bh_cum:>+10.2%} {bh_dd:>10.2%}")

    sh_improve = np.mean(sharpes) - bh_sh
    dd_improve = bh_dd - np.mean(dds)  # positive = DDs improved (less negative)

    print(f"\n{'='*70}")
    print("PRODUCTION VERDICT (with TC=6bp, PIT fix, OOD guard)")
    print(f"{'='*70}")
    print(f"  Sharpe: {np.mean(sharpes):.4f} ± {np.std(sharpes):.4f} vs B&H {bh_sh:.4f}")
    print(f"  CumRet: {np.mean(cums):.2%} ± {np.std(cums):.2%} vs B&H {bh_cum:.2%}")
    print(f"  MaxDD:  {np.mean(dds):.2%} ± {np.std(dds):.2%} vs B&H {bh_dd:.2%}")
    print(f"  Mean position: {np.mean(positions_mean):.1%}")

    if sh_improve > 0.1:
        print(f"\n  ✓✓ Sharpe improvement {sh_improve:+.3f} — STRONG signal survives all fixes")
    elif sh_improve > 0.02:
        print(f"\n  ✓ Modest Sharpe improvement {sh_improve:+.3f}")
    else:
        print(f"\n  ✗ No significant Sharpe improvement ({sh_improve:+.3f})")

    if dd_improve > 0.10:  # 10 percentage points better DD
        print(f"  ✓✓ Drawdown reduced by {dd_improve:.1%} — major risk improvement")
    elif dd_improve > 0.02:
        print(f"  ✓ Drawdown reduced by {dd_improve:.1%}")
    else:
        print(f"  ~ Drawdown improvement marginal ({dd_improve:.1%})")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    np.savez("checkpoints/production_v8b.npz",
             sharpes=sharpes, cums=cums, dds=dds,
             positions_mean=positions_mean, bh_sharpe=bh_sh, bh_cum=bh_cum, bh_dd=bh_dd)


if __name__ == "__main__":
    main()
