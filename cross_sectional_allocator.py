#!/usr/bin/env python3
"""
Phase 9: Multi-Asset Cross-Sectional Regime Allocator

Per Gemini Pro: "Regime models shine in cross-sectional allocation, not binary
timing. When the macro environment shifts, capital doesn't evaporate — it rotates."

Architecture:
  1. Extract RSSM h_t for each ETF using SPY-trained RSSM
  2. Concatenate: [h_SPY, h_TLT, h_GLD] → 384-dim joint state
  3. K-Means clustering on joint state → multi-asset regimes
  4. Per-regime: optimize weights across {SPY, TLT, GLD} via coordinate ascent
  5. Benchmark: 60/40 SPY/TLT static portfolio

Metric: Calmar Ratio (CumReturn / |MaxDD|) — per Gemini Pro recommendation.

Usage:
    python cross_sectional_allocator.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM


def sharpe(r): return r.mean() / (r.std() + 1e-8) * np.sqrt(252)
def _dd(r):
    c = np.cumprod(1 + r); p = np.maximum.accumulate(c)
    return float(np.min((c - p) / p))
def calmar(r):
    c = np.prod(1 + r) - 1; d = _dd(r)
    return c / abs(d) if d != 0 else 0


def extract_multi_states(df, checkpoint_path, tickers, lookback=60):
    """Extract RSSM h_t for each ETF, return stacked features."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get("config", {})
    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128))
    rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128), cfg.get("latent_dim", 32))
    encoder.load_state_dict(ckpt["encoder_state"]); rssm.load_state_dict(ckpt["rssm_state"])
    encoder.eval(); rssm.eval()

    n = len(df)
    n_out = n - lookback
    all_h = np.zeros((n_out, 128 * len(tickers)), dtype=np.float32)

    # Returns for each ETF
    all_rets = np.zeros((n_out, len(tickers)), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        tech = torch.tensor(df[[f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_Volume']].values, dtype=torch.float32)
        fund = torch.tensor(df[[f'{ticker}_ROE', f'{ticker}_Debt_Ratio']].values, dtype=torch.float32)
        acts = np.zeros((n, 7), dtype=np.float32)
        for j, col in enumerate(['US10Y','Yield_Spread','VIX','VIX_1w_Change','US10Y_Volatility']):
            if col in df.columns:
                acts[:, j] = df[col].values.astype(np.float32)

        h_t, z_t = rssm.initial_state(1, torch.device('cpu'))
        with torch.no_grad():
            for t in range(lookback, n):
                idx = t - lookback
                tw = tech[t-lookback:t].unsqueeze(0)
                fw = fund[t].unsqueeze(0)
                e_t = encoder(tw, fw)
                a_prev = torch.tensor(acts[t]).unsqueeze(0)
                out = rssm.observe_step(h_t, z_t, a_prev, e_t)
                h_t, z_t = out["h_t"], out["z_t"]
                all_h[idx, i * 128:(i + 1) * 128] = h_t.cpu().numpy().squeeze(0)

        all_rets[:, i] = df[f'{ticker}_Next_Return'].values[lookback:]

    return all_h, all_rets, df.index[lookback:]


def optimize_multi_weights(labels, rets_matrix, n_regimes, n_assets):
    """
    Coordinate ascent: find optimal [w0, w1, w2] per regime.
    Weights sum to 1, each ∈ [0, 1].
    """
    weights = np.ones((n_regimes, n_assets)) / n_assets  # start equal

    for _ in range(30):
        improved = False
        for c in range(n_regimes):
            mask = labels == c
            if mask.sum() < 5:
                continue
            best_w = weights[c].copy()
            best_calmar = -float("inf")
            # Try adjusting each asset up/down
            for a in range(n_assets):
                for w_val in [0.0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0]:
                    tw = weights[c].copy()
                    tw[a] = w_val
                    # Re-normalize
                    if tw.sum() > 0:
                        tw = tw / tw.sum()
                    else:
                        tw = np.ones(n_assets) / n_assets

                    # Portfolio return for this regime
                    port_rets = (rets_matrix[mask] @ tw)
                    score = calmar(port_rets)
                    if score > best_calmar:
                        best_calmar = score
                        best_w = tw.copy()

            if not np.allclose(best_w, weights[c]):
                weights[c] = best_w
                improved = True
        if not improved:
            break

    return weights


def main():
    # ── Load multi-ETF data ─────────────────────────────────────────────────
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)

    # ── Extract joint states ────────────────────────────────────────────────
    print("Extracting multi-asset RSSM states...")
    h_train, rets_train, _ = extract_multi_states(train_df, "checkpoints/SPY_rssm.pt", tickers)
    h_test, rets_test, _ = extract_multi_states(test_df, "checkpoints/SPY_rssm.pt", tickers)
    print(f"  Train: {len(h_train)}, Test: {len(h_test)}, Dim: {h_train.shape[1]}")

    # ── PCA dimension reduction (Gemini Pro: 384→20 dims) ──────────────────
    from sklearn.decomposition import PCA
    pca_dim = min(20, h_train.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    h_train = pca.fit_transform(h_train)
    h_test = pca.transform(h_test)
    print(f"  PCA reduced: {h_train.shape[1]} dims (explained var: {pca.explained_variance_ratio_.sum():.1%})")

    # ── K-Means on reduced states ───────────────────────────────────────────
    n_regimes = 6
    scaler = StandardScaler()
    h_train_scaled = scaler.fit_transform(h_train)

    print(f"\nMulti-Asset Regime Allocator (K-Means, {n_regimes} regimes)")
    print(f"{'Seed':<6} {'Regime Weights (SPY/TLT/GLD)':<55} {'Calmar':>8} {'Sharpe':>8} {'CumRet':>10} {'MaxDD':>10}")

    results = []

    for seed in [0, 17, 42, 99, 123, 999]:
        kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(h_train_scaled)

        # Optimize per-regime multi-asset weights
        regime_weights = optimize_multi_weights(labels, rets_train, n_regimes, n_assets)

        # Apply to test
        h_test_scaled = scaler.transform(h_test)
        test_labels = kmeans.predict(h_test_scaled)

        positions = np.zeros((len(test_labels), n_assets))
        for t, label in enumerate(test_labels):
            positions[t] = regime_weights[label]

        port_rets = (positions * rets_test).sum(axis=1)

        cal = calmar(port_rets)
        sh = sharpe(port_rets)
        cum = np.prod(1 + port_rets) - 1
        dd = _dd(port_rets)

        # Format weights
        ws_str = "; ".join(
            f"#{c}:{','.join(f'{w:.0%}' for w in regime_weights[c])}"
            for c in range(n_regimes)
        )
        print(f"  {seed:>6} {ws_str:<55} {cal:>8.3f} {sh:>+8.3f} {cum:>+10.2%} {dd:>10.2%}")

        results.append({
            "calmar": cal, "sharpe": sh, "cum": cum, "dd": dd,
            "weights": regime_weights,
        })

    # ── Benchmarks ──────────────────────────────────────────────────────────
    # Risk parity
    vols_test = np.array([rets_test[:, i].std() * np.sqrt(252) for i in range(n_assets)])
    erc_w = (1.0 / vols_test) / np.sum(1.0 / vols_test)
    erc_rets = rets_test @ erc_w
    erc_cal = calmar(erc_rets)
    erc_sh = sharpe(erc_rets)
    erc_cum = np.prod(1 + erc_rets) - 1
    erc_dd = _dd(erc_rets)

    # 60/40
    bench_w = np.zeros(n_assets); bench_w[0] = 0.6; bench_w[1] = 0.4
    bench_rets = rets_test @ bench_w
    bench_cal = calmar(bench_rets)
    bench_sh = sharpe(bench_rets)
    bench_cum = np.prod(1 + bench_rets) - 1
    bench_dd = _dd(bench_rets)

    # Equal weight
    eq_rets = rets_test.mean(axis=1)
    eq_cal = calmar(eq_rets)

    # Individual assets
    spy_rets = rets_test[:, 0]

    print(f"  {'─'*85}")
    print(f"  {'MEAN':<6} {'':<55} {np.mean([r['calmar'] for r in results]):>8.3f} "
          f"{np.mean([r['sharpe'] for r in results]):>+8.3f} "
          f"{np.mean([r['cum'] for r in results]):>+10.2%} "
          f"{np.mean([r['dd'] for r in results]):>10.2%}")
    print(f"  RiskParity ({','.join(f'{w:.0%}' for w in erc_w)}): {erc_cal:>8.3f} {erc_sh:>+8.3f} {erc_cum:>+10.2%} {erc_dd:>10.2%}")
    print(f"  60/40    {'':<55} {bench_cal:>8.3f} {bench_sh:>+8.3f} {bench_cum:>+10.2%} {bench_dd:>10.2%}")
    print(f"  EqWt     {'':<55} {eq_cal:>8.3f}")
    print(f"  SPY      {'':<55} {calmar(spy_rets):>8.3f} {sharpe(spy_rets):>+8.3f} "
          f"{np.prod(1+spy_rets)-1:>+10.2%} {_dd(spy_rets):>10.2%}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MULTI-ASSET VERDICT (PCA 20-dim + DBC commodity)")
    print(f"{'='*60}")

    mean_calmar = np.mean([r["calmar"] for r in results])
    if mean_calmar > erc_cal * 1.2:
        print(f"  ✓ Calmar {mean_calmar:.3f} > RiskParity {erc_cal:.3f} — meaningful improvement")
    elif mean_calmar > erc_cal:
        print(f"  ~ Marginal Calmar edge: {mean_calmar:.3f} vs RiskParity {erc_cal:.3f}")
    else:
        print(f"  ✗ Calmar {mean_calmar:.3f} < RiskParity {erc_cal:.3f} — underperforms")

    mean_sh = np.mean([r["sharpe"] for r in results])
    if mean_sh > erc_sh + 0.1:
        print(f"  ✓ Sharpe {mean_sh:.3f} > RiskParity {erc_sh:.3f}")
    else:
        print(f"  ~ Sharpe {mean_sh:.3f} vs RiskParity {erc_sh:.3f}")


if __name__ == "__main__":
    main()
