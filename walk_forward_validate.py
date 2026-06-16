#!/usr/bin/env python3
"""
Phase 11: Walk-Forward Validation + Ablation Study

Per Gemini Pro's final prescription:
  1. Rolling walk-forward (not static 2022-2024)
  2. Ablation: RSSM vs simple momentum rotation
  3. Regime labeling via macro correlates

Usage:
    python walk_forward_validate.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM


def sharpe(r): return r.mean() / (r.std() + 1e-8) * np.sqrt(252)
def _dd(r):
    c = np.cumprod(1 + r); p = np.maximum.accumulate(c)
    return float(np.min((c - p) / p))
def calmar(r):
    c = np.prod(1 + r) - 1; d = _dd(r)
    return c / abs(d) if d != 0 else 0


def extract_states_for_df(df, checkpoint, tickers, lookback=60):
    """Extract PCA-reduced joint RSSM states for multi-asset data."""
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt.get("config", {})
    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128))
    rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128), cfg.get("latent_dim", 32))
    encoder.load_state_dict(ckpt["encoder_state"]); rssm.load_state_dict(ckpt["rssm_state"])
    encoder.eval(); rssm.eval()

    n = len(df); n_out = n - lookback
    n_assets = len(tickers)
    all_h = np.zeros((n_out, 128 * n_assets), dtype=np.float32)
    all_rets = np.zeros((n_out, n_assets), dtype=np.float32)

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

    return all_h, all_rets


def optimize_weights(labels, rets_matrix, n_regimes, n_assets):
    """Coordinate ascent for per-regime weights."""
    weights = np.ones((n_regimes, n_assets)) / n_assets
    for _ in range(30):
        improved = False
        for c in range(n_regimes):
            mask = labels == c
            if mask.sum() < 5: continue
            best_w = weights[c].copy(); best_calmar = -float("inf")
            for a in range(n_assets):
                for w_val in [0.0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0]:
                    tw = weights[c].copy(); tw[a] = w_val
                    tw = tw / tw.sum() if tw.sum() > 0 else np.ones(n_assets) / n_assets
                    score = calmar(rets_matrix[mask] @ tw)
                    if score > best_calmar: best_calmar = score; best_w = tw.copy()
            if not np.allclose(best_w, weights[c]): weights[c] = best_w; improved = True
        if not improved: break
    return weights


def momentum_weights(rets_matrix, lookback_mom=63):
    """Simple momentum rotation: equal-weight top 2 assets by trailing return."""
    n_assets = rets_matrix.shape[1]
    weights = np.zeros((len(rets_matrix), n_assets))
    for t in range(lookback_mom, len(rets_matrix)):
        mom = rets_matrix[t - lookback_mom:t].sum(axis=0)  # trailing return
        top2 = np.argsort(mom)[-2:]  # top 2 performers
        weights[t, top2] = 0.5  # equal weight
    # Pre-lookback: equal weight
    weights[:lookback_mom] = 1.0 / n_assets
    return weights


def main():
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)
    n_regimes = 6
    lookback = 60

    # Extract all states once
    print("Extracting RSSM states (full dataset)...")
    h_all, rets_all = extract_states_for_df(df, "checkpoints/SPY_rssm.pt", tickers, lookback)
    dates = df.index[lookback:]

    # PCA on first 5 years for initial transformation
    n_initial = 252 * 5  # 5 years training
    pca = PCA(n_components=20, random_state=42)
    pca.fit(h_all[:n_initial])
    h_pca = pca.transform(h_all)

    # ── Walk-forward validation ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("WALK-FORWARD VALIDATION (rolling 5-year train → 1-year test)")
    print(f"{'='*80}")
    print(f"  {'Window':<20} {'RSSM Calmar':>12} {'Mom Calmar':>12} {'RSSM Sharpe':>12} {'Mom Sharpe':>12}")

    rssm_metrics = []
    mom_metrics = []

    for year in range(2016, 2025):
        # Find nearest trading days (holidays may not exist in index)
        train_end_str = f"{year - 1}-12-31"
        test_start_str = f"{year}-01-02"  # avoid Jan 1 holiday
        test_end_str = f"{year}-12-31"

        # Find closest available dates
        train_end = df.index.searchsorted(pd.Timestamp(train_end_str)) - 1
        test_start = df.index.searchsorted(pd.Timestamp(test_start_str))
        test_end = min(df.index.searchsorted(pd.Timestamp(test_end_str)), len(df) - 1)

        if train_end < 0: train_end = 0

        train_end_lb = train_end - lookback
        test_start_lb = test_start - lookback
        test_end_lb = test_end - lookback

        if test_end_lb - test_start_lb < 50:  # too few test days
            continue

        # Train
        h_train = h_pca[:train_end_lb]
        rets_train = rets_all[:train_end_lb]
        labels_train = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit_predict(
            StandardScaler().fit_transform(h_train)
        )
        regime_weights = optimize_weights(labels_train, rets_train, n_regimes, n_assets)

        # Test
        h_test = h_pca[test_start_lb:test_end_lb]
        rets_test = rets_all[test_start_lb:test_end_lb]
        scaler = StandardScaler().fit(h_train)
        test_labels = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit(
            StandardScaler().fit_transform(h_train)
        ).predict(scaler.transform(h_test))

        positions = np.array([regime_weights[l] for l in test_labels])
        rssm_port = (positions * rets_test).sum(axis=1)

        # Momentum baseline
        mom_w = momentum_weights(rets_test)
        mom_port = (mom_w * rets_test).sum(axis=1)[lookback:]  # skip warmup

        rssm_cal = calmar(rssm_port)
        rssm_sh = sharpe(rssm_port)
        mom_cal = calmar(mom_port) if len(mom_port) > 10 else 0
        mom_sh = sharpe(mom_port) if len(mom_port) > 10 else 0

        print(f"  {year} (n={len(rssm_port):>4})   {rssm_cal:>+12.3f} {mom_cal:>+12.3f} {rssm_sh:>+12.3f} {mom_sh:>+12.3f}")

        rssm_metrics.append({"year": year, "calmar": rssm_cal, "sharpe": rssm_sh})
        mom_metrics.append({"year": year, "calmar": mom_cal, "sharpe": mom_sh})

    # ── Summary ─────────────────────────────────────────────────────────────
    rssm_cals = [m["calmar"] for m in rssm_metrics]
    mom_cals = [m["calmar"] for m in mom_metrics]
    rssm_shs = [m["sharpe"] for m in rssm_metrics]
    mom_shs = [m["sharpe"] for m in mom_metrics]

    wins = sum(1 for rc, mc in zip(rssm_cals, mom_cals) if rc > mc)

    print(f"  {'─'*80}")
    print(f"  {'MEAN':<20} {np.mean(rssm_cals):>+12.3f} {np.mean(mom_cals):>+12.3f} "
          f"{np.mean(rssm_shs):>+12.3f} {np.mean(mom_shs):>+12.3f}")
    print(f"  RSSM wins {wins}/{len(rssm_cals)} years on Calmar")

    # ── Regime labeling (macro correlates) ─────────────────────────────────
    print(f"\n{'='*60}")
    print("REGIME LABELS (macro correlates on full dataset)")
    print(f"{'='*60}")

    h_full = h_pca
    labels_full = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit_predict(
        StandardScaler().fit_transform(h_full)
    )

    macro_cols = ["VIX", "Yield_Spread", "US10Y"]
    print(f"  {'Regime':<8} {'Days':>6} {'VIX':>8} {'Spread':>8} {'US10Y':>8} "
          f"{'SPY ret':>9} {'TLT ret':>9} {'GLD ret':>9} {'DBC ret':>9}")
    for c in range(n_regimes):
        mask = labels_full[:len(dates)] == c
        if mask.sum() < 10: continue
        vix = df['VIX'].values[lookback:][mask].mean()
        spread = df['Yield_Spread'].values[lookback:][mask].mean()
        us10y = df['US10Y'].values[lookback:][mask].mean()
        spy_r = rets_all[mask, 0].mean() * 252
        tlt_r = rets_all[mask, 1].mean() * 252
        gld_r = rets_all[mask, 2].mean() * 252
        dbc_r = rets_all[mask, 3].mean() * 252
        print(f"  #{c:<7} {mask.sum():>6} {vix:>8.1f} {spread:>8.2f} {us10y:>8.2f} "
              f"{spy_r:>+9.1%} {tlt_r:>+9.1%} {gld_r:>+9.1%} {dbc_r:>+9.1%}")


if __name__ == "__main__":
    main()
