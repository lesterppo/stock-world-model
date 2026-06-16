#!/usr/bin/env python3
"""
Phase 12: Ablation Study — RSSM vs Raw Macro Features

Per Gemini Pro: "Is the RSSM complexity justified? Perform a baseline comparison
using only macro features in your clusterer, bypassing the RSSM entirely."

Three competing approaches compared in walk-forward:
  1. RSSM latent states (20-dim PCA on h_t) — our model
  2. Raw macro features (VIX, yield spread, US10Y, momentum) — ablation
  3. Momentum rotation (trailing 63-day return) — simple baseline

If RSSM doesn't drastically outperform raw macro clustering, the neural
architecture is not justified.

Usage:
    python ablation_study.py
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


def extract_rssm_states(df, checkpoint, tickers, lookback=60):
    """Extract PCA-reduced RSSM states (same as walk_forward_validate.py)."""
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

    # PCA to 20 dims
    pca = PCA(n_components=20, random_state=42)
    return pca.fit_transform(all_h), all_rets, pca


def extract_macro_features(df, tickers, lookback=60):
    """
    Build raw macro feature matrix — NO RSSM.

    Features per timestep:
      - VIX level
      - VIX 1-week change
      - Yield spread (US10Y - US2Y)
      - US10Y level
      - SPY 21-day momentum
      - SPY 63-day momentum
    """
    n = len(df); n_out = n - lookback
    n_assets = len(tickers)

    # Build feature matrix: [N, 6] (same for all assets — macro is shared)
    features = np.zeros((n_out, 6), dtype=np.float32)
    features[:, 0] = df['VIX'].values[lookback:]
    features[:, 1] = df['VIX_1w_Change'].values[lookback:]
    features[:, 2] = df['Yield_Spread'].values[lookback:]
    features[:, 3] = df['US10Y'].values[lookback:]

    # SPY momentum (lagged, no lookahead)
    spy_close = df['SPY_Close'].values
    for t_idx in range(lookback, n):
        idx = t_idx - lookback
        if t_idx >= lookback + 21:
            features[idx, 4] = spy_close[t_idx - 1] / spy_close[t_idx - 22] - 1
        if t_idx >= lookback + 63:
            features[idx, 5] = spy_close[t_idx - 1] / spy_close[t_idx - 64] - 1

    # Returns matrix (same as RSSM version)
    all_rets = np.zeros((n_out, n_assets), dtype=np.float32)
    for i, ticker in enumerate(tickers):
        all_rets[:, i] = df[f'{ticker}_Next_Return'].values[lookback:]

    return features, all_rets


def optimize_weights(labels, rets_matrix, n_regimes, n_assets):
    """Coordinate ascent (same as before)."""
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
    """Simple momentum rotation baseline."""
    n_assets = rets_matrix.shape[1]
    weights = np.zeros((len(rets_matrix), n_assets))
    for t in range(lookback_mom, len(rets_matrix)):
        mom = rets_matrix[t - lookback_mom:t].sum(axis=0)
        top2 = np.argsort(mom)[-2:]
        weights[t, top2] = 0.5
    weights[:lookback_mom] = 1.0 / n_assets
    return weights


def main():
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)
    n_regimes = 6
    lookback = 60

    # ── Extract features for all three approaches ───────────────────────────
    print("Extracting features...")
    h_rssm, rets_rssm, pca = extract_rssm_states(df, "checkpoints/SPY_rssm.pt", tickers, lookback)
    h_macro, rets_macro = extract_macro_features(df, tickers, lookback)
    print(f"  RSSM states: {h_rssm.shape} (PCA 20-dim)")
    print(f"  Macro features: {h_macro.shape} (VIX, spread, momentum)")

    # ── Walk-forward ablation ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("ABLATION STUDY — RSSM vs Raw Macro vs Momentum (rolling 5yr train → 1yr test)")
    print(f"{'='*90}")
    print(f"  {'Year':<6} {'RSSM Calmar':>12} {'Macro Calmar':>12} {'Mom Calmar':>12} "
          f"{'RSSM Sharpe':>12} {'Macro Sharpe':>12} {'Mom Sharpe':>12}")

    results = {"rssm": [], "macro": [], "mom": []}

    for year in range(2016, 2025):
        train_end_str = f"{year - 1}-12-31"
        test_start_str = f"{year}-01-02"
        test_end_str = f"{year}-12-31"

        train_end = df.index.searchsorted(pd.Timestamp(train_end_str)) - 1
        test_start = df.index.searchsorted(pd.Timestamp(test_start_str))
        test_end = min(df.index.searchsorted(pd.Timestamp(test_end_str)), len(df) - 1)
        if train_end < 0: train_end = 0

        train_end_lb = train_end - lookback
        test_start_lb = test_start - lookback
        test_end_lb = test_end - lookback
        if test_end_lb - test_start_lb < 50: continue

        # ── RSSM approach ──────────────────────────────────────────────────
        h_tr = h_rssm[:train_end_lb]
        r_tr = rets_rssm[:train_end_lb]
        labels_tr = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit_predict(
            StandardScaler().fit_transform(h_tr)
        )
        rw_rssm = optimize_weights(labels_tr, r_tr, n_regimes, n_assets)

        h_te = h_rssm[test_start_lb:test_end_lb]
        r_te = rets_rssm[test_start_lb:test_end_lb]
        scaler = StandardScaler().fit(h_tr)
        test_l = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit(
            StandardScaler().fit_transform(h_tr)
        ).predict(scaler.transform(h_te))
        pos_rssm = np.array([rw_rssm[l] for l in test_l])
        port_rssm = (pos_rssm * r_te).sum(axis=1)

        # ── Raw Macro approach ─────────────────────────────────────────────
        h_tr_m = h_macro[:train_end_lb]
        r_tr_m = rets_macro[:train_end_lb]
        labels_tr_m = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit_predict(
            StandardScaler().fit_transform(h_tr_m)
        )
        rw_macro = optimize_weights(labels_tr_m, r_tr_m, n_regimes, n_assets)

        h_te_m = h_macro[test_start_lb:test_end_lb]
        r_te_m = rets_macro[test_start_lb:test_end_lb]
        scaler_m = StandardScaler().fit(h_tr_m)
        test_l_m = KMeans(n_clusters=n_regimes, random_state=42, n_init=10).fit(
            StandardScaler().fit_transform(h_tr_m)
        ).predict(scaler_m.transform(h_te_m))
        pos_macro = np.array([rw_macro[l] for l in test_l_m])
        port_macro = (pos_macro * r_te_m).sum(axis=1)

        # ── Momentum baseline ──────────────────────────────────────────────
        mom_w = momentum_weights(r_te)
        port_mom = (mom_w * r_te).sum(axis=1)[lookback:]

        rssm_c = calmar(port_rssm); rssm_s = sharpe(port_rssm)
        macro_c = calmar(port_macro); macro_s = sharpe(port_macro)
        mom_c = calmar(port_mom) if len(port_mom) > 10 else 0
        mom_s = sharpe(port_mom) if len(port_mom) > 10 else 0

        print(f"  {year:<6} {rssm_c:>+12.3f} {macro_c:>+12.3f} {mom_c:>+12.3f} "
              f"{rssm_s:>+12.3f} {macro_s:>+12.3f} {mom_s:>+12.3f}")

        results["rssm"].append({"year": year, "calmar": rssm_c, "sharpe": rssm_s})
        results["macro"].append({"year": year, "calmar": macro_c, "sharpe": macro_s})
        results["mom"].append({"year": year, "calmar": mom_c, "sharpe": mom_s})

    # ── Summary ─────────────────────────────────────────────────────────────
    rssm_cals = [m["calmar"] for m in results["rssm"]]
    macro_cals = [m["calmar"] for m in results["macro"]]
    mom_cals = [m["calmar"] for m in results["mom"]]
    rssm_shs = [m["sharpe"] for m in results["rssm"]]
    macro_shs = [m["sharpe"] for m in results["macro"]]
    mom_shs = [m["sharpe"] for m in results["mom"]]

    rssm_wins = sum(1 for rc, mc in zip(rssm_cals, macro_cals) if rc > mc)
    macro_wins = sum(1 for rc, mc in zip(rssm_cals, macro_cals) if mc > rc)
    rssm_vs_mom = sum(1 for rc, mc in zip(rssm_cals, mom_cals) if rc > mc)

    print(f"  {'─'*90}")
    print(f"  {'MEAN':<6} {np.mean(rssm_cals):>+12.3f} {np.mean(macro_cals):>+12.3f} "
          f"{np.mean(mom_cals):>+12.3f} {np.mean(rssm_shs):>+12.3f} "
          f"{np.mean(macro_shs):>+12.3f} {np.mean(mom_shs):>+12.3f}")
    print(f"  RSSM vs Macro: {rssm_wins}/{len(rssm_cals)} years")
    print(f"  RSSM vs Mom:   {rssm_vs_mom}/{len(rssm_cals)} years")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ABLATION VERDICT")
    print(f"{'='*60}")

    rssm_mean_cal = np.mean(rssm_cals)
    macro_mean_cal = np.mean(macro_cals)

    if rssm_mean_cal > macro_mean_cal * 1.3:
        print(f"  ✓ RSSM ({rssm_mean_cal:.3f}) >> Macro ({macro_mean_cal:.3f}) "
              f"— RSSM complexity STRONGLY justified ({rssm_mean_cal/macro_mean_cal:.2f}x)")
    elif rssm_mean_cal > macro_mean_cal * 1.1:
        print(f"  ✓ RSSM ({rssm_mean_cal:.3f}) > Macro ({macro_mean_cal:.3f}) "
              f"— RSSM justified ({rssm_mean_cal/macro_mean_cal:.2f}x)")
    elif rssm_mean_cal > macro_mean_cal:
        print(f"  ~ RSSM marginally better ({rssm_mean_cal/macro_mean_cal:.2f}x) — "
              f"complexity not fully justified")
    else:
        print(f"  ✗ Macro ({macro_mean_cal:.3f}) ≥ RSSM ({rssm_mean_cal:.3f}) "
              f"— RSSM complexity NOT justified. Use raw macro features.")


if __name__ == "__main__":
    main()
