#!/usr/bin/env python3
"""
Phase 13: Optimized Macro Regime Allocator (Production)

Per Gemini Pro + ablation study: RSSM complexity NOT justified.
Raw macro features (VIX, spread, US10Y, momentum) outperform RSSM latent states.

This is the PRODUCTION-READY implementation with all safeguards:
  - Transaction costs (5bp + 1bp slippage)
  - K-Means ensemble across seeds (stability)
  - Velocity cap on position changes
  - OOD guard (Euclidean distance to nearest centroid)
  - Risk parity benchmark (not naive 60/40)
  - Calmar-optimized per-regime weights

Features (6-dim, all point-in-time, no lookahead):
  VIX level, VIX 1w change, Yield Spread (10Y-2Y), US10Y level,
  SPY 21-day momentum, SPY 63-day momentum

Usage:
    python macro_allocator.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def sharpe(r): return r.mean() / (r.std() + 1e-8) * np.sqrt(252)
def _dd(r):
    c = np.cumprod(1 + r); p = np.maximum.accumulate(c)
    return float(np.min((c - p) / p))
def calmar(r):
    c = np.prod(1 + r) - 1; d = _dd(r)
    return c / abs(d) if d != 0 else 0


class MacroRegimeAllocator:
    """
    Production macro regime allocator.

    Features: VIX, yield spread, US10Y, momentum — NO neural networks.
    """

    def __init__(
        self,
        n_regimes: int = 6,
        max_position: float = 1.0,
        velocity_cap: float = 0.15,
        ood_percentile: float = 99.0,
        tc_bps: float = 5.0,
        slippage_bps: float = 1.0,
        n_ensemble: int = 5,
        soft_assign: bool = True,
        soft_temperature: float = 1.0,
    ):
        self.n_regimes = n_regimes
        self.max_position = max_position
        self.velocity_cap = velocity_cap
        self.ood_percentile = ood_percentile
        self.tc_bps = tc_bps
        self.slippage_bps = slippage_bps
        self.n_ensemble = n_ensemble
        self.soft_assign = soft_assign
        self.soft_temperature = soft_temperature

        self.scaler = StandardScaler()
        self.kmeans_models = []     # ensemble of K-Means
        self.regime_weights = []    # per-ensemble per-regime weights
        self.centroids_list = []    # per-ensemble centroids
        self.ood_thresholds = []    # per-ensemble OOD thresholds
        self.regime_stats = []      # per-ensemble regime statistics

    def fit(self, features, returns_matrix):
        """
        Fit ensemble of K-Means models + optimize per-regime weights.

        Args:
            features:       [N, F] — macro features (VIX, spread, etc.)
            returns_matrix: [N, A] — next-day returns for each asset
        """
        F_scaled = self.scaler.fit_transform(features)
        n_assets = returns_matrix.shape[1]

        for seed in range(self.n_ensemble):
            km = KMeans(n_clusters=self.n_regimes, random_state=seed, n_init=10)
            labels = km.fit_predict(F_scaled)

            # Optimize per-regime weights (coordinate ascent)
            weights = np.ones((self.n_regimes, n_assets)) / n_assets
            for _ in range(30):
                improved = False
                for c in range(self.n_regimes):
                    mask = labels == c
                    if mask.sum() < 5: continue
                    best_w = weights[c].copy(); best_calmar = -float("inf")
                    for a in range(n_assets):
                        for w_val in [0.0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0]:
                            tw = weights[c].copy(); tw[a] = w_val
                            tw = tw / tw.sum() if tw.sum() > 0 else np.ones(n_assets) / n_assets
                            score = calmar(returns_matrix[mask] @ tw)
                            if score > best_calmar: best_calmar = score; best_w = tw.copy()
                    if not np.allclose(best_w, weights[c]):
                        weights[c] = best_w; improved = True
                if not improved: break

            # OOD threshold
            centroids = km.cluster_centers_
            min_dists = np.full(len(F_scaled), np.inf)
            for c in range(self.n_regimes):
                d = np.sqrt(np.sum((F_scaled - centroids[c]) ** 2, axis=1))
                min_dists = np.minimum(min_dists, d)
            ood_thresh = float(np.percentile(min_dists, self.ood_percentile))

            self.kmeans_models.append(km)
            self.regime_weights.append(weights)
            self.centroids_list.append(centroids)
            self.ood_thresholds.append(ood_thresh)

            # Regime stats
            stats = {}
            for c in range(self.n_regimes):
                mask = labels == c
                if mask.sum() < 5: continue
                r = returns_matrix[mask]
                port_mean = (r @ weights[c]).mean() * 252
                port_vol = (r @ weights[c]).std() * np.sqrt(252)
                stats[c] = {"ann_ret": float(port_mean), "ann_vol": float(port_vol)}
            self.regime_stats.append(stats)

        return self

    def predict_positions(self, features_test, returns_test):
        """
        Walk-forward ensemble prediction with transaction costs.

        When soft_assign=True (default): blends regime weights by inverse
        distance to ALL centroids, not just the nearest. This prevents
        K-Means cluster collapse on non-stationary test data.

        Returns: positions [N, A], net_rets [N]
        """
        F_scaled = self.scaler.transform(features_test)
        n_assets = returns_test.shape[1]
        N = len(features_test)
        tc_rate = (self.tc_bps + self.slippage_bps) / 10000.0

        positions = np.zeros((N, n_assets))
        prev_pos = np.ones(n_assets) / n_assets  # start equal-weight

        for t in range(N):
            target_w = np.zeros(n_assets)
            n_active = 0

            for m_idx in range(self.n_ensemble):
                km = self.kmeans_models[m_idx]
                centroids = self.centroids_list[m_idx]
                ood_thresh = self.ood_thresholds[m_idx]

                if self.soft_assign:
                    # ── Soft distance-weighted blend across ALL regimes ──
                    dists = np.array([
                        np.sqrt(np.sum((F_scaled[t] - centroids[c]) ** 2))
                        for c in range(self.n_regimes)
                    ])
                    # Temperature-scaled softmax over NEGATIVE distances
                    # Closer centroid → higher weight
                    # Low temperature → harder assignment; high → more uniform
                    logits = -dists / (self.soft_temperature * np.std(dists) + 1e-8)
                    logits -= logits.max()  # numerical stability
                    blend_w = np.exp(logits)
                    blend_w /= blend_w.sum()

                    # OOD guard: if min distance > threshold, pull toward equal-weight
                    min_dist = dists.min()
                    if min_dist > ood_thresh:
                        ood_frac = min(1.0, (min_dist - ood_thresh) / (ood_thresh + 1e-8))
                        blend_w = blend_w * (1 - ood_frac * 0.7) + ood_frac * 0.7 * (1.0 / self.n_regimes)

                    # Blend regime weights
                    w = np.zeros(n_assets)
                    for c in range(self.n_regimes):
                        w += blend_w[c] * self.regime_weights[m_idx][c]
                else:
                    # ── Hard assignment (original behavior) ──
                    label = int(km.predict(F_scaled[t:t+1])[0])
                    min_dist = np.inf
                    for c in range(self.n_regimes):
                        d = np.sqrt(np.sum((F_scaled[t] - centroids[c]) ** 2))
                        min_dist = min(min_dist, d)
                    w = self.regime_weights[m_idx][label].copy()
                    if min_dist > ood_thresh:
                        w *= 0.3

                target_w += w
                n_active += 1

            if n_active > 0:
                target_w /= n_active

            # Velocity cap
            max_change = self.velocity_cap
            change = target_w - prev_pos
            if np.abs(change).max() > max_change:
                change = np.clip(change, -max_change, max_change)
                target_w = prev_pos + change

            # Normalize
            if target_w.sum() > 0:
                target_w /= target_w.sum()

            positions[t] = target_w
            prev_pos = target_w

        # Compute net returns with transaction costs
        net_rets = np.zeros(N)
        prev_pos = np.ones(n_assets) / n_assets
        for t in range(N):
            gross_ret = (positions[t] * returns_test[t]).sum()
            turnover = np.abs(positions[t] - prev_pos).sum()
            cost = tc_rate * turnover
            net_rets[t] = gross_ret - cost
            prev_pos = positions[t]

        return positions, net_rets


def extract_macro_features(df, lookback=60):
    """Build raw macro feature matrix (no RSSM)."""
    n = len(df)
    n_out = n - lookback

    features = np.zeros((n_out, 6), dtype=np.float32)
    features[:, 0] = df['VIX'].values[lookback:]
    features[:, 1] = df['VIX_1w_Change'].values[lookback:]
    features[:, 2] = df['Yield_Spread'].values[lookback:]
    features[:, 3] = df['US10Y'].values[lookback:]

    spy_close = df['SPY_Close'].values
    for t_idx in range(lookback, n):
        idx = t_idx - lookback
        if t_idx >= lookback + 21:
            features[idx, 4] = spy_close[t_idx - 1] / spy_close[t_idx - 22] - 1
        if t_idx >= lookback + 63:
            features[idx, 5] = spy_close[t_idx - 1] / spy_close[t_idx - 64] - 1

    return features


def main():
    df = pd.read_csv("data/multi_SPY_TLT_GLD_DBC_fused.csv", index_col=0, parse_dates=True)
    tickers = ["SPY", "TLT", "GLD", "DBC"]
    n_assets = len(tickers)
    lookback = 60

    # ── Extract features and returns ────────────────────────────────────────
    features = extract_macro_features(df, lookback)
    rets_all = np.zeros((len(features), n_assets), dtype=np.float32)
    for i, t in enumerate(tickers):
        rets_all[:, i] = df[f'{t}_Next_Return'].values[lookback:]
    dates = df.index[lookback:]
    print(f"Features: {features.shape}, Returns: {rets_all.shape}")

    # ── Walk-forward validation ─────────────────────────────────────────────
    n_initial = 252 * 5  # 5 years initial training
    n_regimes = 6
    tc_bps = 5.0

    print(f"\n{'='*85}")
    print(f"PRODUCTION MACRO ALLOCATOR — 9-Year Walk-Forward (TC={tc_bps}bp)")
    print(f"{'='*85}")
    print(f"  {'Year':<6} {'Calmar':>8} {'Sharpe':>8} {'CumRet':>10} {'MaxDD':>10} "
          f"{'T/O':>8} {'MeanPos':>10}")

    for year in range(2016, 2025):
        train_end = df.index.searchsorted(pd.Timestamp(f"{year-1}-12-31")) - 1
        test_start = df.index.searchsorted(pd.Timestamp(f"{year}-01-02"))
        test_end = min(df.index.searchsorted(pd.Timestamp(f"{year}-12-31")), len(df) - 1)
        if train_end < 0: train_end = 0

        train_end_lb = train_end - lookback
        test_start_lb = test_start - lookback
        test_end_lb = test_end - lookback
        if test_end_lb - test_start_lb < 50: continue

        # Fit
        alloc = MacroRegimeAllocator(
            n_regimes=n_regimes, tc_bps=tc_bps, slippage_bps=1.0,
            n_ensemble=3, velocity_cap=0.15,
        )
        alloc.fit(features[:train_end_lb], rets_all[:train_end_lb])

        # Predict
        positions, net_rets = alloc.predict_positions(
            features[test_start_lb:test_end_lb],
            rets_all[test_start_lb:test_end_lb],
        )

        cal = calmar(net_rets)
        sh = sharpe(net_rets)
        cum = np.prod(1 + net_rets) - 1
        dd = _dd(net_rets)
        turnover = np.abs(np.diff(positions, axis=0)).mean()
        mean_pos = positions.mean()

        print(f"  {year:<6} {cal:>+8.3f} {sh:>+8.3f} {cum:>+10.2%} {dd:>10.2%} "
              f"{turnover:>8.4f} {mean_pos:>10.1%}")

    # ── Benchmarks ──────────────────────────────────────────────────────────
    # Full backtest on 2022-2024 for detailed comparison
    train_end = df.index.searchsorted(pd.Timestamp("2021-12-31")) - 1
    test_start = df.index.searchsorted(pd.Timestamp("2022-01-02"))
    train_end_lb = train_end - lookback
    test_start_lb = test_start - lookback

    alloc = MacroRegimeAllocator(
        n_regimes=n_regimes, tc_bps=tc_bps, slippage_bps=1.0,
        n_ensemble=5, velocity_cap=0.15,
    )
    alloc.fit(features[:train_end_lb], rets_all[:train_end_lb])
    positions, net_rets = alloc.predict_positions(
        features[test_start_lb:], rets_all[test_start_lb:]
    )
    rets_test = rets_all[test_start_lb:]

    # Risk parity
    vols_test = np.array([rets_test[:, i].std() * np.sqrt(252) for i in range(n_assets)])
    erc_w = (1.0 / vols_test) / np.sum(1.0 / vols_test)
    erc_rets = (rets_test * erc_w).sum(axis=1)

    # 60/40
    bw = np.zeros(n_assets); bw[0] = 0.6; bw[1] = 0.4
    bench_rets = (rets_test * bw).sum(axis=1)

    # SPY only
    spy_rets = rets_test[:, 0]

    macro_c = calmar(net_rets); macro_s = sharpe(net_rets)
    erc_c = calmar(erc_rets); erc_s = sharpe(erc_rets)
    bench_c = calmar(bench_rets); bench_s = sharpe(bench_rets)
    spy_c = calmar(spy_rets); spy_s = sharpe(spy_rets)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON — SPY/TLT/GLD/DBC 2022-2024 (with TC=6bp)")
    print(f"{'='*60}")
    print(f"  {'Strategy':<20} {'Calmar':>8} {'Sharpe':>8} {'CumRet':>10} {'MaxDD':>10}")
    print(f"  {'Macro (production)':<20} {macro_c:>+8.3f} {macro_s:>+8.3f} "
          f"{np.prod(1+net_rets)-1:>+10.2%} {_dd(net_rets):>10.2%}")
    print(f"  {'Risk Parity':<20} {erc_c:>+8.3f} {erc_s:>+8.3f} "
          f"{np.prod(1+erc_rets)-1:>+10.2%} {_dd(erc_rets):>10.2%}")
    print(f"  {'60/40 SPY/TLT':<20} {bench_c:>+8.3f} {bench_s:>+8.3f} "
          f"{np.prod(1+bench_rets)-1:>+10.2%} {_dd(bench_rets):>10.2%}")
    print(f"  {'SPY only':<20} {spy_c:>+8.3f} {spy_s:>+8.3f} "
          f"{np.prod(1+spy_rets)-1:>+10.2%} {_dd(spy_rets):>10.2%}")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    np.savez("checkpoints/macro_final.npz",
             net_rets=net_rets, positions=positions,
             calmar=macro_c, sharpe=macro_s)


if __name__ == "__main__":
    main()
