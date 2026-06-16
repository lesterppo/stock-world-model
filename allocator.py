"""
Phase 8b: K-Means Regime Allocator with Production Guards

Keeps what worked (K-Means hard clustering + differentiated per-regime weights)
and adds what Gemini Pro prescribed:
  - Transaction costs (5bp/trade + 1bp slippage)
  - OOD distance monitor (Euclidean to nearest centroid)
  - Transition velocity cap
  - PIT-aligned fundamentals (fixed in data pipeline)

NOT using GMM — soft blending destroys the regime differentiation signal.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class ProductionRegimeAllocator:
    """
    K-Means regime allocator with production safeguards.

    Uses hard clustering (like Phase 7 that achieved Sharpe 1.7-1.9)
    but adds: transaction costs, OOD guard, velocity cap.
    """

    def __init__(
        self,
        n_regimes: int = 6,
        max_position: float = 1.5,
        velocity_cap: float = 0.15,
        ood_percentile: float = 99.0,
        tc_bps: float = 5.0,
        slippage_bps: float = 1.0,
    ):
        self.n_regimes = n_regimes
        self.max_position = max_position
        self.velocity_cap = velocity_cap
        self.ood_percentile = ood_percentile
        self.tc_bps = tc_bps
        self.slippage_bps = slippage_bps

        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.regime_weights: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        self.ood_threshold: Optional[float] = None
        self.regime_stats: dict = {}

    def fit(self, h_train: np.ndarray, returns_train: np.ndarray):
        """Fit K-Means and optimize per-regime weights via coordinate ascent."""
        self.scaler = StandardScaler()
        h_scaled = self.scaler.fit_transform(h_train)

        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(h_scaled)
        self.centroids = self.kmeans.cluster_centers_

        # Regime statistics
        for c in range(self.n_regimes):
            mask = labels == c
            if mask.sum() < 5:
                self.regime_stats[c] = {"ann_ret": 0.0, "ann_vol": 0.2, "sharpe": 0.0}
                continue
            r = returns_train[mask]
            self.regime_stats[c] = {
                "ann_ret": float(r.mean() * 252),
                "ann_vol": float(r.std() * np.sqrt(252)),
                "sharpe": float(r.mean() / (r.std() + 1e-8) * np.sqrt(252)),
            }

        # Coordinate ascent optimization (discrete grid — what worked)
        self.regime_weights = np.ones(self.n_regimes) * 0.5
        for _ in range(20):
            improved = False
            for c in range(self.n_regimes):
                mask = labels == c
                if mask.sum() < 5:
                    continue
                best_w = self.regime_weights[c]
                best_score = -float("inf")
                for w in [0.0, 0.25, 0.50, 0.75, 1.00]:
                    tw = self.regime_weights.copy()
                    tw[c] = w
                    strat_rets = np.array([tw[label] * ret for label, ret in zip(labels, returns_train)])
                    sharpe = strat_rets.mean() / (strat_rets.std() + 1e-8) * np.sqrt(252)
                    if sharpe > best_score:
                        best_score = sharpe
                        best_w = w
                if best_w != self.regime_weights[c]:
                    self.regime_weights[c] = best_w
                    improved = True
            if not improved:
                break

        # OOD threshold from training data
        distances = self._centroid_distance(h_scaled)
        self.ood_threshold = float(np.percentile(distances, self.ood_percentile))

        return self

    def _centroid_distance(self, h_scaled: np.ndarray) -> np.ndarray:
        """Minimum Euclidean distance to any centroid."""
        min_dists = np.full(h_scaled.shape[0], np.inf)
        for c in range(self.n_regimes):
            diff = h_scaled - self.centroids[c]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            min_dists = np.minimum(min_dists, dists)
        return min_dists

    def predict_position(
        self,
        h_t: np.ndarray,
        prev_position: float = 0.5,
    ) -> Tuple[float, dict]:
        """
        Production position sizing with all guards.

        1. K-Means hard assignment → regime label
        2. Per-regime optimized weight
        3. OOD guard: if distance > threshold, reduce position to 30%
        4. Velocity cap: max daily change
        """
        h_scaled = self.scaler.transform(h_t.reshape(1, -1))
        label = int(self.kmeans.predict(h_scaled)[0])
        ood_dist = float(self._centroid_distance(h_scaled)[0])

        # Target position from regime weight
        target = float(self.regime_weights[label])

        # OOD guard
        ood_active = ood_dist > self.ood_threshold
        if ood_active:
            target *= 0.3

        # Velocity cap
        max_change = self.velocity_cap
        if target > prev_position + max_change:
            target = prev_position + max_change
        elif target < prev_position - max_change:
            target = prev_position - max_change

        position = float(np.clip(target, 0.0, self.max_position))

        info = {
            "regime_label": label,
            "target_position": float(self.regime_weights[label]),
            "ood_distance": ood_dist,
            "ood_active": ood_active,
        }

        return position, info

    def predict_positions_batch(
        self,
        h_seq: np.ndarray,
        initial_position: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Walk-forward position prediction with diagnostics."""
        N = len(h_seq)
        positions = np.zeros(N)
        labels = np.zeros(N, dtype=int)
        ood_dists = np.zeros(N)
        prev_pos = initial_position

        for t in range(N):
            pos, info = self.predict_position(h_seq[t], prev_pos)
            positions[t] = pos
            labels[t] = info["regime_label"]
            ood_dists[t] = info["ood_distance"]
            prev_pos = pos

        diagnostics = {
            "labels": labels,
            "ood_distances": ood_dists,
            "ood_threshold": self.ood_threshold,
        }

        return positions, labels, diagnostics

    def compute_returns(
        self,
        positions: np.ndarray,
        raw_returns: np.ndarray,
    ) -> np.ndarray:
        """Net returns after transaction costs."""
        N = len(positions)
        net_rets = np.zeros(N)
        tc_rate = (self.tc_bps + self.slippage_bps) / 10000.0
        prev_pos = 0.5

        for t in range(N):
            gross_ret = positions[t] * raw_returns[t]
            turnover = abs(positions[t] - prev_pos)
            cost = tc_rate * turnover
            net_rets[t] = gross_ret - cost
            prev_pos = positions[t]

        return net_rets

    def summary(self) -> str:
        lines = [
            f"ProductionRegimeAllocator: {self.n_regimes} regimes (K-Means)",
            f"Max pos={self.max_position}, velocity={self.velocity_cap}, "
            f"OOD={self.ood_percentile}%, TC={self.tc_bps}bp+{self.slippage_bps}bp",
            f"OOD threshold: {self.ood_threshold:.2f}",
            "",
            f"{'Regime':<8} {'Weight':>8} {'Ann Ret':>9} {'Ann Vol':>9} {'Sharpe':>8}",
        ]
        for k in range(self.n_regimes):
            s = self.regime_stats[k]
            lines.append(
                f"  #{k:<7} {self.regime_weights[k]:>8.3f} "
                f"{s['ann_ret']:>+9.1%} {s['ann_vol']:>9.1%} {s['sharpe']:>+8.2f}"
            )
        return "\n".join(lines)
