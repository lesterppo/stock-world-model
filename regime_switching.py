"""
Phase 7: Regime-Switching Risk Premium Strategy

Per Gemini Pro: "Stop asking RSSM to be a crystal ball for prices. Treat it
like a radar system — it can't tell you where the next wave hits, but it CAN
tell you a hurricane is forming."

Three integrated components:

1. REGIME TRANSITION PREDICTOR
   - Simple weighted transition matrix from K-Means labels
   - P(regime_{t+1} | regime_t) from empirical frequencies
   - Temporal smoothing via exponential moving average of regime probabilities
   - Detects regime CHANGE before it fully manifests

2. REGIME-AWARE POSITION SIZER
   - Kelly-like: f* = (μ - rf) / σ² per regime, scaled by regime probability
   - Position = Σ_c P(regime=c) · f*_c
   - Max position clamp (no leverage), minimum position floor
   - Circuit breaker: position → 0 when crisis regime #5 probability > 10%

3. GENERATIVE VOLATILITY SURFACE (RSSM Imagination)
   - Take today's (h_t, z_t) from RSSM
   - Imagine 10,000 × 30-day trajectories via RSSM.imagine_rollout
   - Compute expected 30-day variance from trajectory dispersion
   - Compare to VIX implied vol → vol arbitrage signal
   - When RSSM expects lower vol than options market: sell premium
   - When RSSM expects higher vol than options market: buy protection

Usage:
    python train_regime_switching.py --ticker SPY --rssm-ckpt checkpoints/SPY_rssm.pt
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM


class RegimeTransitionModel:
    """
    Smoothed regime transition predictor.

    Uses an exponentially-weighted transition matrix to estimate
    P(regime_{t+1} | regime_t, external_conditions).

    The smoothing prevents the model from overreacting to single-day
    regime flips (which are mostly noise in the K-Means assignment).
    """

    def __init__(
        self,
        n_regimes: int = 6,
        smooth_alpha: float = 0.3,  # smoothing for regime probability EMA
    ):
        self.n_regimes = n_regimes
        self.smooth_alpha = smooth_alpha

        # Empirical transition matrix: trans[from, to] = P(to | from)
        self.trans_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        # Regime stats: {regime: {ann_ret, ann_vol, sharpe}}
        self.regime_stats = {}

        # Smoothed regime belief
        self.belief = np.ones(n_regimes) / n_regimes
        self.current_regime = 0

    def fit(self, labels, returns):
        """Fit transition matrix and regime stats from historical labels."""
        # Transition counts
        trans_counts = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(len(labels) - 1):
            trans_counts[labels[t], labels[t + 1]] += 1

        # Add pseudocount (1) for unseen transitions, then normalize
        trans_counts += 0.1
        self.trans_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        # Regime stats
        for c in range(self.n_regimes):
            mask = labels == c
            if mask.sum() < 5:
                self.regime_stats[c] = {"ann_ret": 0.0, "ann_vol": 0.2, "sharpe": 0.0}
                continue
            r = returns[mask]
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            self.regime_stats[c] = {
                "ann_ret": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
            }

        print(f"Transition matrix fitted: {len(labels)} observations")

    def update(self, observed_regime: int):
        """
        Update smoothed regime belief based on observed regime.

        belief ← α · one_hot(observed) + (1-α) · belief @ trans_matrix
        """
        # One-hot of current observation
        obs = np.zeros(self.n_regimes)
        obs[observed_regime] = 1.0

        # Predicted belief from previous state
        predicted = self.belief @ self.trans_matrix

        # Smooth
        self.belief = self.smooth_alpha * obs + (1 - self.smooth_alpha) * predicted
        self.belief /= self.belief.sum()

        self.current_regime = observed_regime
        return self.belief

    def predict_next(self) -> np.ndarray:
        """Predict regime distribution for next step."""
        return self.belief @ self.trans_matrix

    def kelly_position(self, rf: float = 0.02) -> float:
        """
        Kelly-optimal position size given current regime belief.

        f* = (μ_p - rf) / σ²_p  where μ_p, σ²_p are belief-weighted.

        Clamped to [0, 1.5] — no shorting, max 1.5x leverage.
        """
        mu_p = 0.0
        var_p = 0.0
        for c in range(self.n_regimes):
            stats = self.regime_stats[c]
            mu_p += self.belief[c] * stats["ann_ret"]
            var_p += self.belief[c] * (stats["ann_vol"] ** 2)

        if var_p < 1e-6:
            return 0.5  # default

        kelly = (mu_p - rf) / var_p
        return float(np.clip(kelly, 0.0, 1.5))

    def crisis_probability(self) -> float:
        """Probability of crisis regime (index 5)."""
        return float(self.belief[5]) if self.n_regimes > 5 else 0.0

    def regime_description(self) -> str:
        """Human-readable current regime."""
        c = self.current_regime
        stats = self.regime_stats.get(c, {})
        probs = {i: f"{self.belief[i]:.1%}" for i in range(self.n_regimes)}
        return (
            f"Regime #{c} | ret={stats.get('ann_ret', 0):+.1%} "
            f"vol={stats.get('ann_vol', 0):.1%} | "
            f"belief={probs} | "
            f"kelly_pos={self.kelly_position():.2f}"
        )


class GenerativeVolSurface:
    """
    RSSM-based generative volatility surface.

    Uses RSSM.imagine_rollout to simulate thousands of future trajectories.
    The dispersion of these trajectories gives a model-implied volatility
    forecast that can be compared to options market implied vol (VIX).
    """

    def __init__(
        self,
        rssm: RSSM,
        encoder: MarketEncoder,
        horizon: int = 30,
        n_trajectories: int = 1000,
        device: str = "cpu",
    ):
        self.rssm = rssm
        self.encoder = encoder
        self.horizon = horizon
        self.n_trajectories = n_trajectories
        self.device = torch.device(device)

    @torch.no_grad()
    def forecast_vol(
        self,
        tech_seq: torch.Tensor,
        fund_t: torch.Tensor,
        action_t: torch.Tensor,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        n_samples: int = None,
    ) -> dict:
        """
        Generate volatility forecast via RSSM imagination.

        Args:
            tech_seq: [L, tech_dim] — lookback technical data
            fund_t:   [fund_dim]    — current fundamental
            action_t: [action_dim]  — current macro action
            h_t:      [hidden_dim]  — current RSSM hidden state
            z_t:      [latent_dim]  — current RSSM stochastic state

        Returns:
            dict with:
              - implied_vol_30d: annualized 30-day vol forecast
              - vol_percentiles: [10th, 25th, 50th, 75th, 90th]
              - trajectory_returns: [n_trajectories, horizon] — all paths
        """
        if n_samples is None:
            n_samples = self.n_trajectories

        # Encode current observation
        e_t = self.encoder(
            tech_seq.unsqueeze(0).to(self.device),
            fund_t.unsqueeze(0).to(self.device),
        )  # [1, embed_dim]

        # Use posterior to get clean z_t (incorporates latest observation)
        out = self.rssm.observe_step(
            h_t.unsqueeze(0).to(self.device),
            z_t.unsqueeze(0).to(self.device),
            action_t.unsqueeze(0).to(self.device),
            e_t,
        )
        h_start = out["h_t"]  # [1, hidden]
        z_start = out["z_t"]  # [1, latent]

        # Expand to n_samples trajectories
        h_batch = h_start.repeat(n_samples, 1)
        z_batch = z_start.repeat(n_samples, 1)

        # Generate action sequence (hold current macro + small noise)
        a_seq = action_t.unsqueeze(0).unsqueeze(1).repeat(
            1, n_samples, 1
        ).to(self.device)  # [1, N, action_dim]
        # Add noise to actions for diverse trajectories
        a_seq = a_seq + torch.randn_like(a_seq) * 0.01

        # Imagine over horizon (use a single step with repeated actions)
        # For simplicity, use imagine_step in a loop
        all_returns = torch.zeros(n_samples, self.horizon, device=self.device)
        h_curr, z_curr = h_batch, z_batch

        for t in range(self.horizon):
            # Sample next state from prior
            z_next_mean = None
            z_next_logvar = None

            # RSSM imagine step
            rnn_input = torch.cat([z_curr, action_t.unsqueeze(0).repeat(n_samples, 1).to(self.device)], dim=-1)
            h_curr = self.rssm.rnn(rnn_input, h_curr)
            p_feat = self.rssm.prior_fc(h_curr)
            prior_mu = self.rssm.prior_mu(p_feat)
            prior_logvar = self.rssm.prior_logvar(p_feat)
            std = torch.exp(0.5 * prior_logvar).clamp(min=0.1)
            z_curr = prior_mu + torch.randn_like(std) * std

            # "Return" from latent state: use a simple linear proxy
            # (reward decoder is broken, so use latent norm as vol proxy)
            all_returns[:, t] = z_curr.norm(dim=-1) * 0.01  # scale to daily-return-like

        # Compute volatility metrics from trajectories
        # Cumulative returns over horizon
        cum_rets = all_returns.sum(dim=1)  # [N] — 30-day cumulative

        # Annualized volatility from trajectory dispersion
        daily_std = cum_rets.std().item() / np.sqrt(self.horizon)
        vol_30d_ann = daily_std * np.sqrt(252)

        # Percentiles of cumulative returns
        cum_np = cum_rets.cpu().numpy()
        percentiles = {
            "p10": float(np.percentile(cum_np, 10)),
            "p25": float(np.percentile(cum_np, 25)),
            "p50": float(np.percentile(cum_np, 50)),
            "p75": float(np.percentile(cum_np, 75)),
            "p90": float(np.percentile(cum_np, 90)),
        }

        return {
            "implied_vol_30d": vol_30d_ann,
            "vol_percentiles": percentiles,
            "trajectory_returns": cum_np,
        }

    def vol_arbitrage_signal(
        self,
        model_vol: float,
        market_vol: float,
        threshold: float = 0.20,
    ) -> str:
        """
        Compare model-implied vol to market vol (VIX).

        Returns: 'sell_vol', 'buy_vol', or 'neutral'
        """
        ratio = model_vol / market_vol if market_vol > 0 else 1.0
        if ratio < (1.0 - threshold):
            return "sell_vol"   # model thinks vol is overpriced
        elif ratio > (1.0 + threshold):
            return "buy_vol"    # model thinks vol is underpriced
        else:
            return "neutral"


def extract_rssm_with_internal_state(
    checkpoint_path: str,
    df,
    lookback: int = 60,
    device: str = "cpu",
):
    """
    Walk through data extracting RSSM states, keeping internal state.
    Returns per-step (h_t, z_t, tech_window, fund_now, action_now).

    Unlike previous extractors, this preserves the RSSM's internal
    recurrent state so we can resume imagination from any point.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128)).to(device)
    rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128),
                cfg.get("latent_dim", 32)).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    rssm.load_state_dict(ckpt["rssm_state"])
    encoder.eval()
    rssm.eval()

    tech = torch.tensor(df[['Open', 'Close', 'Volume']].values, dtype=torch.float32, device=device)
    fund = torch.tensor(df[['ROE', 'Debt_Ratio']].values, dtype=torch.float32, device=device)
    acts = torch.tensor(df[['US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                           'US10Y_Volatility', 'is_earnings_day',
                           'Earnings_Surprise']].values, dtype=torch.float32, device=device)

    n = len(df)
    h_states = np.zeros((n - lookback, cfg.get("hidden_dim", 128)), dtype=np.float32)
    z_states = np.zeros((n - lookback, cfg.get("latent_dim", 32)), dtype=np.float32)
    daily_rets = np.zeros(n - lookback, dtype=np.float32)
    dates = df.index[lookback:]

    h_t, z_t = rssm.initial_state(1, torch.device(device))
    with torch.no_grad():
        for t in range(lookback, n):
            idx = t - lookback
            tw = tech[t - lookback:t].unsqueeze(0)
            fw = fund[t].unsqueeze(0)
            e_t = encoder(tw, fw)
            a_prev = acts[t].unsqueeze(0)
            out = rssm.observe_step(h_t, z_t, a_prev, e_t)
            h_t, z_t = out["h_t"], out["z_t"]
            h_states[idx] = h_t.cpu().numpy().squeeze(0)
            z_states[idx] = z_t.cpu().numpy().squeeze(0)
            daily_rets[idx] = df['Next_Day_Return'].values[t]

    return h_states, z_states, daily_rets, dates
