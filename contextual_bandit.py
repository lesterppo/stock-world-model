"""
Phase 6: Contextual Bandit Portfolio Allocator

Architecture (per Gemini Pro critique):
  - Single asset: SPY position size ∈ [0, 1] (cash → fully invested)
  - State: RSSM deterministic h_t (128-dim) + recent return context
  - Policy: neural network with dropout (anti-memorization)
  - Reward: Differential Sharpe Ratio (DSR) — online, maintains Markov property
  - Training: REINFORCE with value baseline on realized returns
  - Walk-forward CV: chronological train/val splits, no look-ahead

Key design decisions:
  - RSSM is purely external observer — controller actions do NOT feed back into RSSM
  - Training uses REALIZED returns, not predicted returns (reward decoder is broken)
  - Dropout on state input prevents the GRU's implicit clock from being memorized
  - Differential Sharpe avoids trailing-window state explosion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Optional, Tuple


class BanditPolicy(nn.Module):
    """
    Contextual bandit policy: h_t → position size ∈ [0, 1].

    Architecture:
      - State dropout (p=0.3) to fight GRU clock memorization
      - 3-layer MLP with residual connections
      - Output: sigmoid → position ∈ [0, 1]
    """

    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        action_noise: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_noise = action_noise

        # State dropout (applied BEFORE the network)
        self.state_dropout = nn.Dropout(dropout)

        # Shared backbone
        self.fc1 = nn.Linear(state_dim + 3, hidden_dim)  # h_t + [ret_1d, ret_5d, ret_20d]
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Action head: outputs mean position
        self.action_mean = nn.Linear(hidden_dim // 2, 1)
        self.action_log_std = nn.Parameter(torch.tensor(0.0))  # learnable

        # Value head: baseline for REINFORCE variance reduction
        self.value = nn.Linear(hidden_dim // 2, 1)

        # DSR state (not parameters, managed externally)
        self.register_buffer("dsr_A", torch.tensor(0.0))  # EMA of returns
        self.register_buffer("dsr_B", torch.tensor(0.0))  # EMA of squared returns
        self.register_buffer("dsr_decay", torch.tensor(0.99))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small initial action mean (start near 0.5 position)
        nn.init.constant_(self.action_mean.bias, -0.0)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)

    def forward(
        self,
        h_t: torch.Tensor,
        recent_rets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t:          [B, 128] — RSSM deterministic state
            recent_rets:  [B, 3]   — [ret_1d, ret_5d, ret_20d]
        Returns:
            action_mean:  [B, 1] — raw logit (before sigmoid)
            action_std:   [B, 1] — action noise
            value:        [B, 1] — state value baseline
        """
        # Dropout on state to prevent memorization
        h_drop = self.state_dropout(h_t)

        x = torch.cat([h_drop, recent_rets], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + F.linear(x, self.fc2.weight[:x.shape[-1]].T.detach())  # residual
        x = F.relu(self.fc3(x))

        action_mean = self.action_mean(x)  # logit space
        action_std = torch.exp(self.action_log_std).clamp(min=0.05).expand_as(action_mean)
        value = self.value(x)

        return action_mean, action_std, value

    @torch.no_grad()
    def get_action_deterministic(
        self,
        h_t: torch.Tensor,
        recent_rets: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic action: sigmoid(action_mean). Use for backtesting."""
        action_mean, _, _ = self.forward(h_t, recent_rets)
        return torch.sigmoid(action_mean)

    def get_action_stochastic(
        self,
        h_t: torch.Tensor,
        recent_rets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action for training (REINFORCE).

        Returns:
            action:   [B, 1] ∈ [0, 1] — position size
            log_prob: [B, 1] — log probability
            value:    [B, 1] — state value
        """
        action_mean, action_std, value = self.forward(h_t, recent_rets)

        # Sample in logit space, then sigmoid
        dist = Normal(action_mean, action_std)
        logit_sample = dist.rsample()
        action = torch.sigmoid(logit_sample)

        # Log prob with sigmoid transform correction
        log_prob = dist.log_prob(logit_sample)
        # Jacobian correction: d/d(logit) sigmoid(logit) = sigmoid(logit) * (1 - sigmoid(logit))
        log_prob -= torch.log(action * (1.0 - action) + 1e-8)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_action(
        self,
        h_t: torch.Tensor,
        recent_rets: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log prob of a previously sampled action.
        Used for multi-epoch PPO-style updates on stored trajectories.

        Args:
            action: [B, 1] in [0, 1]
        """
        action_mean, action_std, value = self.forward(h_t, recent_rets)

        # Inverse sigmoid
        logit = torch.log((action + 1e-8) / (1.0 - action + 1e-8))

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(logit)
        log_prob -= torch.log(action * (1.0 - action) + 1e-8)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value


class DifferentialSharpeRatio:
    """
    Differential Sharpe Ratio — online, maintains Markov property.

    D_t = (B_{t-1}·ΔA_t - A_{t-1}·ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}

    Where:
      A_t = η·A_{t-1} + (1-η)·R_t      (EMA of returns)
      B_t = η·B_{t-1} + (1-η)·R_t²     (EMA of squared returns)

    The DSR is the instantaneous contribution to the trailing Sharpe.
    Positive DSR means the current return IMPROVES the Sharpe;
    negative DSR means it worsens it.

    This avoids the hidden-state problem of trailing-window metrics.
    """

    def __init__(self, eta: float = 0.02):
        """
        Args:
            eta: decay rate. η = 0.02 → half-life ~35 days, η = 0.05 → half-life ~14 days
        """
        self.eta = eta
        self.reset()

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self.t = 0

    def update(self, R_t: float) -> float:
        """
        Compute DSR given the current return.

        Args:
            R_t: scalar return for this step
        Returns:
            D_t: differential Sharpe ratio
        """
        self.t += 1
        eta = self.eta

        # Exponential moving averages
        dA = (1.0 - eta) * (R_t - self.A)
        dB = (1.0 - eta) * (R_t ** 2 - self.B)

        denom = self.B - self.A ** 2
        if denom <= 1e-10:
            dsr = 0.0
        else:
            dsr = (self.B * dA - 0.5 * self.A * dB) / (denom ** 1.5)

        self.A += dA
        self.B += dB

        return dsr


def compute_reward(
    weight: torch.Tensor,
    ret_next: torch.Tensor,
    risk_aversion: float = 1.0,
    vol_est: float = 0.01,
    prev_weight: Optional[torch.Tensor] = None,
    turnover_cost: float = 0.001,
) -> torch.Tensor:
    """
    Compute instantaneous reward for bandit training.

    R_t = w_t · r_{t+1} - λ · w_t² · σ² - γ · |w_t - w_{t-1}|

    Args:
        weight:      [B, 1] ∈ [0, 1]
        ret_next:    [B]    next-day realized return
        risk_aversion: λ — risk penalty coefficient
        vol_est:     daily volatility estimate for risk penalty
        prev_weight: [B, 1] previous position (for turnover)
        turnover_cost: γ — cost per unit of turnover

    Returns:
        reward: [B, 1] scalar reward per sample
    """
    # Return component: positive weight × positive return = good
    ret_component = weight.squeeze(-1) * ret_next

    # Risk component: penalize large positions
    risk_component = risk_aversion * (weight.squeeze(-1) ** 2) * (vol_est ** 2)

    # Turnover component: penalize rapid changes
    turnover_component = torch.zeros_like(ret_component)
    if prev_weight is not None:
        turnover_component = turnover_cost * torch.abs(
            weight.squeeze(-1) - prev_weight.squeeze(-1)
        )

    reward = ret_component - risk_component - turnover_component
    return reward.unsqueeze(-1)


def reinforce_update(
    policy: BanditPolicy,
    h_seq: torch.Tensor,
    ret_ctx_seq: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict:
    """
    PPO-style update for the contextual bandit policy.

    Uses GAE for advantages and clipped surrogate objective.
    Multi-epoch updates with importance sampling correction.

    Args:
        h_seq:         [T, B, 128] — state sequence
        ret_ctx_seq:   [T, B, 3]   — return context
        actions:       [T, B, 1]   — stored actions
        log_probs_old: [T, B, 1]   — stored log probs
        rewards:       [T, B, 1]   — computed rewards
        values:        [T, B, 1]   — stored values
    Returns:
        metrics dict
    """
    T, B = rewards.shape[0], rewards.shape[1]
    device = rewards.device

    # Detach all inputs to prevent double-backward on multi-epoch updates
    h_seq = h_seq.detach()
    ret_ctx_seq = ret_ctx_seq.detach()
    actions = actions.detach()
    log_probs_old = log_probs_old.detach()
    rewards = rewards.detach()
    values = values.detach()

    # ── GAE Advantage Computation ──────────────────────────────────────────
    advantages = torch.zeros(B, 1, device=device)
    returns_list = []

    for t in reversed(range(T)):
        next_val = torch.zeros(B, 1, device=device) if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        advantages = delta + gamma * lam * advantages
        returns_list.insert(0, advantages.clone())

    advantages = torch.stack(returns_list)  # [T, B, 1]
    returns = advantages + values            # GAE returns = advantage + V

    # Flatten for batch processing
    h_flat = h_seq.reshape(-1, h_seq.shape[-1])
    ret_ctx_flat = ret_ctx_seq.reshape(-1, ret_ctx_seq.shape[-1])
    actions_flat = actions.reshape(-1, 1)
    log_probs_old_flat = log_probs_old.reshape(-1, 1)
    advantages_flat = advantages.reshape(-1, 1)
    returns_flat = returns.reshape(-1, 1)

    # Normalize advantages
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for _ in range(n_epochs):
        # Shuffle
        perm = torch.randperm(h_flat.shape[0], device=device)

        for i in range(0, h_flat.shape[0], B):
            idx = perm[i:i + B]
            if len(idx) < 2:
                continue

            h_batch = h_flat[idx]
            ctx_batch = ret_ctx_flat[idx]
            act_batch = actions_flat[idx]
            old_lp_batch = log_probs_old_flat[idx]
            adv_batch = advantages_flat[idx]
            ret_batch = returns_flat[idx]

            # Evaluate current policy on stored actions
            new_lp, entropy, new_val = policy.evaluate_action(h_batch, ctx_batch, act_batch)

            # PPO clipped surrogate
            ratio = torch.exp(new_lp - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_val, ret_batch)

            # Entropy bonus (encourage exploration)
            entropy_loss = -entropy_coef * entropy.mean()

            loss = policy_loss + 0.5 * value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

    n_updates = max(1, (h_flat.shape[0] // B) * n_epochs)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }


def extract_rssm_states_with_context(
    checkpoint_path: str,
    df,
    lookback: int = 60,
    device: str = "cpu",
):
    """
    Walk through historical data, extract RSSM h_t + return context.
    Unlike metactl.extract_states, this also captures 1d/5d/20d return context
    and leaves RSSM internal state tracking intact.

    Returns:
        h_states:   [N, 128] — RSSM deterministic state
        ret_ctx:    [N, 3]   — [ret_1d, ret_5d, ret_20d] lagged returns
        daily_rets: [N]      — next-day realized returns
        dates:      [N]      — dates
    """
    from model import MarketEncoder, RSSM

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128)).to(device)
    rssm = RSSM(cfg.get("embed_dim", 128), 7, cfg.get("hidden_dim", 128),
                cfg.get("latent_dim", 32)).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    rssm.load_state_dict(ckpt["rssm_state"])
    encoder.eval()
    rssm.eval()

    tech = torch.tensor(df[['Open', 'Close', 'Volume']].values, dtype=torch.float32)
    fund = torch.tensor(df[['ROE', 'Debt_Ratio']].values, dtype=torch.float32)
    acts = torch.tensor(df[['US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                           'US10Y_Volatility', 'is_earnings_day',
                           'Earnings_Surprise']].values, dtype=torch.float32)
    rets = df['Next_Day_Return'].values

    n = len(df)
    h_states = np.zeros((n - lookback, cfg.get("hidden_dim", 128)), dtype=np.float32)
    ret_ctx = np.zeros((n - lookback, 3), dtype=np.float32)
    daily_rets = np.zeros(n - lookback, dtype=np.float32)
    dates = df.index[lookback:]

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

            # Return context: lagged returns (no look-ahead)
            if idx >= 20:
                ret_ctx[idx, 0] = rets[t - 1] if t > 0 else 0.0
                ret_ctx[idx, 1] = np.mean(rets[t - 5:t]) if t >= 5 else 0.0
                ret_ctx[idx, 2] = np.mean(rets[t - 20:t]) if t >= 20 else 0.0
            elif idx >= 5:
                ret_ctx[idx, 0] = rets[t - 1] if t > 0 else 0.0
                ret_ctx[idx, 1] = np.mean(rets[t - 5:t]) if t >= 5 else 0.0
                ret_ctx[idx, 2] = 0.0
            else:
                ret_ctx[idx, 0] = rets[t - 1] if t > 0 else 0.0
                ret_ctx[idx, 1] = 0.0
                ret_ctx[idx, 2] = 0.0

            daily_rets[idx] = rets[t]

    return h_states, ret_ctx, daily_rets, dates
