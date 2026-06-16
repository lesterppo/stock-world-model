"""
Stock World Model — Controller (Actor-Critic)

Phase 2: The trading agent that learns to optimize risk-adjusted returns
entirely within the M-Dynamics latent imagination (dream).

Architecture:
  - Shared backbone: MLP over latent state z_t
  - Actor head: Gaussian policy → position size ∈ [-1, 1]
  - Critic head: scalar value V(z_t) for advantage estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TradingController(nn.Module):
    """
    Actor-Critic controller for latent imagination RL.

    Input:  z_t [B, latent_dim] — encoded market state from V-Encoder
    Output: action distribution (μ, σ) + value estimate V(z_t)

    Action space: continuous position ∈ [-1, 1]
      -1 = maximum short, 0 = flat, 1 = maximum long
      The tanh-squashed Gaussian ensures actions stay bounded.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        action_std_init: float = 0.5,
        min_std: float = 0.05,
    ):
        super().__init__()
        self.min_std = min_std

        # Shared backbone: state → features
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: features → action mean + log_std
        self.actor_mean = nn.Linear(hidden_dim, 1)  # position size mean
        self.actor_log_std = nn.Parameter(
            torch.full((1,), action_std_init).log()
        )  # learnable global std

        # Critic head: features → scalar value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)
        # Actor mean: small initial outputs near zero
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, z_t: torch.Tensor):
        """
        Args:
            z_t: [B, latent_dim] — current latent market state
        Returns:
            action_mean:  [B, 1] — raw action mean before tanh
            action_std:   [B, 1] — action standard deviation
            value:        [B, 1] — state value estimate
        """
        features = self.backbone(z_t)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().clamp(min=self.min_std).expand_as(action_mean)
        value = self.critic(features)
        return action_mean, action_std, value

    def get_action(
        self,
        z_t: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            z_t: [B, latent_dim]
            deterministic: if True, return mean action (no sampling)
        Returns:
            action:       [B, 1] — position size in [-1, 1]
            log_prob:     [B, 1] — log probability of the action
            value:        [B, 1] — state value
        """
        action_mean, action_std, value = self.forward(z_t)

        if deterministic:
            action = torch.tanh(action_mean)
            # Log prob under deterministic action is not well-defined;
            # return zero so it doesn't affect gradient.
            log_prob = torch.zeros_like(action)
        else:
            dist = Normal(action_mean, action_std)
            raw_action = dist.rsample()  # reparameterized sample
            action = torch.tanh(raw_action)

            # Log probability with tanh correction (SAC-style)
            log_prob = dist.log_prob(raw_action)
            # tanh correction: log(1 - tanh(x)^2 + ε)
            log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_action(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of a given action.
        Used during PPO update with stored actions.

        Args:
            z_t:    [B, latent_dim]
            action: [B, 1] — previously sampled action in [-1, 1]
        Returns:
            log_prob:  [B, 1]
            entropy:   [B, 1]
            value:     [B, 1]
        """
        action_mean, action_std, value = self.forward(z_t)

        # Inverse tanh to get raw action
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        # Entropy correction for tanh is tiny — skip for stability

        return log_prob, entropy, value

    @property
    def entropy(self) -> torch.Tensor:
        """Current entropy of the policy distribution (for logging)."""
        return self.actor_log_std.exp().clamp(min=self.min_std).log() + 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi)))
