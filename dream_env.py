"""
Stock World Model — Dream Environment (Latent Imagination)

Wraps M-Dynamics + Reward Decoder into a gym-like environment that runs
entirely in latent space. The controller interacts with this environment
during Phase 2 PPO training — no real market data needed for rollouts.

Imagination loop:
  z_0 = V-Encoder(history)
  for t in 0..K-1:
    action_t = controller(z_t)          # position size ∈ [-1, 1]
    r_t = action_t * reward_decoder(z_t) # predicted next-day return × position
    z_{t+1} = M-Dynamics(z_t, macro_t, micro_t)
"""

import torch
import torch.nn as nn
from typing import Optional

from model import StockDynamicsModel, RewardDecoder


class DreamRollout:
    """
    A single imagination trajectory (batch of parallel rollouts).

    Stores all tensors needed for PPO: states, actions, log_probs,
    rewards, values, dones, and advantages.
    """

    def __init__(self, horizon: int, batch_size: int, latent_dim: int, device: torch.device):
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = device

        # Buffers
        self.states: list[torch.Tensor] = []        # z_t at each step
        self.actions: list[torch.Tensor] = []        # position sizes
        self.log_probs: list[torch.Tensor] = []      # log π(action|state)
        self.rewards: list[torch.Tensor] = []        # immediate rewards
        self.values: list[torch.Tensor] = []         # V(z_t)
        self.dones: list[torch.Tensor] = []          # terminal flags
        self.returns: Optional[torch.Tensor] = None   # computed after rollout
        self.advantages: Optional[torch.Tensor] = None

    def add_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        if done is None:
            done = torch.zeros(self.batch_size, 1, device=self.device)
        self.dones.append(done)

    def stack(self):
        """Convert lists to stacked tensors [horizon, batch, ...]"""
        self.states_t = torch.stack(self.states)       # [K, B, latent_dim]
        self.actions_t = torch.stack(self.actions)     # [K, B, 1]
        self.log_probs_t = torch.stack(self.log_probs) # [K, B, 1]
        self.rewards_t = torch.stack(self.rewards)     # [K, B, 1]
        self.values_t = torch.stack(self.values)       # [K, B, 1]
        self.dones_t = torch.stack(self.dones)         # [K, B, 1]

    def compute_gae(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        next_value: torch.Tensor = None,
    ):
        """
        Generalized Advantage Estimation (GAE).
        Modifies self.advantages and self.returns in-place.
        """
        self.stack()

        K, B, _ = self.rewards_t.shape

        if next_value is None:
            next_value = torch.zeros(B, 1, device=self.device)

        advantages = torch.zeros(B, 1, device=self.device)
        returns = next_value
        advantage_list = []
        return_list = []

        for t in reversed(range(K)):
            # TD error: r_t + γ * V(z_{t+1}) * (1 - done) - V(z_t)
            next_val = returns if t == K - 1 else self.values_t[t + 1]
            delta = (
                self.rewards_t[t]
                + gamma * next_val * (1.0 - self.dones_t[t])
                - self.values_t[t]
            )
            advantages = delta + gamma * lam * (1.0 - self.dones_t[t]) * advantages
            returns = self.rewards_t[t] + gamma * returns * (1.0 - self.dones_t[t])

            advantage_list.insert(0, advantages.clone())
            return_list.insert(0, returns.clone())

        self.advantages = torch.stack(advantage_list)  # [K, B, 1]
        self.returns = torch.stack(return_list)          # [K, B, 1]

        return self.advantages, self.returns


class DreamEnv:
    """
    Latent imagination environment.

    Given a starting state z_0 and a sequence of macro/micro actions,
    runs K-step rollouts using M-Dynamics for transitions and
    Reward Decoder for reward signals.
    """

    def __init__(
        self,
        m_dynamics: StockDynamicsModel,
        reward_decoder: RewardDecoder,
        horizon: int = 20,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.m_dynamics = m_dynamics
        self.reward_decoder = reward_decoder
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.device = device

    @torch.no_grad()
    def rollout(
        self,
        controller: nn.Module,
        z_0: torch.Tensor,
        macro_actions: torch.Tensor,
        micro_actions: torch.Tensor,
    ) -> DreamRollout:
        """
        Run one imagination trajectory.

        Args:
            controller:    TradingController (in eval mode for rollout)
            z_0:           [B, latent_dim] — starting state from V-Encoder
            macro_actions: [K, B, macro_dim] — external macro environment sequence
            micro_actions: [K, B, micro_dim] — external micro shock sequence
        Returns:
            DreamRollout with collected trajectories.
        """
        B = z_0.shape[0]
        K = self.horizon
        rollout = DreamRollout(K, B, z_0.shape[-1], self.device)

        z_t = z_0
        for t in range(K):
            # Controller decides position size
            action, log_prob, value = controller.get_action(z_t)

            # Reward: predicted next-day return × position size
            # The reward decoder predicts r_{t+1} from z_t
            pred_return = self.reward_decoder(z_t)  # [B, 1]
            reward = action * pred_return

            # Store
            rollout.add_step(z_t, action, log_prob, reward, value)

            # Transition: M-Dynamics predicts z_{t+1}
            macro_t = macro_actions[t]  # [B, macro_dim]
            micro_t = micro_actions[t]  # [B, micro_dim]
            z_t, _, _ = self.m_dynamics(z_t, macro_t, micro_t)

        # Compute value of final state (bootstrap)
        _, _, final_value = controller.get_action(z_t)
        # The final state's value is used for GAE bootstrapping
        # But we never actually took an action there, so the "next_value" for
        # the last real step is V(z_K)
        rollout.compute_gae(
            gamma=self.gamma,
            lam=self.lam,
            next_value=final_value,
        )

        # Also collect the final state's value as next_value
        rollout.final_value = final_value

        return rollout
