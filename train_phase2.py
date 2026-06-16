#!/usr/bin/env python3
"""
Stock World Model — Phase 2: Latent Imagination PPO Training

Trains a trading controller entirely within the M-Dynamics dream environment.
The controller learns to output position sizes ∈ [-1, 1] that maximize
risk-adjusted returns over K-step imagined trajectories.

Requires a pretrained Phase 1 checkpoint (V-Encoder + M-Dynamics + Reward Decoder).

Usage:
    # First train Phase 1:
    python train_phase1.py --epochs 50

    # Then train Phase 2:
    python train_phase2.py --checkpoint checkpoints/phase1_final.pt --epochs 100

    # Or train both together (Phase 1 first, then Phase 2):
    python train_phase1.py --epochs 30 && python train_phase2.py --checkpoint checkpoints/phase1_final.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MacroStockEncoder, StockDynamicsModel, RewardDecoder
from controller import TradingController
from dream_env import DreamEnv, DreamRollout
from losses import RiskAdjustedLoss
from data import (
    build_mock_fused_df,
    Phase2Dataset,
    TECH_FEATURE_COLS,
    FUND_FEATURE_COLS,
    MACRO_ACTION_COLS,
    MICRO_ACTION_COLS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "latent_dim": 64,
    "lookback_window": 60,
    "future_horizon": 20,
    "batch_size": 64,
    "ppo_epochs": 10,
    "env_steps": 500,          # total environment steps per PPO epoch
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "log_interval": 5,
    "risk_free_rate": 0.0,
    "mdd_lambda": 0.0,         # Phase 2 focuses on return; Phase 3 adds MDD
}


# ══════════════════════════════════════════════════════════════════════════════
# PPO Update
# ══════════════════════════════════════════════════════════════════════════════


def ppo_update(
    controller: TradingController,
    optimizer: torch.optim.Optimizer,
    rollout: DreamRollout,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    n_epochs: int = 4,
    batch_size: int = 64,
) -> dict[str, float]:
    """
    Run PPO update epochs over a collected rollout.

    Returns average metrics across all update epochs.
    """
    # Flatten rollout data: [K, B, ...] → [K*B, ...]
    states = rollout.states_t.reshape(-1, rollout.states_t.shape[-1])
    actions = rollout.actions_t.reshape(-1, 1)
    old_log_probs = rollout.log_probs_t.reshape(-1, 1)
    advantages = rollout.advantages.reshape(-1, 1)
    returns = rollout.returns.reshape(-1, 1)

    # Normalize advantages (stabilizes training)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = states.shape[0]
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        # Shuffle
        indices = torch.randperm(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = indices[start:end]

            s = states[idx]
            a = actions[idx]
            log_p_old = old_log_probs[idx]
            adv = advantages[idx]
            ret = returns[idx]

            # Evaluate current policy on stored actions
            log_p_new, entropy, value = controller.evaluate_action(s, a)

            # PPO clipped surrogate objective
            ratio = (log_p_new - log_p_old).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped, like PPO paper)
            value_pred = value
            value_loss = nn.functional.mse_loss(value_pred, ret)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Combined loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading (Phase 2 — trajectory slices)
# ══════════════════════════════════════════════════════════════════════════════


def load_data(lookback: int, horizon: int, n_days: int = 500, seed: int = 42):
    """Build mock data and Phase 2 trajectory dataset."""
    df = build_mock_fused_df(n_days=n_days, seed=seed)
    ds = Phase2Dataset(df, lookback_window=lookback, future_horizon=horizon)
    return DataLoader(ds, batch_size=DEFAULTS["batch_size"], shuffle=True, drop_last=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Latent Imagination PPO Training"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to Phase 1 checkpoint (V-Encoder + M-Dynamics + Reward Decoder)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULTS["ppo_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--latent-dim", type=int, default=DEFAULTS["latent_dim"])
    parser.add_argument("--lookback", type=int, default=DEFAULTS["lookback_window"])
    parser.add_argument("--horizon", type=int, default=DEFAULTS["future_horizon"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--lam", type=float, default=DEFAULTS["lam"])
    parser.add_argument("--clip-epsilon", type=float, default=DEFAULTS["clip_epsilon"])
    parser.add_argument("--value-coef", type=float, default=DEFAULTS["value_coef"])
    parser.add_argument("--entropy-coef", type=float, default=DEFAULTS["entropy_coef"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Load Phase 1 models ──────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    print(f"Loaded Phase 1 checkpoint: epoch {ckpt.get('epoch', '?')}")

    # Infer feature dimensions from dataset
    loader = load_data(args.lookback, args.horizon, n_days=500, seed=args.seed)
    sample = next(iter(loader))
    tech_dim = sample[0].shape[-1]   # V-features total
    a_dim = sample[1].shape[-1]      # action features total
    macro_dim = len(MACRO_ACTION_COLS)
    micro_dim = len(MICRO_ACTION_COLS)
    fund_dim = len(FUND_FEATURE_COLS)
    tech_seq_dim = len(TECH_FEATURE_COLS)

    print(f"Dims: tech_seq={tech_seq_dim}, fund={fund_dim}, macro={macro_dim}, micro={micro_dim}")

    # V-Encoder
    v_encoder = MacroStockEncoder(
        tech_dim=tech_seq_dim, fund_dim=fund_dim, latent_dim=args.latent_dim
    ).to(device)
    # M-Dynamics
    m_dynamics = StockDynamicsModel(
        latent_dim=args.latent_dim, macro_dim=macro_dim, micro_dim=micro_dim
    ).to(device)
    # Reward Decoder
    reward_decoder = RewardDecoder(latent_dim=args.latent_dim).to(device)

    # Load weights
    v_encoder.load_state_dict(ckpt["v_encoder_state"])
    m_dynamics.load_state_dict(ckpt["m_dynamics_state"])
    reward_decoder.load_state_dict(ckpt["reward_decoder_state"])
    print("Phase 1 weights loaded successfully.")

    # Freeze world model (we're training the controller only)
    for p in v_encoder.parameters():
        p.requires_grad = False
    for p in m_dynamics.parameters():
        p.requires_grad = False
    for p in reward_decoder.parameters():
        p.requires_grad = False
    v_encoder.eval()
    m_dynamics.eval()
    reward_decoder.eval()

    # ── Controller ───────────────────────────────────────────────────────────
    controller = TradingController(
        latent_dim=args.latent_dim,
        hidden_dim=128,
        action_std_init=0.5,
        min_std=0.05,
    ).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    print(f"Controller params: {sum(p.numel() for p in controller.parameters()):,}")

    # ── Dream Environment ────────────────────────────────────────────────────
    dream_env = DreamEnv(
        m_dynamics=m_dynamics,
        reward_decoder=reward_decoder,
        horizon=args.horizon,
        gamma=args.gamma,
        lam=args.lam,
        device=device,
    )

    # ── Training Loop ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Latent Imagination PPO — {args.epochs} epochs")
    print(f"  Horizon: {args.horizon}, Gamma: {args.gamma}, Lambda: {args.lam}")
    print(f"  Clip ε: {args.clip_epsilon}, Entropy: {args.entropy_coef}")
    print(f"{'='*60}\n")

    best_reward = -float("inf")

    for epoch in range(args.epochs):
        t_start = time.time()

        # Collect one rollout per batch in the dataloader
        epoch_rewards = []
        epoch_returns = []
        epoch_sharpes = []

        controller.train()
        for batch_idx, (x_history, x_actions, y_trajectory) in enumerate(loader):
            x_history = x_history.to(device)     # [B, L, v_feats]
            x_actions = x_actions.to(device)     # [B, K, a_feats]
            B = x_history.shape[0]

            # Split V-encoder input: tech (first columns) + fund (last columns)
            tech_seq = x_history[:, :, :tech_seq_dim]  # [B, L, tech_dim]
            fund_vec = x_history[:, -1, tech_seq_dim:]  # [B, fund_dim] — last day's fundamentals

            # Split actions: macro + micro
            macro_seq = x_actions[:, :, :macro_dim].permute(1, 0, 2)  # [K, B, macro_dim]
            micro_seq = x_actions[:, :, macro_dim:].permute(1, 0, 2) # [K, B, micro_dim]

            # Encode starting state z_0
            with torch.no_grad():
                _, _, z_0 = v_encoder(tech_seq, fund_vec)

            # Run dream rollout
            with torch.no_grad():
                rollout = dream_env.rollout(controller, z_0, macro_seq, micro_seq)

            # PPO update
            ppo_metrics = ppo_update(
                controller, optimizer, rollout,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=DEFAULTS["max_grad_norm"],
                n_epochs=4,
                batch_size=args.batch_size,
            )

            # Collect metrics
            epoch_rewards.append(rollout.rewards_t.mean().item())
            epoch_returns.append(rollout.returns.mean().item())

            # Compute simulated Sharpe on the rollout
            # rollout.rewards_t: [K, B, 1]
            mean_ret = rollout.rewards_t.mean(dim=0)  # [B, 1]
            std_ret = rollout.rewards_t.std(dim=0) + 1e-6
            sharpe = (mean_ret / std_ret).mean().item()
            epoch_sharpes.append(sharpe)

        # ── Epoch summary ────────────────────────────────────────────────────
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_return = sum(epoch_returns) / len(epoch_returns)
        avg_sharpe = sum(epoch_sharpes) / len(epoch_sharpes)
        elapsed = time.time() - t_start

        # Compute action stats to detect policy collapse
        with torch.no_grad():
            sample_z = z_0[:32]  # reuse last batch
            action, _, _ = controller.get_action(sample_z)
            action_mean = action.mean().item()
            action_std = action.std().item()

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Avg Reward: {avg_reward:+.4f} | "
            f"Avg Return: {avg_return:+.4f} | "
            f"Sim Sharpe: {avg_sharpe:+.3f} | "
            f"Policy: {ppo_metrics['policy_loss']:.4f} | "
            f"Value: {ppo_metrics['value_loss']:.4f} | "
            f"Entropy: {ppo_metrics['entropy']:.3f} | "
            f"Act μ: {action_mean:+.3f} σ: {action_std:.3f} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        best_reward = max(best_reward, avg_reward)
        checkpoint_dir = Path(DEFAULTS["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if avg_reward >= best_reward:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "controller_state": controller.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "avg_reward": avg_reward,
                    "avg_sharpe": avg_sharpe,
                    "config": vars(args),
                },
                checkpoint_dir / "phase2_best.pt",
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "controller_state": controller.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "avg_reward": avg_reward,
                },
                checkpoint_dir / f"phase2_epoch_{epoch + 1:03d}.pt",
            )

    # ── Final ────────────────────────────────────────────────────────────────
    # Run a deterministic evaluation rollout
    print(f"\n{'='*60}")
    print("Final Evaluation (deterministic policy)")
    print(f"{'='*60}")

    controller.eval()
    with torch.no_grad():
        eval_rewards = []
        eval_sharpes = []
        for batch_idx, (x_history, x_actions, _) in enumerate(loader):
            if batch_idx >= 5:  # sample 5 batches
                break
            x_history = x_history.to(device)
            x_actions = x_actions.to(device)
            B = x_history.shape[0]

            tech_seq = x_history[:, :, :tech_seq_dim]
            fund_vec = x_history[:, -1, tech_seq_dim:]

            macro_seq = x_actions[:, :, :macro_dim].permute(1, 0, 2)
            micro_seq = x_actions[:, :, macro_dim:].permute(1, 0, 2)

            _, _, z_t = v_encoder(tech_seq, fund_vec)

            traj_rewards = []
            for t in range(args.horizon):
                action, _, _ = controller.get_action(z_t, deterministic=True)
                pred_ret = reward_decoder(z_t)
                reward = action * pred_ret
                traj_rewards.append(reward)
                z_t, _, _ = m_dynamics(z_t, macro_seq[t], micro_seq[t])

            traj = torch.stack(traj_rewards).squeeze(-1)  # [K, B]
            mean_r = traj.mean(dim=0)
            std_r = traj.std(dim=0) + 1e-6
            sharpe = (mean_r / std_r).mean().item()
            eval_sharpes.append(sharpe)
            eval_rewards.append(traj.mean().item())

        print(f"Eval Reward: {sum(eval_rewards)/len(eval_rewards):+.4f}")
        print(f"Eval Sharpe: {sum(eval_sharpes)/len(eval_sharpes):+.3f}")

    torch.save(
        {
            "epoch": args.epochs,
            "controller_state": controller.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_reward": avg_reward,
        },
        checkpoint_dir / "phase2_final.pt",
    )
    print(f"\nTraining complete. Checkpoints in: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
