#!/usr/bin/env python3
"""
Stock World Model — Phase 5: RSSM Training with KL Annealing

Replaces the old Variational GRU encoder + M-Dynamics with a unified
RSSM (Recurrent State Space Model, DreamerV2/V3 architecture).

Key fix: prevents posterior collapse (KL ≈ 0.0001 in Phase 1) by:
  1. Separating deterministic h_t from stochastic z_t
  2. KL annealing: gradually ramp KL weight from 0 → 1
  3. Free bits: minimum KL per dimension to force information through bottleneck

The trained RSSM can then be used for latent imagination (replacing
M-Dynamics in Phase 2) with healthy stochastic rollouts.

Usage:
    python train_phase5.py --epochs 50
    python train_phase5.py --epochs 100 --seq-len 30 --free-bits 1.0
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MarketEncoder, RSSM, RSSMRewardDecoder, ContrastiveHead
from losses import KLAnnealer, rssm_phase5_loss, contrastive_loss
from data import build_multi_regime_df

# ══════════════════════════════════════════════════════════════════════════════
# RSSM Sequence Dataset
# ══════════════════════════════════════════════════════════════════════════════


class RSSMSequenceDataset(Dataset):
    """
    Provides fixed-length temporal sequences for RSSM training.

    Each sample is a chunk of seq_len consecutive market days:
      - tech_seq: [seq_len, tech_dim] — technical features
      - fund_seq: [seq_len, fund_dim] — fundamental features
      - action_seq: [seq_len, action_dim] — macro + micro actions
      - reward_seq: [seq_len] — next-day returns

    The RSSM processes these step by step: at step t, it encodes the
    observation (tech_seq[t], fund_seq[t]) to e_t, uses action_seq[t]
    as the action, and predicts reward_seq[t].
    """
    def __init__(self, dataframe, seq_len: int = 20, lookback: int = 60):
        self.seq_len = seq_len
        self.lookback = lookback

        df = dataframe.sort_index().copy()
        df["Next_Day_Return"] = df["Close"].pct_change().shift(-1).fillna(0.0)

        # Feature columns
        self.tech_cols = ["Open", "Close", "Volume"]
        self.fund_cols = ["ROE", "Debt_Ratio"]
        self.macro_cols = ["US10Y", "Yield_Spread", "VIX", "VIX_1w_Change", "US10Y_Volatility"]
        self.micro_cols = ["is_earnings_day", "Earnings_Surprise"]

        # Pre-compute encoded observations using a rolling window
        self.tech_feats = df[self.tech_cols].values.astype(np.float32)
        self.fund_feats = df[self.fund_cols].values.astype(np.float32)
        self.action_feats = df[self.macro_cols + self.micro_cols].values.astype(np.float32)
        self.rewards = df["Next_Day_Return"].values.astype(np.float32)

        # For each step t, we need lookback days of tech history to encode e_t.
        # We pre-compute encoded observations by sliding the lookback window.
        n = len(df)
        self.valid_starts = list(range(0, n - lookback - seq_len))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        start = self.valid_starts[idx]

        # tech_seq[t] = tech features for days [start+t : start+t+lookback]
        # We'll encode these in the training loop.
        # For now, return raw features for the training loop to encode.
        tech_buf = []
        fund_buf = []
        for t in range(self.seq_len):
            day = start + t
            tech_buf.append(self.tech_feats[day : day + self.lookback])
            fund_buf.append(self.fund_feats[day + self.lookback - 1])

        tech_seq = np.stack(tech_buf, axis=0)     # [seq_len, lookback, tech_dim]
        fund_seq = np.stack(fund_buf, axis=0)      # [seq_len, fund_dim]
        action_seq = self.action_feats[start + self.lookback : start + self.lookback + self.seq_len]
        reward_seq = self.rewards[start + self.lookback : start + self.lookback + self.seq_len]

        return (
            torch.tensor(tech_seq, dtype=torch.float32),    # [S, L, tech_dim]
            torch.tensor(fund_seq, dtype=torch.float32),    # [S, fund_dim]
            torch.tensor(action_seq, dtype=torch.float32),  # [S, action_dim]
            torch.tensor(reward_seq, dtype=torch.float32),  # [S]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════


DEFAULTS = {
    "embed_dim": 128,
    "hidden_dim": 128,
    "latent_dim": 64,
    "seq_len": 20,
    "lookback": 60,
    "batch_size": 32,
    "epochs": 50,
    "lr": 3e-4,
    "reward_weight": 1.0,
    "anneal_steps": 5000,
    "free_bits": 0.5,
    "grad_clip": 1.0,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "log_interval": 10,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 5: RSSM Training with KL Annealing")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--lookback", type=int, default=DEFAULTS["lookback"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--embed-dim", type=int, default=DEFAULTS["embed_dim"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"])
    parser.add_argument("--latent-dim", type=int, default=DEFAULTS["latent_dim"])
    parser.add_argument("--reward-weight", type=float, default=DEFAULTS["reward_weight"])
    parser.add_argument("--anneal-steps", type=int, default=DEFAULTS["anneal_steps"])
    parser.add_argument("--free-bits", type=float, default=DEFAULTS["free_bits"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=DEFAULTS["log_interval"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building multi-regime data...")
    df, regimes = build_multi_regime_df(n_days=500, seed=args.seed)
    ds = RSSMSequenceDataset(df, seq_len=args.seq_len, lookback=args.lookback)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"Data: {len(ds)} sequences, train={len(train_ds)}, val={len(val_ds)}")

    # Get dimensions from a sample
    sample = ds[0]
    tech_dim = sample[0].shape[-1]
    fund_dim = sample[1].shape[-1]
    action_dim = sample[2].shape[-1]
    print(f"Dims: tech={tech_dim}, fund={fund_dim}, action={action_dim}")

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = MarketEncoder(tech_dim, fund_dim, args.embed_dim).to(device)
    rssm = RSSM(args.embed_dim, action_dim, args.hidden_dim, args.latent_dim).to(device)
    reward_decoder = RSSMRewardDecoder(args.hidden_dim, args.latent_dim).to(device)

    n_params = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in rssm.parameters())
        + sum(p.numel() for p in reward_decoder.parameters())
    )
    print(f"Total parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    all_params = list(encoder.parameters()) + list(rssm.parameters()) + list(reward_decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    # ── KL Annealer ───────────────────────────────────────────────────────────
    annealer = KLAnnealer(
        anneal_steps=args.anneal_steps,
        mode="monotonic",
        free_bits=args.free_bits,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    best_val_loss = float("inf")

    print(f"\n{'='*65}")
    print(f"Phase 5: RSSM Training — {args.epochs} epochs")
    print(f"  Anneal steps: {args.anneal_steps}, Free bits: {args.free_bits}")
    print(f"  Seq len: {args.seq_len}, Latent: {args.latent_dim}")
    print(f"{'='*65}\n")

    for epoch in range(args.epochs):
        t_start = time.time()

        # Train
        encoder.train()
        rssm.train()
        reward_decoder.train()

        total_kl = total_reward = total_loss = 0.0
        n_batches = 0

        for batch_idx, (tech_seq, fund_seq, action_seq, reward_seq) in enumerate(train_loader):
            # tech_seq: [B, S, L, tech_dim] or [S, B, L, tech_dim] depending on collation
            # DataLoader default stacks batch first. We want [S, B, ...] for RSSM.
            # Current shape: [B, S, L, tech_dim]
            B, S, L, Tdim = tech_seq.shape
            tech_seq = tech_seq.permute(1, 0, 2, 3).to(device)  # [S, B, L, tech_dim]
            fund_seq = fund_seq.permute(1, 0, 2).to(device)     # [S, B, fund_dim]
            action_seq = action_seq.permute(1, 0, 2).to(device)  # [S, B, action_dim]
            reward_seq = reward_seq.permute(1, 0).to(device)    # [S, B]

            # Encode observations step by step
            e_list = []
            for t in range(S):
                e_t = encoder(tech_seq[t], fund_seq[t])  # [B, embed_dim]
                e_list.append(e_t)
            e_seq = torch.stack(e_list)  # [S, B, embed_dim]

            # RSSM observe rollout
            rssm_out = rssm.observe_rollout(e_seq, action_seq)

            # Predict returns from all states
            h_flat = rssm_out["h"].reshape(-1, args.hidden_dim)  # [S*B, hidden]
            z_flat = rssm_out["z"].reshape(-1, args.latent_dim)  # [S*B, latent]
            pred_return = reward_decoder(h_flat, z_flat).view(S, B)  # [S, B]

            # Loss
            loss, metrics = rssm_phase5_loss(
                rssm_out, pred_return, reward_seq,
                annealer, reward_weight=args.reward_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

            total_kl += metrics["kl_loss"]
            total_reward += metrics["reward_loss"]
            total_loss += metrics["total_loss"]
            n_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"  Batch {batch_idx+1:3d}/{len(train_loader)} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"KL: {metrics['kl_loss']:.4f} (w={metrics['kl_weight']:.3f}) | "
                    f"Reward: {metrics['reward_loss']:.4f}"
                )

        avg_train_kl = total_kl / n_batches
        avg_train_reward = total_reward / n_batches
        avg_train_loss = total_loss / n_batches

        # Validate
        encoder.eval()
        rssm.eval()
        reward_decoder.eval()
        val_total = 0.0
        val_kl = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for tech_seq, fund_seq, action_seq, reward_seq in val_loader:
                B, S, L, Tdim = tech_seq.shape
                tech_seq = tech_seq.permute(1, 0, 2, 3).to(device)
                fund_seq = fund_seq.permute(1, 0, 2).to(device)
                action_seq = action_seq.permute(1, 0, 2).to(device)
                reward_seq = reward_seq.permute(1, 0).to(device)

                e_list = [encoder(tech_seq[t], fund_seq[t]) for t in range(S)]
                e_seq = torch.stack(e_list)
                rssm_out = rssm.observe_rollout(e_seq, action_seq)
                h_flat = rssm_out["h"].reshape(-1, args.hidden_dim)
                z_flat = rssm_out["z"].reshape(-1, args.latent_dim)
                pred_return = reward_decoder(h_flat, z_flat).view(S, B)
                loss, m = rssm_phase5_loss(
                    rssm_out, pred_return, reward_seq,
                    annealer, reward_weight=args.reward_weight,
                )
                val_total += m["total_loss"]
                val_kl += m["kl_loss"]
                n_val_batches += 1

        avg_val_loss = val_total / n_val_batches
        avg_val_kl = val_kl / n_val_batches

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train KL: {avg_train_kl:.4f} Reward: {avg_train_reward:.4f} | "
            f"Val KL: {avg_val_kl:.4f} Total: {avg_val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "rssm_state": rssm.state_dict(),
                "reward_decoder_state": reward_decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "annealer_step": annealer.step,
                "config": vars(args),
            }, checkpoint_dir / "phase5_best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "rssm_state": rssm.state_dict(),
                "reward_decoder_state": reward_decoder.state_dict(),
            }, checkpoint_dir / f"phase5_epoch_{epoch+1:03d}.pt")

    # ── Final ─────────────────────────────────────────────────────────────────
    torch.save({
        "epoch": args.epochs,
        "encoder_state": encoder.state_dict(),
        "rssm_state": rssm.state_dict(),
        "reward_decoder_state": reward_decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "annealer_step": annealer.step,
        "config": vars(args),
    }, checkpoint_dir / "phase5_final.pt")

    # Check posterior collapse
    print(f"\n{'='*65}")
    print("Posterior Collapse Check")
    print(f"{'='*65}")
    print(f"Final KL div: {avg_train_kl:.4f}")
    if avg_train_kl < 0.01:
        print("⚠️  WARNING: KL < 0.01 — possible posterior collapse. Increase free_bits or anneal_steps.")
    elif avg_train_kl < 0.5:
        print("✓  KL > 0.01 — stochastic bottleneck is alive (healthy range).")
    else:
        print("✓  KL > 0.5 — strong stochastic signal flowing through bottleneck.")
    print(f"Annealer step: {annealer.step}, Final KL weight: {annealer():.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
