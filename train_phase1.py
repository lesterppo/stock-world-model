#!/usr/bin/env python3
"""
Stock World Model — Phase 1: Self-Supervised World Model Training

Trains V-Encoder + M-Dynamics + Reward Decoder jointly using:
  - KL divergence: M-Dynamics prediction vs V-Encoder encoding of next state
  - MSE reward loss: predicted next-day return vs actual

Freezes the Controller (not yet implemented in Phase 1).
The trained world model can then be used in Phase 2 for latent imagination RL.

Usage:
    python train_phase1.py                          # mock data, 50 epochs
    python train_phase1.py --real                   # real FRED data (needs internet)
    python train_phase1.py --epochs 200 --lr 3e-4   # custom hyperparams
    python train_phase1.py --resume checkpoints/phase1_epoch_050.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MacroStockEncoder, StockDynamicsModel, RewardDecoder
from losses import phase1_loss
from data import (
    build_mock_fused_df,
    build_macro_feature_matrix,
    fuse_stock_and_macro,
    generate_mock_stock_data,
    align_asymmetric_pipeline,
    Phase1Dataset,
    TECH_FEATURE_COLS,
    FUND_FEATURE_COLS,
    MACRO_ACTION_COLS,
    MICRO_ACTION_COLS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "latent_dim": 64,
    "lookback_window": 60,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    "kl_weight": 1.0,
    "reward_weight": 0.1,
    "grad_clip": 1.0,
    "val_split": 0.1,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "log_interval": 10,  # batches
}

CHECKPOINT_DIR = Path(DEFAULTS["checkpoint_dir"])


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_real_data(lookback: int) -> tuple[Phase1Dataset, Phase1Dataset]:
    """Download real FRED macro data + generate mock stock data, fuse them."""
    print("Downloading macro data from FRED...")
    macro_df = build_macro_feature_matrix(
        start_date="2019-12-01", end_date="2023-01-01"
    )
    print(f"  Macro data: {len(macro_df)} days, columns={list(macro_df.columns)}")

    print("Generating mock stock data...")
    daily_df, fund_df = generate_mock_stock_data(
        start_date="2020-01-01", end_date="2022-12-31"
    )
    print(f"  Daily data: {len(daily_df)} days")

    # PIT alignment
    aligned = align_asymmetric_pipeline(daily_df, fund_df)
    print(f"  Aligned: {len(aligned)} days")

    # Fuse with macro
    fused = fuse_stock_and_macro(aligned, macro_df)
    print(f"  Fused: {len(fused)} days")

    # Ensure all required columns exist
    required = (
        TECH_FEATURE_COLS
        + FUND_FEATURE_COLS
        + MACRO_ACTION_COLS
        + MICRO_ACTION_COLS
    )
    missing = [c for c in required if c not in fused.columns]
    if missing:
        raise KeyError(f"Missing columns in fused DataFrame: {missing}")

    # Split
    full_ds = Phase1Dataset(fused, lookback_window=lookback)
    n_val = max(1, int(len(full_ds) * DEFAULTS["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(DEFAULTS["seed"]),
    )
    print(f"  Samples: train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds


def load_mock_data(lookback: int, n_days: int = 500) -> tuple[Phase1Dataset, Phase1Dataset]:
    """Build fully synthetic data for fast iteration."""
    print(f"Building mock fused data ({n_days} days)...")
    df = build_mock_fused_df(n_days=n_days, seed=DEFAULTS["seed"])
    print(f"  Mock data: {len(df)} days, columns={list(df.columns)}")

    full_ds = Phase1Dataset(df, lookback_window=lookback)
    n_val = max(1, int(len(full_ds) * DEFAULTS["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(DEFAULTS["seed"]),
    )
    print(f"  Samples: train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════


def train_epoch(
    v_encoder: MacroStockEncoder,
    m_dynamics: StockDynamicsModel,
    reward_decoder: RewardDecoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    reward_weight: float,
    grad_clip: float,
    log_interval: int,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch. Returns average metrics."""
    v_encoder.train()
    m_dynamics.train()
    reward_decoder.train()

    total_kl = 0.0
    total_reward = 0.0
    total_loss = 0.0
    n_batches = 0

    t0 = time.time()
    for batch_idx, batch in enumerate(dataloader):
        (
            tech_seq_t,    # [B, L, tech_dim]
            fund_vec_t,    # [B, fund_dim]
            macro_t,       # [B, macro_dim]
            micro_t,       # [B, micro_dim]
            reward_true,   # [B]
            tech_seq_t1,   # [B, L, tech_dim]
            fund_vec_t1,   # [B, fund_dim]
        ) = [x.to(device) for x in batch]

        # 1. Encode current state z_t
        z_mu_t, z_logvar_t, z_sample_t = v_encoder(tech_seq_t, fund_vec_t)

        # 2. M-Dynamics: predict next state distribution
        #    Use z_sample_t for the forward pass (not mu — reparameterized)
        _, z_pred_mu, z_pred_logvar = m_dynamics(z_sample_t, macro_t, micro_t)

        # 3. Encode true next state z_{t+1}
        z_mu_t1, z_logvar_t1, _ = v_encoder(tech_seq_t1, fund_vec_t1)

        # 4. Reward Decoder: predict next-day return from z_t
        pred_return = reward_decoder(z_sample_t)

        # 5. Compute combined loss
        loss, metrics = phase1_loss(
            z_pred_mu,
            z_pred_logvar,
            z_mu_t1,
            z_logvar_t1,
            pred_return,
            reward_true,
            kl_weight=kl_weight,
            reward_weight=reward_weight,
        )

        # 6. Backprop
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(v_encoder.parameters())
                + list(m_dynamics.parameters())
                + list(reward_decoder.parameters()),
                grad_clip,
            )
        optimizer.step()

        # Accumulate
        total_kl += metrics["kl_loss"]
        total_reward += metrics["reward_loss"]
        total_loss += metrics["total_loss"]
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  Batch {batch_idx + 1:4d}/{len(dataloader)} | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"KL: {metrics['kl_loss']:.4f} | "
                f"Reward: {metrics['reward_loss']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            t0 = time.time()

    return {
        "kl_loss": total_kl / n_batches,
        "reward_loss": total_reward / n_batches,
        "total_loss": total_loss / n_batches,
    }


@torch.no_grad()
def validate_epoch(
    v_encoder: MacroStockEncoder,
    m_dynamics: StockDynamicsModel,
    reward_decoder: RewardDecoder,
    dataloader: DataLoader,
    kl_weight: float,
    reward_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch."""
    v_encoder.eval()
    m_dynamics.eval()
    reward_decoder.eval()

    total_kl = 0.0
    total_reward = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        (
            tech_seq_t,
            fund_vec_t,
            macro_t,
            micro_t,
            reward_true,
            tech_seq_t1,
            fund_vec_t1,
        ) = [x.to(device) for x in batch]

        z_mu_t, z_logvar_t, z_sample_t = v_encoder(tech_seq_t, fund_vec_t)
        _, z_pred_mu, z_pred_logvar = m_dynamics(z_sample_t, macro_t, micro_t)
        z_mu_t1, z_logvar_t1, _ = v_encoder(tech_seq_t1, fund_vec_t1)
        pred_return = reward_decoder(z_sample_t)

        loss, metrics = phase1_loss(
            z_pred_mu, z_pred_logvar,
            z_mu_t1, z_logvar_t1,
            pred_return, reward_true,
            kl_weight=kl_weight,
            reward_weight=reward_weight,
        )

        total_kl += metrics["kl_loss"]
        total_reward += metrics["reward_loss"]
        total_loss += metrics["total_loss"]
        n_batches += 1

    return {
        "kl_loss": total_kl / n_batches,
        "reward_loss": total_reward / n_batches,
        "total_loss": total_loss / n_batches,
    }


def save_checkpoint(
    v_encoder: MacroStockEncoder,
    m_dynamics: StockDynamicsModel,
    reward_decoder: RewardDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    config: dict,
    path: Path,
):
    """Save full training state for resumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "v_encoder_state": v_encoder.state_dict(),
            "m_dynamics_state": m_dynamics.state_dict(),
            "reward_decoder_state": reward_decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    v_encoder: MacroStockEncoder,
    m_dynamics: StockDynamicsModel,
    reward_decoder: RewardDecoder,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    """Load training state. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    v_encoder.load_state_dict(ckpt["v_encoder_state"])
    m_dynamics.load_state_dict(ckpt["m_dynamics_state"])
    reward_decoder.load_state_dict(ckpt["reward_decoder_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Stock World Model — Phase 1 Self-Supervised Training"
    )
    parser.add_argument("--real", action="store_true", help="Use real FRED data (needs internet)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--latent-dim", type=int, default=DEFAULTS["latent_dim"])
    parser.add_argument("--lookback", type=int, default=DEFAULTS["lookback_window"])
    parser.add_argument("--kl-weight", type=float, default=DEFAULTS["kl_weight"])
    parser.add_argument("--reward-weight", type=float, default=DEFAULTS["reward_weight"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--log-interval", type=int, default=DEFAULTS["log_interval"],
                        help="Batches between log prints")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.real:
        train_ds, val_ds = load_real_data(lookback=args.lookback)
    else:
        train_ds, val_ds = load_mock_data(lookback=args.lookback, n_days=500)

    if len(train_ds) == 0:
        print("ERROR: Training dataset is empty. Increase n_days or reduce lookback.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # safe default
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print(f"DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Determine feature dimensions from a sample
    sample = train_ds[0]
    tech_dim = sample[0].shape[-1]   # tech_seq_t columns
    fund_dim = sample[1].shape[-1]   # fund_vec_t columns
    macro_dim = sample[2].shape[-1]  # macro_t columns
    micro_dim = sample[3].shape[-1]  # micro_t columns

    print(f"Feature dims: tech={tech_dim}, fund={fund_dim}, macro={macro_dim}, micro={micro_dim}")

    v_encoder = MacroStockEncoder(
        tech_dim=tech_dim, fund_dim=fund_dim, latent_dim=args.latent_dim
    ).to(device)

    m_dynamics = StockDynamicsModel(
        latent_dim=args.latent_dim, macro_dim=macro_dim, micro_dim=micro_dim
    ).to(device)

    reward_decoder = RewardDecoder(latent_dim=args.latent_dim).to(device)

    # Count parameters
    n_params = (
        sum(p.numel() for p in v_encoder.parameters())
        + sum(p.numel() for p in m_dynamics.parameters())
        + sum(p.numel() for p in reward_decoder.parameters())
    )
    print(f"Total parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(v_encoder.parameters())
        + list(m_dynamics.parameters())
        + list(reward_decoder.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            Path(args.resume), v_encoder, m_dynamics, reward_decoder, optimizer, device
        )

    # ── Training Loop ─────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Phase 1 Training — {args.epochs} epochs")
    print(f"  KL weight: {args.kl_weight}, Reward weight: {args.reward_weight}")
    print(f"  LR: {args.lr}, Batch size: {args.batch_size}, Grad clip: {args.grad_clip}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()

        # Train
        train_metrics = train_epoch(
            v_encoder, m_dynamics, reward_decoder,
            train_loader, optimizer,
            kl_weight=args.kl_weight,
            reward_weight=args.reward_weight,
            grad_clip=args.grad_clip,
            log_interval=DEFAULTS["log_interval"],
            device=device,
        )

        # Validate
        val_metrics = validate_epoch(
            v_encoder, m_dynamics, reward_decoder,
            val_loader,
            kl_weight=args.kl_weight,
            reward_weight=args.reward_weight,
            device=device,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} "
            f"(KL: {train_metrics['kl_loss']:.4f}, R: {train_metrics['reward_loss']:.4f}) | "
            f"Val Loss: {val_metrics['total_loss']:.4f} "
            f"(KL: {val_metrics['kl_loss']:.4f}, R: {val_metrics['reward_loss']:.4f}) | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Checkpoint
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_checkpoint(
                v_encoder, m_dynamics, reward_decoder, optimizer,
                epoch + 1, train_metrics, val_metrics,
                vars(args),
                checkpoint_dir / "phase1_best.pt",
            )

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                v_encoder, m_dynamics, reward_decoder, optimizer,
                epoch + 1, train_metrics, val_metrics,
                vars(args),
                checkpoint_dir / f"phase1_epoch_{epoch + 1:03d}.pt",
            )

    # ── Final Save ────────────────────────────────────────────────────────────
    save_checkpoint(
        v_encoder, m_dynamics, reward_decoder, optimizer,
        args.epochs, train_metrics, val_metrics,
        vars(args),
        checkpoint_dir / "phase1_final.pt",
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
