#!/usr/bin/env python3
"""
Stock World Model — Phase 4: Contrastive World Model Training

Extends Phase 1 self-supervised training with an InfoNCE contrastive loss
that pulls together latent states from similar macro regimes and pushes apart
states from different regimes.

Loss = KL(z_pred || z_true) + α·MSE(reward_pred, reward_true) + β·InfoNCE(z_proj)

The contrastive head maps V-Encoder states to an L2-normalized space where
temporally-adjacent states (same regime) are positive pairs and all other
in-batch samples are negatives (SimCLR-style).

This improves the V-Encoder's generalization to unseen market regimes.

Usage:
    python train_phase4.py --epochs 50
    python train_phase4.py --checkpoint checkpoints/phase1_final.pt --epochs 30  # finetune
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MacroStockEncoder, StockDynamicsModel, RewardDecoder, ContrastiveHead
from losses import phase1_loss, contrastive_loss
from data import (
    build_multi_regime_df,
    Phase1Dataset,
    TECH_FEATURE_COLS,
    FUND_FEATURE_COLS,
    MACRO_ACTION_COLS,
    MICRO_ACTION_COLS,
)


DEFAULTS = {
    "latent_dim": 64,
    "proj_dim": 32,
    "lookback_window": 60,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    "kl_weight": 1.0,
    "reward_weight": 0.1,
    "contrastive_weight": 0.05,
    "contrastive_temperature": 0.07,
    "grad_clip": 1.0,
    "val_split": 0.1,
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "log_interval": 10,
}


def load_data(lookback: int, n_days: int = 500, seed: int = 42):
    """Build multi-regime mock data and Phase 1 dataset."""
    df, _regimes = build_multi_regime_df(n_days=n_days, seed=seed)
    full_ds = Phase1Dataset(df, lookback_window=lookback)

    n_val = max(1, int(len(full_ds) * DEFAULTS["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_ds, val_ds


def train_epoch(
    v_encoder, m_dynamics, reward_decoder, contrastive_head,
    dataloader, optimizer, device,
    kl_weight, reward_weight, contrastive_weight, temperature, grad_clip, log_interval,
):
    v_encoder.train()
    m_dynamics.train()
    reward_decoder.train()
    contrastive_head.train()

    total_kl = total_reward = total_contrastive = total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        (tech_seq_t, fund_vec_t, macro_t, micro_t, reward_true,
         tech_seq_t1, fund_vec_t1) = [x.to(device) for x in batch]

        B = tech_seq_t.shape[0]

        # ── 1. Encode current state (use mu for contrastive, sample for dynamics)
        z_mu_t, z_logvar_t, z_sample_t = v_encoder(tech_seq_t, fund_vec_t)

        # ── 2. M-Dynamics: predict next state
        _, z_pred_mu, z_pred_logvar = m_dynamics(z_sample_t, macro_t, micro_t)

        # ── 3. Encode true next state
        z_mu_t1, z_logvar_t1, _ = v_encoder(tech_seq_t1, fund_vec_t1)

        # ── 4. Reward prediction
        pred_return = reward_decoder(z_sample_t)

        # ── 5. Phase 1 losses (KL + reward MSE)
        loss_p1, metrics_p1 = phase1_loss(
            z_pred_mu, z_pred_logvar, z_mu_t1, z_logvar_t1,
            pred_return, reward_true,
            kl_weight=kl_weight, reward_weight=reward_weight,
        )

        # ── 6. Contrastive loss (InfoNCE)
        if contrastive_weight > 0 and B >= 4:
            loss_contr = contrastive_loss(
                z_mu_t, macro_t, contrastive_head, temperature=temperature,
            )
        else:
            loss_contr = torch.tensor(0.0, device=device)
            if B < 4 and contrastive_weight > 0:
                # Skip contrastive for tiny batches (can't form enough pairs)
                pass

        # ── 7. Combined loss
        loss = loss_p1 + contrastive_weight * loss_contr

        # ── 8. Backprop
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(v_encoder.parameters())
                + list(m_dynamics.parameters())
                + list(reward_decoder.parameters())
                + list(contrastive_head.parameters()),
                grad_clip,
            )
        optimizer.step()

        total_kl += metrics_p1["kl_loss"]
        total_reward += metrics_p1["reward_loss"]
        total_contrastive += loss_contr.item()
        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  Batch {batch_idx + 1:4d}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"KL: {metrics_p1['kl_loss']:.4f} | "
                f"Reward: {metrics_p1['reward_loss']:.4f} | "
                f"Contr: {loss_contr.item():.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            t0 = time.time()

    return {
        "kl_loss": total_kl / n_batches,
        "reward_loss": total_reward / n_batches,
        "contrastive_loss": total_contrastive / n_batches,
        "total_loss": total_loss / n_batches,
    }


@torch.no_grad()
def validate_epoch(
    v_encoder, m_dynamics, reward_decoder, contrastive_head,
    dataloader, device,
    kl_weight, reward_weight, contrastive_weight, temperature,
):
    v_encoder.eval()
    m_dynamics.eval()
    reward_decoder.eval()
    contrastive_head.eval()

    total_kl = total_reward = total_contrastive = total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        (tech_seq_t, fund_vec_t, macro_t, micro_t, reward_true,
         tech_seq_t1, fund_vec_t1) = [x.to(device) for x in batch]

        B = tech_seq_t.shape[0]
        z_mu_t, z_logvar_t, z_sample_t = v_encoder(tech_seq_t, fund_vec_t)
        _, z_pred_mu, z_pred_logvar = m_dynamics(z_sample_t, macro_t, micro_t)
        z_mu_t1, z_logvar_t1, _ = v_encoder(tech_seq_t1, fund_vec_t1)
        pred_return = reward_decoder(z_sample_t)

        loss_p1, metrics_p1 = phase1_loss(
            z_pred_mu, z_pred_logvar, z_mu_t1, z_logvar_t1,
            pred_return, reward_true,
            kl_weight=kl_weight, reward_weight=reward_weight,
        )

        loss_contr = torch.tensor(0.0, device=device)
        if contrastive_weight > 0 and B >= 4:
            loss_contr = contrastive_loss(
                z_mu_t, macro_t, contrastive_head, temperature=temperature,
            )

        loss = loss_p1 + contrastive_weight * loss_contr

        total_kl += metrics_p1["kl_loss"]
        total_reward += metrics_p1["reward_loss"]
        total_contrastive += loss_contr.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        "kl_loss": total_kl / n_batches,
        "reward_loss": total_reward / n_batches,
        "contrastive_loss": total_contrastive / n_batches,
        "total_loss": total_loss / n_batches,
    }


def save_checkpoint(models, optimizer, epoch, metrics, config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "v_encoder_state": models["v_encoder"].state_dict(),
        "m_dynamics_state": models["m_dynamics"].state_dict(),
        "reward_decoder_state": models["reward_decoder"].state_dict(),
        "contrastive_head_state": models["contrastive_head"].state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_metrics": metrics.get("train", {}),
        "val_metrics": metrics.get("val", {}),
        "config": config,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, models, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    models["v_encoder"].load_state_dict(ckpt["v_encoder_state"])
    models["m_dynamics"].load_state_dict(ckpt["m_dynamics_state"])
    models["reward_decoder"].load_state_dict(ckpt["reward_decoder_state"])
    if "contrastive_head_state" in ckpt:
        models["contrastive_head"].load_state_dict(ckpt["contrastive_head_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Contrastive World Model Training")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--latent-dim", type=int, default=DEFAULTS["latent_dim"])
    parser.add_argument("--proj-dim", type=int, default=DEFAULTS["proj_dim"])
    parser.add_argument("--lookback", type=int, default=DEFAULTS["lookback_window"])
    parser.add_argument("--kl-weight", type=float, default=DEFAULTS["kl_weight"])
    parser.add_argument("--reward-weight", type=float, default=DEFAULTS["reward_weight"])
    parser.add_argument("--contrastive-weight", type=float, default=DEFAULTS["contrastive_weight"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["contrastive_temperature"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=DEFAULTS["log_interval"])
    parser.add_argument("--no-contrastive", action="store_true",
                        help="Disable contrastive loss (baseline mode)")
    args = parser.parse_args()

    if args.no_contrastive:
        args.contrastive_weight = 0.0

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = load_data(lookback=args.lookback)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    sample = train_ds[0]
    tech_dim = sample[0].shape[-1]
    fund_dim = sample[1].shape[-1]
    macro_dim = sample[2].shape[-1]
    micro_dim = sample[3].shape[-1]
    print(f"Dims: tech={tech_dim}, fund={fund_dim}, macro={macro_dim}, micro={micro_dim}")

    # ── Models ────────────────────────────────────────────────────────────────
    v_encoder = MacroStockEncoder(tech_dim, fund_dim, args.latent_dim).to(device)
    m_dynamics = StockDynamicsModel(args.latent_dim, macro_dim, micro_dim).to(device)
    reward_decoder = RewardDecoder(args.latent_dim).to(device)
    contrastive_head = ContrastiveHead(args.latent_dim, args.proj_dim).to(device)

    models = {
        "v_encoder": v_encoder,
        "m_dynamics": m_dynamics,
        "reward_decoder": reward_decoder,
        "contrastive_head": contrastive_head,
    }

    n_params = sum(
        sum(p.numel() for p in m.parameters()) for m in models.values()
    )
    print(f"Total parameters: {n_params:,}")
    print(f"Contrastive weight: {args.contrastive_weight}, temperature: {args.temperature}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    all_params = []
    for m in models.values():
        all_params.extend(m.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), models, optimizer, device)

    # ── Training ──────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    best_val_loss = float("inf")

    print(f"\n{'='*65}")
    print(f"Phase 4: Contrastive World Model — {args.epochs} epochs")
    print(f"  KL: {args.kl_weight}, Reward: {args.reward_weight}, "
          f"Contrastive: {args.contrastive_weight}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()

        train_metrics = train_epoch(
            v_encoder, m_dynamics, reward_decoder, contrastive_head,
            train_loader, optimizer, device,
            kl_weight=args.kl_weight,
            reward_weight=args.reward_weight,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )

        val_metrics = validate_epoch(
            v_encoder, m_dynamics, reward_decoder, contrastive_head,
            val_loader, device,
            kl_weight=args.kl_weight,
            reward_weight=args.reward_weight,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_metrics['total_loss']:.4f} "
            f"(KL:{train_metrics['kl_loss']:.4f} "
            f"R:{train_metrics['reward_loss']:.4f} "
            f"C:{train_metrics['contrastive_loss']:.4f}) | "
            f"Val: {val_metrics['total_loss']:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_checkpoint(
                models, optimizer, epoch + 1,
                {"train": train_metrics, "val": val_metrics},
                vars(args),
                checkpoint_dir / "phase4_best.pt",
            )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                models, optimizer, epoch + 1,
                {"train": train_metrics, "val": val_metrics},
                vars(args),
                checkpoint_dir / f"phase4_epoch_{epoch + 1:03d}.pt",
            )

    # ── Final ─────────────────────────────────────────────────────────────────
    save_checkpoint(
        models, optimizer, args.epochs,
        {"train": train_metrics, "val": val_metrics},
        vars(args),
        checkpoint_dir / "phase4_final.pt",
    )

    # ── Contrastive quality metrics ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Contrastive Embedding Quality")
    print(f"{'='*65}")

    v_encoder.eval()
    contrastive_head.eval()
    with torch.no_grad():
        # Sample a batch and compute embedding similarities
        batch = next(iter(val_loader))
        tech_seq, fund_vec, macro_t, _, _, _, _ = [x.to(device) for x in batch]
        z_mu, _, _ = v_encoder(tech_seq, fund_vec)
        proj = contrastive_head(z_mu)

        # Within-pair similarity (consecutive samples)
        if proj.shape[0] >= 2:
            pair_sim = (proj[0::2] * proj[1::2]).sum(dim=-1)
            # Between-pair similarity (random)
            all_sim = proj @ proj.T
            mask = torch.eye(all_sim.shape[0], device=device, dtype=torch.bool)
            all_sim = all_sim[~mask]

            print(f"  Avg within-pair cosine sim:  {pair_sim.mean().item():.4f}")
            print(f"  Avg between-pair cosine sim: {all_sim.mean().item():.4f}")
            print(f"  Separation ratio (within/between): "
                  f"{pair_sim.mean().item() / (all_sim.mean().item() + 1e-8):.2f}x")
            print(f"  (Higher ratio = better regime clustering)")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
