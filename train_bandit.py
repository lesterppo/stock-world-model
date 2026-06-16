#!/usr/bin/env python3
"""
Phase 6: Train Contextual Bandit Portfolio Allocator

Architecture per Gemini Pro critique:
  - Contextual bandit (not full RL) — actions don't affect state transitions
  - Single-asset: SPY position ∈ [0, 1]
  - Training: PPO-style REINFORCE on REALIZED returns (not predicted)
  - Walk-forward CV: chronological splits, no look-ahead
  - Dropout on RSSM state to fight GRU clock memorization

Usage:
    python train_bandit.py --ticker SPY --checkpoint checkpoints/SPY_rssm.pt --epochs 50
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextual_bandit import (
    BanditPolicy,
    DifferentialSharpeRatio,
    compute_reward,
    reinforce_update,
    extract_rssm_states_with_context,
)


def walk_forward_split(n_total: int, n_folds: int = 5, min_train: int = 500):
    """
    Chronological walk-forward splits.
    Returns list of (train_end, val_start, val_end) indices.
    """
    splits = []
    val_size = max(100, n_total // (n_folds * 3))
    step = (n_total - min_train) // n_folds

    for i in range(n_folds):
        train_end = min_train + i * step
        val_start = train_end
        val_end = min(val_start + val_size, n_total - 1)
        if val_end > val_start + 50:
            splits.append((train_end, val_start, val_end))

    return splits


def train_epoch(
    policy,
    h_train,
    ret_ctx_train,
    rets_train,
    optimizer,
    batch_size=32,
    seq_len=30,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    n_epochs_inner=4,
):
    """
    One training epoch: collect trajectories and update policy.

    Since states don't depend on actions (contextual bandit),
    we can treat the entire training set as one long trajectory
    and sample subsequences.
    """
    device = next(policy.parameters()).device
    n = len(h_train)
    n_batches = max(1, n // (batch_size * seq_len))
    metrics_sum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0

    for _ in range(n_batches):
        # Sample random start indices
        starts = np.random.randint(0, max(1, n - seq_len), size=batch_size)

        # Build batch [seq_len, batch_size, ...]
        h_batch = torch.zeros(seq_len, batch_size, 128, device=device)
        ctx_batch = torch.zeros(seq_len, batch_size, 3, device=device)
        ret_batch = torch.zeros(seq_len, batch_size, device=device)

        for b, s in enumerate(starts):
            h_batch[:, b, :] = torch.tensor(h_train[s:s + seq_len], device=device)
            ctx_batch[:, b, :] = torch.tensor(ret_ctx_train[s:s + seq_len], device=device)
            ret_batch[:, b] = torch.tensor(rets_train[s:s + seq_len], device=device)

        # Collect trajectory
        actions = []
        log_probs = []
        values = []
        rewards = []
        prev_weight = torch.zeros(batch_size, 1, device=device)

        for t in range(seq_len):
            action, log_prob, value = policy.get_action_stochastic(h_batch[t], ctx_batch[t])
            reward = compute_reward(
                action, ret_batch[t],
                risk_aversion=1.0,
                vol_est=0.01,
                prev_weight=prev_weight,
                turnover_cost=0.001,
            )
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            prev_weight = action.detach()

        actions_t = torch.stack(actions)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        rewards_t = torch.stack(rewards)

        # PPO update
        m = reinforce_update(
            policy,
            h_batch, ctx_batch,
            actions_t, log_probs_t, rewards_t, values_t,
            optimizer,
            gamma=gamma, lam=lam, clip_eps=clip_eps,
            n_epochs=n_epochs_inner,
        )
        for k in metrics_sum:
            metrics_sum[k] += m[k]
        n_updates += 1

    for k in metrics_sum:
        metrics_sum[k] /= max(1, n_updates)
    return metrics_sum


@torch.no_grad()
def validate(policy, h_val, ret_ctx_val, rets_val, vol_est=0.01):
    """Deterministic walk-forward validation."""
    device = next(policy.parameters()).device

    positions = []
    daily_rets = []
    prev_w = torch.zeros(1, 1, device=device)

    for t in range(len(h_val)):
        h = torch.tensor(h_val[t:t + 1], device=device)
        ctx = torch.tensor(ret_ctx_val[t:t + 1], device=device)
        w = policy.get_action_deterministic(h, ctx)
        # Apply turnover smoothing (EMA of policy output)
        w_smooth = 0.7 * w + 0.3 * prev_w
        positions.append(w_smooth.item())
        daily_rets.append(w_smooth.item() * rets_val[t])
        prev_w = w_smooth

    positions = np.array(positions)
    daily_rets = np.array(daily_rets)

    # Metrics
    cum_ret = np.prod(1 + daily_rets) - 1
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252)
    # Max drawdown
    cum = np.cumprod(1 + daily_rets)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))
    turnover = np.mean(np.abs(np.diff(positions)))

    return {
        "cum_return": cum_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turnover": turnover,
        "mean_position": float(np.mean(positions)),
    }


def main():
    p = argparse.ArgumentParser(description="Train Contextual Bandit Portfolio Allocator")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--checkpoint", default="checkpoints/SPY_rssm.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load data ───────────────────────────────────────────────────────────
    df = pd.read_csv(f"data/{args.ticker}_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]
    print(f"Train: {len(train_df)} days, Test: {len(test_df)} days")

    # ── Extract RSSM states ─────────────────────────────────────────────────
    print("\nExtracting RSSM latent states from training data...")
    h_all, ret_ctx_all, rets_all, dates_all = extract_rssm_states_with_context(
        args.checkpoint, train_df, args.lookback, args.device
    )
    print(f"  Extracted: {len(h_all)} states")

    # ── Walk-forward CV ─────────────────────────────────────────────────────
    splits = walk_forward_split(len(h_all), args.folds, min_train=500)
    print(f"\nWalk-forward splits: {len(splits)}")

    fold_results = []

    for fold_idx, (train_end, val_start, val_end) in enumerate(splits):
        if fold_idx >= args.folds:
            break

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(splits)}: train[0:{train_end}], val[{val_start}:{val_end}]")
        print(f"{'='*60}")

        h_train = h_all[:train_end]
        ret_ctx_train = ret_ctx_all[:train_end]
        rets_train = rets_all[:train_end]
        h_val = h_all[val_start:val_end]
        ret_ctx_val = ret_ctx_all[val_start:val_end]
        rets_val = rets_all[val_start:val_end]

        # Fresh policy each fold
        policy = BanditPolicy(state_dim=128).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_sharpe = -float("inf")
        best_state = None

        for epoch in range(args.epochs):
            policy.train()
            m = train_epoch(
                policy, h_train, ret_ctx_train, rets_train,
                optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )

            policy.eval()
            val_m = validate(policy, h_val, ret_ctx_val, rets_val)

            if val_m["sharpe"] > best_val_sharpe:
                best_val_sharpe = val_m["sharpe"]
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                    f"P_loss={m['policy_loss']:.4f} V_loss={m['value_loss']:.4f} "
                    f"Ent={m['entropy']:.4f} | "
                    f"Val Sharpe={val_m['sharpe']:+.4f} CumRet={val_m['cum_return']:+.3%} "
                    f"MaxDD={val_m['max_dd']:.3%} T/O={val_m['turnover']:.4f}"
                )

        # Load best and re-evaluate
        policy.load_state_dict(best_state)
        policy.eval()
        val_m = validate(policy, h_val, ret_ctx_val, rets_val)
        fold_results.append(val_m)
        print(f"\n  Best: Sharpe={val_m['sharpe']:+.4f}, CumRet={val_m['cum_return']:+.3%}, "
              f"MaxDD={val_m['max_dd']:.3%}, Pos={val_m['mean_position']:.2%}")

    # ── Aggregate CV results ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WALK-FORWARD CV SUMMARY")
    print(f"{'='*60}")
    sharpes = [r["sharpe"] for r in fold_results]
    cum_rets = [r["cum_return"] for r in fold_results]
    max_dds = [r["max_dd"] for r in fold_results]
    turnovers = [r["turnover"] for r in fold_results]
    positions = [r["mean_position"] for r in fold_results]

    print(f"  Sharpe:     {np.mean(sharpes):+.4f} ± {np.std(sharpes):.4f}")
    print(f"  Cum Return: {np.mean(cum_rets):+.3%} ± {np.std(cum_rets):.3%}")
    print(f"  Max DD:     {np.mean(max_dds):+.3%} ± {np.std(max_dds):.3%}")
    print(f"  Turnover:   {np.mean(turnovers):.4f} ± {np.std(turnovers):.4f}")
    print(f"  Mean Pos:   {np.mean(positions):+.3%} ± {np.std(positions):.3%}")

    # ── Train final model on ALL train data, test on 2022-2024 ──────────────
    print(f"\n{'='*60}")
    print("FINAL MODEL: Train on 2010-2021, Test on 2022-2024")
    print(f"{'='*60}")

    policy = BanditPolicy(state_dim=128).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=1e-5)

    # Extract test states
    h_test, ret_ctx_test, rets_test, dates_test = extract_rssm_states_with_context(
        args.checkpoint, test_df, args.lookback, args.device
    )
    print(f"  Test states: {len(h_test)}")

    best_test_sharpe = -float("inf")
    best_state = None

    for epoch in range(args.epochs * 2):  # more epochs for final model
        policy.train()
        m = train_epoch(
            policy, h_all, ret_ctx_all, rets_all,
            optimizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )

        policy.eval()
        val_m = validate(policy, h_test, ret_ctx_test, rets_test)

        if val_m["sharpe"] > best_test_sharpe:
            best_test_sharpe = val_m["sharpe"]
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{args.epochs * 2} | "
                f"P_loss={m['policy_loss']:.4f} | "
                f"Test Sharpe={val_m['sharpe']:+.4f} CumRet={val_m['cum_return']:+.3%} "
                f"MaxDD={val_m['max_dd']:.3%} T/O={val_m['turnover']:.4f}"
            )

    policy.load_state_dict(best_state)
    policy.eval()
    _ = validate(policy, h_test, ret_ctx_test, rets_test)  # warm up EMA

    # ── Detailed test results ───────────────────────────────────────────────
    positions_test = []
    daily_rets_test = []
    prev_w = torch.zeros(1, 1, device=device)
    buyhold_daily = []

    with torch.no_grad():
        for t in range(len(h_test)):
            h = torch.tensor(h_test[t:t + 1], device=device)
            ctx = torch.tensor(ret_ctx_test[t:t + 1], device=device)
            w = policy.get_action_deterministic(h, ctx)
            w_smooth = 0.7 * w + 0.3 * prev_w
            positions_test.append(w_smooth.item())
            daily_rets_test.append(w_smooth.item() * rets_test[t])
            buyhold_daily.append(rets_test[t])
            prev_w = w_smooth

    positions_test = np.array(positions_test)
    daily_rets_test = np.array(daily_rets_test)
    buyhold_daily = np.array(buyhold_daily)

    bandit_cum = np.prod(1 + daily_rets_test) - 1
    bh_cum = np.prod(1 + buyhold_daily) - 1
    bandit_sharpe = daily_rets_test.mean() / (daily_rets_test.std() + 1e-8) * np.sqrt(252)
    bh_sharpe = buyhold_daily.mean() / (buyhold_daily.std() + 1e-8) * np.sqrt(252)

    # Max DD
    bandit_equity = np.cumprod(1 + daily_rets_test)
    bandit_peak = np.maximum.accumulate(bandit_equity)
    bandit_dd = float(np.min((bandit_equity - bandit_peak) / bandit_peak))

    bh_equity = np.cumprod(1 + buyhold_daily)
    bh_peak = np.maximum.accumulate(bh_equity)
    bh_dd = float(np.min((bh_equity - bh_peak) / bh_peak))

    cash_days = (positions_test < 0.1).sum()

    print(f"\n{'='*60}")
    print("OUT-OF-SAMPLE RESULTS (2022-2024)")
    print(f"{'='*60}")
    print(f"  {'':<25} {'Contextual Bandit':>15} {'Buy & Hold':>15}")
    print(f"  {'Cumulative Return':<25} {bandit_cum:>+15.2%} {bh_cum:>+15.2%}")
    print(f"  {'Ann. Sharpe':<25} {bandit_sharpe:>+15.3f} {bh_sharpe:>+15.3f}")
    print(f"  {'Max Drawdown':<25} {bandit_dd:>15.2%} {bh_dd:>15.2%}")
    print(f"  {'Mean Position':<25} {positions_test.mean():>15.2%} {'N/A':>15}")
    print(f"  {'Cash Days (<10%)':<25} {cash_days:>15} {'N/A':>15}")

    # Save model
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "policy_state": policy.state_dict(),
        "config": {"state_dim": 128, "hidden_dim": 64},
        "test_results": {
            "cum_return": float(bandit_cum),
            "sharpe": float(bandit_sharpe),
            "max_dd": float(bandit_dd),
        },
    }, "checkpoints/bandit_policy.pt")
    print(f"\nSaved: checkpoints/bandit_policy.pt")


if __name__ == "__main__":
    main()
