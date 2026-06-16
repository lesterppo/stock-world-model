#!/usr/bin/env python3
"""
Honest Evaluation — real out-of-sample return prediction R² for RSSM.

This measures what actually matters: can the world model predict real
next-day returns on data it hasn't seen?

Evaluates:
  1. Prediction R² on test set (2022-2024)
  2. Directional accuracy (% of sign matches)
  3. Prediction vs naive baseline (predicting mean)
  4. Regime-level breakdown (bull/bear/sideways performance)

Usage:
    python evaluate_real.py --checkpoint checkpoints/phase5_real.pt --data data/SPY_fused.csv
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import MarketEncoder, RSSM, RSSMRewardDecoder


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of times pred and true have the same sign."""
    return (np.sign(y_true) == np.sign(y_pred)).mean()


def sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio."""
    return returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained RSSM world model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    encoder = MarketEncoder(3, 2, config.get("embed_dim", 128)).to(device)
    rssm = RSSM(config.get("embed_dim", 128), 7,
                config.get("hidden_dim", 128),
                config.get("latent_dim", 32)).to(device)
    reward_dec = RSSMRewardDecoder(config.get("hidden_dim", 128),
                                    config.get("latent_dim", 32)).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    rssm.load_state_dict(ckpt["rssm_state"])
    reward_dec.load_state_dict(ckpt["reward_decoder_state"])
    encoder.eval()
    rssm.eval()
    reward_dec.eval()
    return encoder, rssm, reward_dec


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(
    encoder: MarketEncoder,
    rssm: RSSM,
    reward_dec: RSSMRewardDecoder,
    df: pd.DataFrame,
    lookback: int = 60,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Walk-forward evaluation: at each step, encode history, run RSSM one step,
    predict next-day return, compare to actual. No look-ahead.
    """
    # Column definitions
    tech_cols = ["Open", "Close", "Volume"]
    fund_cols = ["ROE", "Debt_Ratio"]
    action_cols = ["US10Y", "Yield_Spread", "VIX", "VIX_1w_Change",
                   "US10Y_Volatility", "is_earnings_day", "Earnings_Surprise"]

    tech = torch.tensor(df[tech_cols].values, dtype=torch.float32)
    fund = torch.tensor(df[fund_cols].values, dtype=torch.float32)
    actions = torch.tensor(df[action_cols].values, dtype=torch.float32)
    true_returns = df["Next_Day_Return"].values

    n = len(df)
    predictions = []
    actuals = []

    h_t, z_t = rssm.initial_state(1, device)

    for t in range(lookback, n - 1):
        # Encode current observation
        tech_window = tech[t - lookback:t].unsqueeze(0).to(device)  # [1, L, 3]
        fund_now = fund[t].unsqueeze(0).to(device)                   # [1, 2]
        e_t = encoder(tech_window, fund_now)

        # RSSM step
        a_prev = actions[t].unsqueeze(0).to(device)  # [1, 7]
        out = rssm.observe_step(h_t, z_t, a_prev, e_t)
        h_t, z_t = out["h_t"], out["z_t"]

        # Predict return
        pred = reward_dec(h_t, z_t).item()
        predictions.append(pred)
        actuals.append(true_returns[t])

        if (t - lookback) % 500 == 0 and t > lookback:
            print(f"  Progress: {t - lookback}/{n - lookback} steps")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Metrics
    r2 = r2_score(actuals, predictions)
    dir_acc = directional_accuracy(actuals, predictions)
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    actual_mean = actuals.mean()
    actual_std = actuals.std()

    # Baseline: always predict the training mean
    baseline_pred = np.full_like(actuals, actuals.mean())
    baseline_r2 = r2_score(actuals, baseline_pred)

    # Regime breakdown
    regimes = {
        "all": slice(None),
        "bull (S&P up > 10% yr)": actuals > actuals.mean() + actuals.std(),
        "bear (S&P down)": actuals < actuals.mean() - actuals.std(),
        "normal": (actuals >= actuals.mean() - actuals.std()) &
                  (actuals <= actuals.mean() + actuals.std()),
    }

    regime_metrics = {}
    for name, mask in regimes.items():
        if mask is slice(None):
            sub_actuals = actuals
            sub_preds = predictions
        else:
            sub_actuals = actuals[mask]
            sub_preds = predictions[mask]
        if len(sub_actuals) < 5:
            continue
        regime_metrics[name] = {
            "n": len(sub_actuals),
            "r2": r2_score(sub_actuals, sub_preds),
            "dir_acc": directional_accuracy(sub_actuals, sub_preds),
            "actual_mean": sub_actuals.mean(),
            "pred_mean": sub_preds.mean(),
        }

    # Trading simulation: if pred > 0, go long; else flat
    sim_returns = np.where(predictions > 0, actuals, 0.0)
    sim_sharpe = sharpe(sim_returns)
    buyhold_sharpe = sharpe(actuals)

    return {
        "r2": r2,
        "baseline_r2": baseline_r2,
        "directional_accuracy": dir_acc,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "actual_mean": actual_mean,
        "actual_std": actual_std,
        "n_samples": len(actuals),
        "regimes": regime_metrics,
        "sim_sharpe": sim_sharpe,
        "buyhold_sharpe": buyhold_sharpe,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Honest RSSM evaluation on real data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained RSSM checkpoint")
    parser.add_argument("--data", type=str, default="data/SPY_fused.csv",
                        help="Path to fused stock+macro CSV")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load data
    data_path = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run: python data_real.py --ticker SPY")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Data: {len(df)} rows, {list(df.columns)}")

    # Split
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]
    print(f"Train: {len(train_df)} days ({train_df.index[0].date()} — {train_df.index[-1].date()})")
    print(f"Test:  {len(test_df)} days ({test_df.index[0].date()} — {test_df.index[-1].date()})")

    # Real market stats
    print(f"\nReal market stats (test set):")
    print(f"  Mean daily return: {test_df['Next_Day_Return'].mean():.6f} "
          f"({test_df['Next_Day_Return'].mean()*252:.2%} ann)")
    print(f"  Std daily return:  {test_df['Next_Day_Return'].std():.6f} "
          f"({test_df['Next_Day_Return'].std()*np.sqrt(252):.2%} ann)")
    print(f"  Sharpe:            {sharpe(test_df['Next_Day_Return'].values):.4f}")
    print(f"  Buy & Hold 2022-24: {((test_df['Close'].iloc[-1]/test_df['Close'].iloc[0]) - 1)*100:.1f}%")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    encoder, rssm, reward_dec = load_model(args.checkpoint, device)

    # Evaluate on TEST set (never seen during training)
    print(f"\n{'='*60}")
    print("EVALUATION: Out-of-sample return prediction")
    print(f"{'='*60}")
    print("Walking forward, encoding each day, predicting next-day return...")
    results = evaluate(encoder, rssm, reward_dec, test_df,
                        lookback=args.lookback, device=device)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {results['n_samples']}")
    print()
    print(f"  Prediction R²:        {results['r2']:+.6f}")
    print(f"  Baseline R² (mean):   {results['baseline_r2']:+.6f}")
    print(f"  Improvement over baseline: {results['r2'] - results['baseline_r2']:+.6f}")
    print(f"  Directional accuracy:  {results['directional_accuracy']:.4f} "
          f"({'above' if results['directional_accuracy'] > 0.5 else 'below'} 50%)")
    print()
    print(f"  Actual mean return:  {results['actual_mean']:.6f}")
    print(f"  Predicted mean:      {results['pred_mean']:.6f}")
    print(f"  Actual std:          {results['actual_std']:.6f}")
    print(f"  Predicted std:       {results['pred_std']:.6f}")
    print()
    print(f"  Simulated Sharpe (pred>0 → long): {results['sim_sharpe']:.4f}")
    print(f"  Buy & Hold Sharpe:                 {results['buyhold_sharpe']:.4f}")

    # Regime breakdown
    print(f"\n{'='*60}")
    print("REGIME BREAKDOWN")
    print(f"{'='*60}")
    for name, m in results["regimes"].items():
        print(f"  {name:<30}: n={m['n']:4d}  R²={m['r2']:+.6f}  "
              f"DirAcc={m['dir_acc']:.3f}  "
              f"Actual μ={m['actual_mean']:+.6f}  Pred μ={m['pred_mean']:+.6f}")

    # Honest verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if results["r2"] > 0.001:
        print("  ✓ Positive R² — model captures real predictive signal")
    elif results["r2"] > 0.0:
        print("  ~ Marginally positive R² — weak but real signal")
    elif results["r2"] > -0.001:
        print("  ≈ Zero R² — model extracts no signal beyond mean")
    else:
        print("  ✗ Negative R² — model is worse than just predicting the mean")

    if results["r2"] > results["baseline_r2"]:
        print(f"  ✓ Beats naive baseline by {results['r2'] - results['baseline_r2']:+.6f}")
    else:
        print(f"  ✗ Worse than naive baseline by {results['r2'] - results['baseline_r2']:+.6f}")

    if results["directional_accuracy"] > 0.51:
        print(f"  ✓ Directional accuracy {results['directional_accuracy']:.3f} > 51%")
    elif results["directional_accuracy"] > 0.50:
        print(f"  ~ Directional accuracy at {results['directional_accuracy']:.3f}")
    else:
        print(f"  ✗ Directional accuracy {results['directional_accuracy']:.3f} — worse than coin flip")


if __name__ == "__main__":
    main()
