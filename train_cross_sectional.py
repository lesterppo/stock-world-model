#!/usr/bin/env python3
"""
Phase 6b: Train Cross-Sectional Relative Strength Ranker

Predicts which ETF (SPY, TLT, GLD) will have the highest return tomorrow.
Classification problem — fundamentally easier than regression on returns.

Usage:
    python train_cross_sectional.py --tickers SPY,TLT,GLD --epochs 60
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional import (
    CrossSectionalRanker,
    extract_multi_rssm_states,
    train_ranker,
    backtest_ranker,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,TLT,GLD")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    tag = "_".join(tickers)

    # ── Load data ───────────────────────────────────────────────────────────
    multi_path = f"data/multi_{tag}_fused.csv"
    df = pd.read_csv(multi_path, index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]
    print(f"Multi-ETF data: {len(df)} days")
    print(f"Train: {len(train_df)} ({train_df.index[0].date()} — {train_df.index[-1].date()})")
    print(f"Test:  {len(test_df)} ({test_df.index[0].date()} — {test_df.index[-1].date()})")

    # ── Checkpoint mapping ──────────────────────────────────────────────────
    checkpoint_paths = {}
    for ticker in tickers:
        ckpt = f"checkpoints/{ticker}_rssm.pt"
        if Path(ckpt).exists():
            checkpoint_paths[ticker] = ckpt
            print(f"  {ticker}: using {ckpt}")
        else:
            # Need to train RSSM for this ETF first
            print(f"  {ticker}: no RSSM checkpoint — need to train first")

    # If we only have SPY checkpoint, train RSSM for TLT and GLD
    if len(checkpoint_paths) < len(tickers):
        print("\nTraining RSSM for missing ETFs...")
        from train_real import StridedSeqDataset, main as train_real_main
        import subprocess

        for ticker in tickers:
            if ticker not in checkpoint_paths:
                # Need a single-ETF fused dataset for this ticker
                # Quick approach: extract from multi dataset
                single_cols = ['Open', 'Close', 'Volume', 'ROE', 'Debt_Ratio',
                              'US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                              'US10Y_Volatility', 'is_earnings_day', 'Earnings_Surprise',
                              'Next_Day_Return']
                single_df = pd.DataFrame(index=df.index)
                single_df['Open'] = df[f'{ticker}_Open']
                single_df['Close'] = df[f'{ticker}_Close']
                single_df['Volume'] = df[f'{ticker}_Volume']
                single_df['High'] = df[f'{ticker}_Close'] * 1.01  # approximate
                single_df['Low'] = df[f'{ticker}_Close'] * 0.99   # approximate
                single_df['ROE'] = df[f'{ticker}_ROE']
                single_df['Debt_Ratio'] = df[f'{ticker}_Debt_Ratio']
                for col in ['US10Y', 'Yield_Spread', 'VIX', 'VIX_1w_Change',
                            'US10Y_Volatility']:
                    if col in df.columns:
                        single_df[col] = df[col]
                single_df['is_earnings_day'] = 0
                single_df['Earnings_Surprise'] = 0.0
                single_df['Next_Day_Return'] = df[f'{ticker}_Next_Return']
                single_df = single_df.dropna()

                out_path = f"data/{ticker}_fused.csv"
                single_df.to_csv(out_path)
                print(f"  Created {out_path}: {len(single_df)} rows")

                # Train RSSM
                print(f"  Training RSSM for {ticker}...")
                # Use subprocess to avoid import issues
                result = subprocess.run([
                    "python", "train_real.py",
                    "--ticker", ticker,
                    "--epochs", "10",
                    "--stride", "10",
                    "--batch-size", "32",
                ], cwd=str(Path(__file__).resolve().parent),
                   capture_output=True, text=True)
                print(f"    {ticker} training: {result.stdout.split(chr(10))[-3:]}")
                if result.returncode != 0:
                    print(f"    WARNING: {ticker} RSSM training failed, using SPY checkpoint as fallback")
                else:
                    checkpoint_paths[ticker] = f"checkpoints/{ticker}_rssm.pt"

    # Ensure all ETFs have a checkpoint (fallback to SPY)
    for ticker in tickers:
        if ticker not in checkpoint_paths:
            checkpoint_paths[ticker] = "checkpoints/SPY_rssm.pt"
            print(f"  {ticker}: falling back to SPY RSSM")

    # ── Extract multi-ETF RSSM states ───────────────────────────────────────
    print("\nExtracting multi-asset RSSM states...")
    h_train, macro_train, labels_train, dates_train = extract_multi_rssm_states(
        checkpoint_paths, train_df, tickers, args.lookback, args.device
    )
    h_test, macro_test, labels_test, dates_test = extract_multi_rssm_states(
        checkpoint_paths, test_df, tickers, args.lookback, args.device
    )
    print(f"  Train states: {len(labels_train)}, Test states: {len(labels_test)}")

    # ── Label distribution ──────────────────────────────────────────────────
    print(f"\nTraining label distribution:")
    for i, ticker in enumerate(tickers):
        count = (labels_train == i).sum()
        print(f"  {ticker}: {count} days ({count/len(labels_train):.1%})")
    # Baseline accuracy: always pick most common
    baseline_acc = max((labels_train == i).mean() for i in range(len(tickers)))
    print(f"  Baseline (always pick most common): {baseline_acc:.3f}")

    # ── Train ranker ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Cross-Sectional Ranker — {args.epochs} epochs")
    print(f"{'='*60}")

    ranker = CrossSectionalRanker(
        n_assets=len(tickers),
        state_dim=128,
        hidden_dim=128,
        dropout=0.3,
    ).to(device)

    results = train_ranker(
        ranker, h_train, macro_train, labels_train, tickers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, val_split=0.15,
    )

    print(f"\nBest val accuracy: {results['best_val_acc']:.4f}")
    print(f"Improvement over baseline: {results['best_val_acc'] - baseline_acc:+.4f}")

    # ── Backtest on 2022-2024 ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BACKTEST: Cross-Sectional Ranking — SPY/TLT/GLD (2022-2024)")
    print(f"{'='*60}")

    bt = backtest_ranker(
        ranker, h_test, macro_test, test_df, tickers,
        args.lookback, args.device,
    )

    print(f"\n  Hit Rate: {bt['hit_rate']:.3f} (random baseline: {1/len(tickers):.3f})")
    print(f"  Mean Confidence: {bt['confidences'].mean():.3f}")

    print(f"\n  Per-class hit rate (when ETF was actually best):")
    for ticker, acc in bt['per_class_acc'].items():
        print(f"    {ticker}: {acc:.3f}")

    print(f"\n  {'Strategy':<30} {'Cum Return':>12} {'Sharpe':>10} {'Max DD':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Ranking (top-1 ETF)':<30} {bt['rank_cum']:>+12.2%} {bt['rank_sharpe']:>10.3f} {bt['rank_dd']:>10.2%}")
    print(f"  {'Equal Weight':<30} {bt['eq_cum']:>+12.2%} {bt['eq_sharpe']:>10.3f} {bt['eq_dd']:>10.2%}")

    for ticker in tickers:
        cum, sh, dd = np.prod(1 + bt['ind_rets'][ticker]) - 1, \
                      bt['ind_rets'][ticker].mean() / (bt['ind_rets'][ticker].std() + 1e-8) * np.sqrt(252), \
                      float(np.min((np.cumprod(1 + bt['ind_rets'][ticker]) - np.maximum.accumulate(np.cumprod(1 + bt['ind_rets'][ticker]))) / np.maximum.accumulate(np.cumprod(1 + bt['ind_rets'][ticker]))))
        print(f"  {'B&H ' + ticker:<30} {cum:>+12.2%} {sh:>10.3f} {dd:>10.2%}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if bt['hit_rate'] > 1/len(tickers) + 0.05:
        print(f"  ✓ Hit rate {bt['hit_rate']:.3f} > random {1/len(tickers):.3f} — significant")
    elif bt['hit_rate'] > 1/len(tickers) + 0.01:
        print(f"  ~ Marginal improvement: {bt['hit_rate']:.3f} vs {1/len(tickers):.3f}")
    else:
        print(f"  ✗ Hit rate at random level ({bt['hit_rate']:.3f} ≈ {1/len(tickers):.3f})")

    sh_improve = bt['rank_sharpe'] - bt['eq_sharpe']
    if sh_improve > 0.1:
        print(f"  ✓ Ranking Sharpe {bt['rank_sharpe']:.3f} > Equal-weight {bt['eq_sharpe']:.3f}")
    elif sh_improve > 0:
        print(f"  ~ Marginal Sharpe improvement: +{sh_improve:+.3f}")
    else:
        print(f"  ✗ Ranking underperforms equal-weight in Sharpe ({bt['rank_sharpe']:.3f} vs {bt['eq_sharpe']:.3f})")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "ranker_state": ranker.state_dict(),
        "config": {"n_assets": len(tickers), "state_dim": 128},
        "test_results": {
            "hit_rate": float(bt['hit_rate']),
            "rank_sharpe": float(bt['rank_sharpe']),
            "eq_sharpe": float(bt['eq_sharpe']),
            "rank_cum": float(bt['rank_cum']),
            "eq_cum": float(bt['eq_cum']),
        },
    }, "checkpoints/ranker.pt")
    print(f"\nSaved: checkpoints/ranker.pt")


if __name__ == "__main__":
    main()
