#!/usr/bin/env python3
"""
Volatility Prediction — train RSSM to predict realized volatility (not returns).

Uses Parkinson volatility estimator from High/Low data.
Target: 5-day forward realized volatility (more predictable than 1-day).

Usage:
    python train_vol.py --ticker SPY --epochs 15
    Then evaluation is printed inline.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM, RSSMRewardDecoder
from losses import KLAnnealer


class VolSeqDataset(Dataset):
    def __init__(self, df, seq_len=10, lookback=60, stride=10):
        self.seq_len = seq_len; self.lookback = lookback
        self.tech = torch.tensor(df[['Open','Close','Volume']].values, dtype=torch.float32)
        self.fund = torch.tensor(df[['ROE','Debt_Ratio']].values, dtype=torch.float32)
        self.acts = torch.tensor(df[['US10Y','Yield_Spread','VIX','VIX_1w_Change',
                                     'US10Y_Volatility','is_earnings_day',
                                     'Earnings_Surprise']].values, dtype=torch.float32)
        self.target = torch.tensor(df['Target_Vol_5d'].values, dtype=torch.float32)
        n = len(df)
        self.starts = list(range(0, n - lookback - seq_len - 5, stride))  # -5 for forward window

    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        s = self.starts[idx]; L = self.lookback; S = self.seq_len
        return (
            torch.stack([self.tech[s+t:s+t+L] for t in range(S)]),
            torch.stack([self.fund[s+t+L-1] for t in range(S)]),
            self.acts[s+L:s+L+S],
            self.target[s+L:s+L+S],
        )


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--lookback", type=int, default=60)
    args = p.parse_args()

    device = torch.device("cpu")

    # Load
    df = pd.read_csv(f"data/{args.ticker}_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    test_df = df['2022-01-01':]

    # Volatility stats
    print(f"Train vol: mean={train_df['Realized_Vol'].mean():.4f}, std={train_df['Realized_Vol'].std():.4f}")
    print(f"Test vol:  mean={test_df['Realized_Vol'].mean():.4f}, std={test_df['Realized_Vol'].std():.4f}")
    print(f"Train 5d vol: mean={train_df['Target_Vol_5d'].mean():.4f}, std={train_df['Target_Vol_5d'].std():.4f}")

    # Dataset
    ds = VolSeqDataset(train_df, seq_len=args.seq_len, lookback=args.lookback, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Train sequences: {len(ds)}")

    # Models (volatility decoder instead of return decoder)
    encoder = MarketEncoder(3, 2, 128).to(device)
    rssm = RSSM(128, 7, 128, 32).to(device)
    vol_decoder = RSSMRewardDecoder(128, 32).to(device)  # same architecture, different target
    params = list(encoder.parameters()) + list(rssm.parameters()) + list(vol_decoder.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    annealer = KLAnnealer(anneal_steps=2000, free_bits=0.1)

    # Loss: MSE on log volatility (log-normal assumption)
    loss_fn = torch.nn.MSELoss()
    # Use proper KL loss from losses.py with free bits
    from losses import rssm_kl_loss

    print(f"\nTraining volatility predictor ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        encoder.train(); rssm.train(); vol_decoder.train()
        total_kl = total_mse = 0.0
        for tech, fund, acts, target in loader:
            B, S, L, Td = tech.shape
            tech = tech.permute(1,0,2,3).to(device)
            fund = fund.permute(1,0,2).to(device)
            acts = acts.permute(1,0,2).to(device)
            target = target.permute(1,0).to(device).clamp(min=1e-6)  # vol > 0

            e_seq = torch.stack([encoder(tech[t], fund[t]) for t in range(S)])
            out = rssm.observe_rollout(e_seq, acts)
            h_f = out['h'].reshape(-1,128); z_f = out['z'].reshape(-1,32)
            pred = vol_decoder(h_f, z_f).squeeze(-1).view(S, B)

            # Proper KL loss with free bits
            kl_weight = annealer()
            kl = rssm_kl_loss(
                out["post_mu"], out["post_logvar"],
                out["prior_mu"], out["prior_logvar"],
                free_bits=annealer.free_bits,
            )

            mse = loss_fn(pred, target)
            loss = kl_weight * kl + mse

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            total_kl += kl.item(); total_mse += mse.item()

        n_b = len(loader)
        print(f"Epoch {epoch+1:2d}/{args.epochs} | KL={total_kl/n_b:.4f} (w={kl_weight:.3f}) | Vol MSE={total_mse/n_b:.6f}")

    # ── Evaluation on test set ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION: Volatility Prediction (Test Set 2022-2024)")
    print(f"{'='*60}")

    encoder.eval(); rssm.eval(); vol_decoder.eval()
    preds = []
    actuals = []

    tech_all = torch.tensor(test_df[['Open','Close','Volume']].values, dtype=torch.float32)
    fund_all = torch.tensor(test_df[['ROE','Debt_Ratio']].values, dtype=torch.float32)
    acts_all = torch.tensor(test_df[['US10Y','Yield_Spread','VIX','VIX_1w_Change',
                                     'US10Y_Volatility','is_earnings_day',
                                     'Earnings_Surprise']].values, dtype=torch.float32)
    target_all = test_df['Target_Vol_5d'].values

    h_t, z_t = rssm.initial_state(1, device)
    n_test = len(test_df)
    with torch.no_grad():
        for t in range(args.lookback, n_test - 5):
            tw = tech_all[t-args.lookback:t].unsqueeze(0).to(device)
            fw = fund_all[t].unsqueeze(0).to(device)
            e_t = encoder(tw, fw)
            a_prev = acts_all[t].unsqueeze(0).to(device)
            out = rssm.observe_step(h_t, z_t, a_prev, e_t)
            h_t, z_t = out['h_t'], out['z_t']
            pred = vol_decoder(h_t, z_t).item()
            preds.append(pred)
            actuals.append(target_all[t])

    preds = np.array(preds); actuals = np.array(actuals)
    # Remove NaN (last 5 days have no 5d forward vol)
    mask = ~np.isnan(actuals)
    preds = preds[mask]; actuals = actuals[mask]

    vol_r2 = r2_score(actuals, preds)
    baseline_pred = np.full_like(actuals, actuals.mean())
    baseline_r2 = r2_score(actuals, baseline_pred)
    corr = np.corrcoef(actuals, preds)[0,1] if len(actuals) > 2 else 0

    print(f"Samples: {len(actuals)}")
    print(f"Actual vol mean: {actuals.mean():.4f}, std: {actuals.std():.4f}")
    print(f"Pred vol mean:   {preds.mean():.4f}, std: {preds.std():.4f}")
    print(f"Correlation:     {corr:+.4f}")
    print(f"Volatility R²:   {vol_r2:+.6f}")
    print(f"Baseline R²:     {baseline_r2:+.6f}")
    print(f"Improvement:     {vol_r2 - baseline_r2:+.6f}")

    if vol_r2 > 0.05:
        print("\n✓ STRONG: R² > 0.05 — model captures meaningful volatility structure")
    elif vol_r2 > 0.01:
        print("\n✓ GOOD: R² > 0.01 — predictive signal confirmed")
    elif vol_r2 > 0.0:
        print("\n~ MARGINAL: Slightly positive R² — weak but real signal")
    else:
        print(f"\n✗ NEGATIVE: R² = {vol_r2:.4f} — still not capturing volatility structure")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        'encoder_state': encoder.state_dict(),
        'rssm_state': rssm.state_dict(),
        'vol_decoder_state': vol_decoder.state_dict(),
        'config': {'embed_dim':128,'hidden_dim':128,'latent_dim':32},
    }, f"checkpoints/{args.ticker}_vol.pt")
    print(f"\nSaved: checkpoints/{args.ticker}_vol.pt")


if __name__ == "__main__":
    main()
