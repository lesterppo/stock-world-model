#!/usr/bin/env python3
"""
Quick RSSM training on real SPY data. Optimized for CPU:
  - Strided sequences (skip every N days) to reduce dataset size
  - Fewer epochs, fewer batches
  - Saves checkpoint incrementally

Usage:
    python train_real.py --ticker SPY --epochs 15
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarketEncoder, RSSM, RSSMRewardDecoder
from losses import KLAnnealer, rssm_phase5_loss


class StridedSeqDataset(Dataset):
    """Sequence dataset with stride to reduce size for CPU training."""
    def __init__(self, df, seq_len=10, lookback=60, stride=10):
        self.seq_len = seq_len
        self.lookback = lookback
        self.tech = torch.tensor(df[['Open','Close','Volume']].values, dtype=torch.float32)
        self.fund = torch.tensor(df[['ROE','Debt_Ratio']].values, dtype=torch.float32)
        self.acts = torch.tensor(df[['US10Y','Yield_Spread','VIX','VIX_1w_Change',
                                     'US10Y_Volatility','is_earnings_day',
                                     'Earnings_Surprise']].values, dtype=torch.float32)
        self.rets = torch.tensor(df['Next_Day_Return'].values, dtype=torch.float32)
        n = len(df)
        self.starts = list(range(0, n - lookback - seq_len, stride))

    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        s = self.starts[idx]; L = self.lookback; S = self.seq_len
        return (
            torch.stack([self.tech[s+t:s+t+L] for t in range(S)]),
            torch.stack([self.fund[s+t+L-1] for t in range(S)]),
            self.acts[s+L:s+L+S],
            self.rets[s+L:s+L+S],
        )


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

    # Load data
    df = pd.read_csv(f"data/{args.ticker}_fused.csv", index_col=0, parse_dates=True)
    train_df = df[:'2021-12-31']
    print(f"Train data: {len(train_df)} days")

    ds = StridedSeqDataset(train_df, seq_len=args.seq_len, lookback=args.lookback, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Sequences: {len(ds)} (stride={args.stride}), Batches: {len(loader)}")

    # Models
    encoder = MarketEncoder(3, 2, 128).to(device)
    rssm = RSSM(128, 7, 128, 32).to(device)
    reward_dec = RSSMRewardDecoder(128, 32).to(device)
    params = list(encoder.parameters()) + list(rssm.parameters()) + list(reward_dec.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    annealer = KLAnnealer(anneal_steps=2000, free_bits=0.1)
    print(f"Params: {sum(p.numel() for p in params):,}")

    # Train
    for epoch in range(args.epochs):
        encoder.train(); rssm.train(); reward_dec.train()
        total_kl = total_r = 0.0
        for tech, fund, acts, rets in loader:
            B, S, L, Td = tech.shape
            tech = tech.permute(1,0,2,3).to(device)
            fund = fund.permute(1,0,2).to(device)
            acts = acts.permute(1,0,2).to(device)
            rets = rets.permute(1,0).to(device)

            e_seq = torch.stack([encoder(tech[t], fund[t]) for t in range(S)])
            out = rssm.observe_rollout(e_seq, acts)
            h_f = out['h'].reshape(-1,128); z_f = out['z'].reshape(-1,32)
            pred = reward_dec(h_f, z_f).view(S, B)
            loss, m = rssm_phase5_loss(out, pred, rets, annealer)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_kl += m['kl_loss']; total_r += m['reward_loss']

        n_b = len(loader)
        print(f"Epoch {epoch+1:2d}/{args.epochs} | KL={total_kl/n_b:.4f} (w={annealer():.3f}) | Reward MSE={total_r/n_b:.6f}")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        'encoder_state': encoder.state_dict(),
        'rssm_state': rssm.state_dict(),
        'reward_decoder_state': reward_dec.state_dict(),
        'config': {'embed_dim':128, 'hidden_dim':128, 'latent_dim':32},
    }, f"checkpoints/{args.ticker}_rssm.pt")
    print(f"\nSaved: checkpoints/{args.ticker}_rssm.pt")
    print(f"Final KL: {total_kl/n_b:.4f} — {'HEALTHY' if total_kl/n_b > 0.01 else 'WARNING: possible collapse'}")


if __name__ == "__main__":
    main()
