"""
Phase 6b: Cross-Sectional Relative Strength Ranking

Key insight: predicting WHICH ETF will outperform is easier than predicting
absolute returns. Common factors (market beta, macro shocks) cancel out in
relative prediction.

Architecture:
  1. Extract RSSM h_t for each ETF independently (shared RSSM weights)
  2. Stack: [h_t(SPY), h_t(TLT), h_t(GLD)] → 384-dim joint state
  3. RankingHead: MLP → 3-class softmax → predicted best ETF
  4. Training: cross-entropy loss on next-day return ranking
  5. Backtest: allocate 100% to top-ranked ETF each day

This is a classification problem — R² doesn't apply. We measure hit rate
(% of days the model correctly picks the best performer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class CrossSectionalRanker(nn.Module):
    """
    Ranking model: joint RSSM states → ETF ranking.

    Input:  [h_t(ETF1), h_t(ETF2), h_t(ETF3)] concatenated → 384-dim
    Output: 3-class softmax → P(ETF_i is best tomorrow)
    """

    def __init__(
        self,
        n_assets: int = 3,
        state_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_assets = n_assets
        input_dim = n_assets * state_dim + 6  # +6 for macro context

        self.state_dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_assets),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Zero init final layer bias for uniform start
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(
        self,
        h_states: torch.Tensor,     # [B, n_assets * state_dim]
        macro_ctx: torch.Tensor,    # [B, 6]
    ) -> torch.Tensor:
        """
        Args:
            h_states:  [B, 384] — concatenated RSSM states for all ETFs
            macro_ctx: [B, 6]   — VIX, yield spread, etc.
        Returns:
            logits: [B, 3] — log-probabilities for each ETF being best
        """
        h_drop = self.state_dropout(h_states)
        x = torch.cat([h_drop, macro_ctx], dim=-1)
        return self.net(x)

    def predict_ranking(self, h_states, macro_ctx):
        """Returns ranking: best ETF index (0,1,2) and confidence."""
        logits = self.forward(h_states, macro_ctx)
        probs = F.softmax(logits, dim=-1)
        best = torch.argmax(logits, dim=-1)
        conf = probs.gather(1, best.unsqueeze(-1)).squeeze(-1)
        return best, conf, probs


def extract_multi_rssm_states(
    checkpoint_paths: dict,
    df,
    tickers: list,
    lookback: int = 60,
    device: str = "cpu",
):
    """
    Extract RSSM states for multiple ETFs, aligned on common dates.

    Args:
        checkpoint_paths: dict of {ticker: checkpoint_path}
        df: DataFrame with columns {ticker}_Open, {ticker}_Close, etc.
        tickers: list of ticker names

    Returns:
        h_dict:   {ticker: [N, 128] ndarray}
        macro:    [N, 6] ndarray (VIX, yield_spread, etc.)
        rankings: [N, 3] ndarray — return rank for each ETF (0=worst, 2=best)
        dates:    [N] index
    """
    from model import MarketEncoder, RSSM
    import torch

    h_dict = {}
    n = len(df)

    macro_cols = ["VIX", "Yield_Spread", "VIX_1w_Change", "US10Y_Volatility"]
    macro = np.zeros((n - lookback, 6), dtype=np.float32)

    for ticker in tickers:
        ckpt_path = checkpoint_paths.get(ticker, f"checkpoints/{ticker}_rssm.pt")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            # Use SPY checkpoint as fallback for TLT, GLD
            ckpt_path = checkpoint_paths.get("SPY", "checkpoints/SPY_rssm.pt")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        encoder = MarketEncoder(3, 2, cfg.get("embed_dim", 128)).to(device)
        rssm = RSSM(cfg.get("embed_dim", 128), 7,
                    cfg.get("hidden_dim", 128),
                    cfg.get("latent_dim", 32)).to(device)
        encoder.load_state_dict(ckpt["encoder_state"])
        rssm.load_state_dict(ckpt["rssm_state"])
        encoder.eval()
        rssm.eval()

        # Build features from multi-ETF columns
        tech_cols = [f"{ticker}_Open", f"{ticker}_Close", f"{ticker}_Volume"]
        fund_cols = [f"{ticker}_ROE", f"{ticker}_Debt_Ratio"]

        # For actions, use shared macro + synthetic micro
        action_data = np.zeros((n, 7), dtype=np.float32)
        for i, col in enumerate(["US10Y", "Yield_Spread", "VIX", "VIX_1w_Change",
                                  "US10Y_Volatility"]):
            if col in df.columns:
                action_data[:, i] = df[col].values.astype(np.float32)

        tech = torch.tensor(df[tech_cols].values, dtype=torch.float32)
        fund = torch.tensor(df[fund_cols].values, dtype=torch.float32)
        acts = torch.tensor(action_data, dtype=torch.float32)

        h_states = np.zeros((n - lookback, cfg.get("hidden_dim", 128)), dtype=np.float32)
        h_t, z_t = rssm.initial_state(1, torch.device(device))

        with torch.no_grad():
            for t in range(lookback, n):
                idx = t - lookback
                tw = tech[t - lookback:t].unsqueeze(0).to(device)
                fw = fund[t].unsqueeze(0).to(device)
                e_t = encoder(tw, fw)
                a_prev = acts[t].unsqueeze(0).to(device)
                out = rssm.observe_step(h_t, z_t, a_prev, e_t)
                h_t, z_t = out["h_t"], out["z_t"]
                h_states[idx] = h_t.cpu().numpy().squeeze(0)

        h_dict[ticker] = h_states

    # Build macro context
    for i, col in enumerate(macro_cols):
        if col in df.columns:
            macro[:, i] = df[col].values[lookback:].astype(np.float32)
    # Add 5d and 20d trailing returns for SPY
    spy_rets = df["SPY_Return"].values
    for t_idx in range(lookback, n):
        idx = t_idx - lookback
        if idx >= 5:
            macro[idx, 4] = np.mean(spy_rets[t_idx - 5:t_idx])
        if idx >= 20:
            macro[idx, 5] = np.mean(spy_rets[t_idx - 20:t_idx])

    # Build ranking labels
    rankings = np.zeros((n - lookback, len(tickers)), dtype=np.int64)
    for i, ticker in enumerate(tickers):
        col = f"{ticker}_Next_Return"
        if col in df.columns:
            next_rets = df[col].values[lookback:]
            rankings[:, i] = np.argsort(np.argsort(next_rets))  # rank within day
    # We want: which ETF has highest return (index of max)
    labels = np.argmax(rankings, axis=-1)  # [N] — 0, 1, or 2

    dates = df.index[lookback:]

    return h_dict, macro, labels, dates


def train_ranker(
    ranker: CrossSectionalRanker,
    h_dict: dict,
    macro: np.ndarray,
    labels: np.ndarray,
    tickers: list,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    val_split: float = 0.2,
) -> dict:
    """
    Train the cross-sectional ranking model.

    Returns:
        dict with training history and validation metrics.
    """
    n = len(labels)
    n_val = int(n * val_split)
    n_train = n - n_val

    # Build feature matrix: [B, n_assets * 128 + 6]
    features = np.concatenate(
        [h_dict[t] for t in tickers] + [macro],
        axis=-1
    )  # [N, 384 + 6]

    X_train = torch.tensor(features[:n_train], dtype=torch.float32)
    y_train = torch.tensor(labels[:n_train], dtype=torch.int64)
    X_val = torch.tensor(features[n_train:], dtype=torch.float32)
    y_val = torch.tensor(labels[n_train:], dtype=torch.int64)

    optimizer = torch.optim.AdamW(ranker.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_acc": [], "val_loss": []}
    best_val_acc = 0.0
    best_state = None

    ranker.to(device)

    for epoch in range(epochs):
        ranker.train()
        # Shuffle
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            h_batch = X_train[idx, :384].to(device)
            m_batch = X_train[idx, 384:].to(device)
            y_batch = y_train[idx].to(device)

            logits = ranker(h_batch, m_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validate
        ranker.eval()
        with torch.no_grad():
            val_logits = ranker(X_val[:, :384].to(device), X_val[:, 384:].to(device))
            val_loss = F.cross_entropy(val_logits, y_val.to(device)).item()
            val_pred = torch.argmax(val_logits, dim=-1).cpu()
            val_acc = (val_pred == y_val).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone().cpu() for k, v in ranker.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

    ranker.load_state_dict(best_state)
    ranker.to(device)

    return {"best_val_acc": best_val_acc, "history": history}


def backtest_ranker(
    ranker: CrossSectionalRanker,
    h_dict: dict,
    macro: np.ndarray,
    df_test,
    tickers: list,
    lookback: int = 60,
    device: str = "cpu",
) -> dict:
    """
    Walk-forward backtest of ranking-based allocation.

    Strategy: each day, predict best ETF, allocate 100% to it.
    Compare to equal-weight portfolio and individual ETF B&H.
    """
    n = len(h_dict[tickers[0]])
    features = np.concatenate(
        [h_dict[t] for t in tickers] + [macro],
        axis=-1
    )

    # Get returns for each ETF
    n_full = len(df_test)
    rets = np.zeros((n_full - lookback, len(tickers)))
    for i, ticker in enumerate(tickers):
        col = f"{ticker}_Next_Return"
        rets[:, i] = df_test[col].values[lookback:]

    # Rank-based allocation
    predictions = np.zeros(n, dtype=np.int64)
    confidences = np.zeros(n)
    daily_rets = np.zeros(n)

    for t in range(n):
        h_t = torch.tensor(features[t:t + 1, :384], dtype=torch.float32, device=device)
        m_t = torch.tensor(features[t:t + 1, 384:], dtype=torch.float32, device=device)

        with torch.no_grad():
            best, conf, _ = ranker.predict_ranking(h_t, m_t)

        predictions[t] = best.item()
        confidences[t] = conf.item()
        daily_rets[t] = rets[t, best.item()]

    # Equal-weight benchmark
    eq_rets = rets.mean(axis=1)

    # Individual B&H
    ind_rets = {}
    for i, ticker in enumerate(tickers):
        ind_rets[ticker] = rets[:, i]

    # Metrics
    def stats(r):
        cum = np.prod(1 + r) - 1
        sh = r.mean() / (r.std() + 1e-8) * np.sqrt(252)
        eq = np.cumprod(1 + r)
        peak = np.maximum.accumulate(eq)
        dd = float(np.min((eq - peak) / peak))
        return cum, sh, dd

    rank_cum, rank_sharpe, rank_dd = stats(daily_rets)
    eq_cum, eq_sharpe, eq_dd = stats(eq_rets)

    # Hit rate: % of days we correctly pick the best
    best_actual = np.argmax(rets, axis=1)
    hit_rate = (predictions == best_actual).mean()

    # Per-class accuracy
    from collections import Counter
    label_counts = Counter(best_actual)
    per_class_acc = {}
    for label in range(len(tickers)):
        mask = best_actual == label
        if mask.sum() > 0:
            per_class_acc[tickers[label]] = (predictions[mask] == label).mean()

    return {
        "rank_cum": rank_cum,
        "rank_sharpe": rank_sharpe,
        "rank_dd": rank_dd,
        "eq_cum": eq_cum,
        "eq_sharpe": eq_sharpe,
        "eq_dd": eq_dd,
        "hit_rate": hit_rate,
        "per_class_acc": per_class_acc,
        "ind_rets": ind_rets,
        "daily_rets": daily_rets,
        "eq_rets": eq_rets,
        "predictions": predictions,
        "confidences": confidences,
        "best_actual": best_actual,
    }
