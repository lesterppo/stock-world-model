# Stock World Model

A deep learning system that learns market dynamics in latent space using a DreamerV2-style Recurrent State Space Model (RSSM).

Built in PyTorch. Trains a world model that compresses market data into a 32-dimensional stochastic latent state, then uses it for regime detection and risk management.

## Architecture

```
Raw Market Data (OHLCV + Fundamentals + Macro)
        │
        ▼
   MarketEncoder (GRU + MLP two-stream fusion)
        │
        ▼  e_t (128-dim embedding)
   RSSM (Recurrent State Space Model)
   ├─ h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])   ← deterministic state
   └─ z_t ~ q(z_t | h_t, e_t)                   ← stochastic state (32-dim)
        │
        ├─→ Reward Decoder: predicts target from (h_t, z_t)
        ├─→ PPO Controller: outputs position ∈ [-1,1]
        └─→ Meta-Controller: regime detection + risk management
```

### Key features

- **RSSM (DreamerV2/V3 style)**: Separates deterministic and stochastic states to prevent posterior collapse
- **KL annealing + free bits**: Gradual information bottleneck prevents the GRU from finding deterministic shortcuts
- **Latent imagination**: The RSSM can generate realistic market scenarios without real data
- **Regime detection**: K-Means clustering on latent states discovers market regimes (bull/bear/crisis)
- **Epistemic uncertainty**: Variance in stochastic states flags anomalous market conditions
- **Contrastive learning**: InfoNCE head for cross-regime generalization

## Project structure

```
stock-world-model/
├── model.py              # Core architectures: MarketEncoder, RSSM, EnsembleRSSM, ContrastiveHead
├── controller.py         # PPO Actor-Critic (Gaussian policy, tanh-squashed)
├── losses.py             # KL divergence, free bits, KL annealer, InfoNCE, RiskAdjustedLoss
├── data.py               # Mock data generators (multi-regime, basic)
├── data_real.py          # Real data pipeline (Yahoo Finance + FRED macro, PIT-aligned)
├── dream_env.py          # Latent imagination environment + GAE
├── train_phase1.py       # Phase 1: old variational GRU training (archived)
├── train_phase2.py       # Phase 2: PPO latent imagination RL (archived)
├── train_phase3.py       # Phase 3: stress testing engine (archived)
├── train_phase4.py       # Phase 4: contrastive world model (archived)
├── train_phase5.py       # Phase 5: RSSM training with KL annealing
├── train_real.py         # RSSM training on real stock data (optimized for CPU)
├── train_vol.py          # RSSM training for volatility prediction
├── evaluate_real.py      # Honest out-of-sample evaluation (R², directional accuracy)
├── metactl.py            # Meta-Controller: latent regime detection + circuit breaker
└── requirements.txt      # torch, pandas, numpy, yfinance, scikit-learn
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --break-system-packages
```

### 2. Download real data

```bash
python data_real.py --ticker SPY
```

This downloads S&P 500 ETF data (2010-2024) from Yahoo Finance and fuses it with macro indicators.
FRED data is used when available; falls back to historically-realistic synthetic macro if geo-blocked.

### 3. Train the world model

```bash
python train_real.py --ticker SPY --epochs 15
```

Trains the RSSM on SPY data (2010-2021). Saves checkpoint to `checkpoints/SPY_rssm.pt`.

### 4. Evaluate honestly

```bash
python evaluate_real.py --checkpoint checkpoints/SPY_rssm.pt --data data/SPY_fused.csv
```

Walks forward through the test set (2022-2024) and measures R² of return predictions.

### 5. Run the Meta-Controller

```bash
python metactl.py --ticker SPY --checkpoint checkpoints/SPY_rssm.pt --n-clusters 6
```

Extracts latent states, clusters them into market regimes, profiles each regime, and backtests
a rule-based meta-controller with circuit breaker (cash on high-uncertainty days).

## Results (SPY, 2022-2024 test set)

| Capability | Metric | Status |
|---|---|---|
| Return prediction | R² = -1.19 | Daily returns are a random walk |
| Volatility prediction | R² = -1.64 (improving) | Marginal with daily data |
| Regime detection | 6 meaningful clusters discovered | Working |
| Drawdown protection | -13% vs B&H -21% | Working |
| Sharpe improvement | -0.23 vs B&H | Risk tool, not return enhancer |

## Design philosophy

The world model learns to compress market data into a structured latent space.
It does NOT try to predict daily returns (R² is inevitably negative for single assets).
Instead, it captures market structure: regimes, risk levels, and uncertainty.
This makes it useful as a risk management overlay and meta-controller for simpler strategies.

## License

MIT
