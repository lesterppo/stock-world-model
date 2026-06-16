# Stock World Model

**Regime-based multi-asset portfolio allocation using macro-economic features.**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](paper/stock_world_model.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Key Finding

**6 simple macro features outperform a 130,000-parameter deep recurrent state-space model for regime-based portfolio allocation.**

| Approach | Parameters | Mean Calmar | Mean Sharpe |
|---|---|---|---|
| Raw Macro (6-dim) | ~30 | **2.05** | **1.19** |
| RSSM + PCA (20-dim) | ~130,000 | 1.83 | 1.00 |
| Momentum Rotation | 0 | 1.32 | 0.75 |

9-year walk-forward validation (2016–2024), 4 assets (SPY/TLT/GLD/DBC), with transaction costs, OOD guard, and ensemble K-Means.

## Architecture

```
Raw Macro Features (VIX, spread, momentum, US10Y)
         │
         ▼
    K-Means Clustering (K=6)
         │
         ▼
   Per-Regime Weight Optimization (Calmar-maximizing, Bayesian shrinkage)
         │
         ▼
   Multi-Asset Portfolio Allocation (SPY/TLT/GLD/DBC)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (Yahoo Finance + FRED macro)
python data_real.py --ticker SPY

# Download multi-ETF data
python etf_data.py --tickers SPY,TLT,GLD,DBC

# Run production macro allocator (walk-forward 2016-2024)
python macro_allocator.py

# Run ablation study (RSSM vs macro vs momentum)
python ablation_study.py
```

## Project Status

- **12 phases** of iterative development
- **4 failed prediction approaches** documented (return/vol/bandit/classifier)
- **1 successful framework**: macro regime allocation
- **Paper**: 17 pages, peer-reviewed by Gemini Pro, ready for JFDS submission
- **Code**: ~5,000 lines Python, 22 files, MIT licensed

## Files

| File | Purpose |
|---|---|
| `macro_allocator.py` | **Production** macro regime allocator with TC, OOD, ensemble |
| `ablation_study.py` | Head-to-head RSSM vs macro comparison |
| `model.py` | DreamerV2/V3 RSSM architecture |
| `losses.py` | KL annealing, RSSM training losses |
| `walk_forward_validate.py` | 9-year rolling walk-forward framework |
| `cross_sectional_allocator.py` | RSSM-based multi-asset regime allocation |
| `allocator.py` | Production K-Means allocator with all guards |
| `etf_data.py` | Multi-ETF data pipeline (Yahoo Finance) |
| `data_real.py` | SPY data pipeline with PIT-aligned fundamentals |

## Citation

```bibtex
@article{ppo2026regime,
  title={The Efficiency of Parsimonious Feature Sets in Financial Regime Detection},
  author={PPO, Lester},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
