# Stock World Model

**Four-phase journey from neural world models to Dual Momentum for multi-asset portfolio allocation.**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](paper/stock_world_model.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Key Finding

**Unlevered Dual Momentum achieves risk-adjusted parity with SPY buy-and-hold while reducing maximum drawdown by 40%.** A 130,000-parameter Recurrent State-Space Model (RSSM) adds zero value; simple, rule-based strategies dominate.

| Phase | Approach | Params | AnnRet | Calmar | MaxDD | Verdict |
|---|---|---|---|---|---|---|
| 1 | RSSM (return prediction) | 130K | — | N/A ($R^2 < 0$) | — | Collapsed |
| 2 | K-Means Regime Clustering | ~30 | — | 0.09 | — | Dead end |
| 3 | Vol Target + Trend Follow | ~10 | +4.9–7.1% | 0.14–0.21 | -34.8% | Underperforms |
| **4** | **Dual Momentum (unlevered)** | **~5** | **+8.1%** | **0.396** | **-20.3%** | **Winner** |
| — | SPY Buy & Hold | 0 | +13.6% | 0.404 | -33.7% | Benchmark |

All Calmar ratios use corrected annualized return (`CAGR / maxDD`), not cumulative return. A measurement error using cumulative returns—discovered and corrected during this project—inflates Calmar by 10–40× in multi-year backtests.

## Architecture

```
Phase 1 (Failed):      OHLCV Data → RSSM → 4 Model Collapses
Phase 2 (Failed):      RSSM States → PCA → K-Means → Regime Clusters (collapse OOS)
Phase 3 (Underwhelms): Macro Features → Vol Target + Trend → Chronic Underperformance
Phase 4 (Winner):      Price Data → 12-Month Momentum → Top-2 Assets → Monthly Rebalance

                        Dual Momentum Framework
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     SPY 12m Return > 0                SPY 12m Return ≤ 0
     (Risk-On)                         (Risk-Off)
              │                               │
              ▼                               ▼
     Top 2 of {SPY,GLD,DBC}           TLT if trending,
     50/50 equal weight               else 100% Cash
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (Yahoo Finance + FRED macro)
python data_real.py --ticker SPY
python etf_data.py --tickers SPY,TLT,GLD,DBC

# Run the winning strategy (Dual Momentum, monthly rebalance)
python dual_momentum.py

# Run vol targeting + trend following comparison
python vol_allocator.py

# Run levered Dual Momentum (leverage trap demonstration)
python levered_dm.py

# Run K-Means macro allocator (documented failure)
python macro_allocator.py

# Comprehensive walk-forward comparison (all strategies)
python verify_final.py
```

## Project Journey

**14 phases** of iterative development, guided by 7 rounds of AI peer review (Gemini Pro):

1. **Phase 1–4:** RSSM architecture + 4 prediction tasks → all collapsed to degenerate solutions
2. **Phase 5:** RSSM training with KL annealing + free bits → healthy bottleneck but no predictive power
3. **Phase 6:** Discovered posterior collapse (KL near zero = RNN bypasses stochastic state)
4. **Phase 7–8:** Regime clustering on RSSM states → in-sample interpretable, out-of-sample collapse
5. **Phase 9–10:** Walk-forward validation, ablation study, Calmar ratio correction
6. **Phase 11:** K-Means soft-blending → degenerate to equal-weight
7. **Phase 12:** Vol targeting + trend following + combined approach
8. **Phase 13:** Dual Momentum (unlevered) → **risk-adjusted parity with SPY**
9. **Phase 14:** Levered Dual Momentum → leverage trap documented, paper published

## Files

| File | Purpose |
|---|---|
| `dual_momentum.py` | **Winner**: unlevered Dual Momentum, monthly rebalance, 12-month lookback |
| `levered_dm.py` | Tiered-leverage DM (0x/1.0x/1.5x) — leverage trap demonstration |
| `vol_allocator.py` | Vol targeting, trend following, combined Trend+Vol with corrected Calmar |
| `verify_final.py` | Comprehensive walk-forward comparison (all strategies vs benchmarks) |
| `macro_allocator.py` | K-Means regime allocator with soft distance-weighted blending |
| `ablation_study.py` | RSSM vs raw macro vs momentum head-to-head comparison |
| `walk_forward_validate.py` | 9-year rolling walk-forward validation framework |
| `model.py` | DreamerV2/V3 RSSM architecture (MarketEncoder, RSSM, EnsembleRSSM) |
| `losses.py` | KL annealing, InfoNCE contrastive loss, RSSM training losses |
| `allocator.py` | Production K-Means allocator with transaction costs and OOD guard |
| `cross_sectional_allocator.py` | RSSM-based multi-asset regime allocation |
| `etf_data.py` | Multi-ETF data pipeline (Yahoo Finance) |
| `data_real.py` | SPY data pipeline with PIT-aligned fundamental data |
| `paper/paper.md` | Full paper source (Markdown + LaTeX) |
| `paper/stock_world_model.pdf` | Published PDF (16 pages, 20 references) |
| `paper/refs.bib` | BibTeX bibliography |

## Why Not RSSM?

RSSMs (DreamerV2/V3) are designed for Atari and MuJoCo—fully observable, stationary environments compressing 100K pixels into a 1024-dim latent state. Financial markets are the opposite: partially observed, non-stationary, and already compressed (12 input features). Expanding 12 inputs into 128 dimensions and 130K parameters on 2,902 training days with near-zero signal-to-noise ratio guarantees overfitting to noise. The models that "collapsed" were performing optimal inference in a noise-dominated environment.

## Why Dual Momentum?

Dual Momentum (Antonacci, 2014) combines relative momentum (cross-asset comparison) with absolute momentum (trend filter). It has 200+ years of out-of-sample evidence across asset classes and geographies. Our implementation:

- **Lookback**: 12-month trailing return
- **Rebalancing**: Monthly (last trading day)
- **Risk-On**: Top 2 of SPY, GLD, DBC at 50/50
- **Risk-Off**: TLT if trending, else Cash
- **Transaction costs**: 5bp one-way
- **SPY allocation**: 86% of days, mean 41%

## The Leverage Trap

Applying tiered margin leverage (1.5× on SPY during strong trends, 5.5% annual margin rate) adds only +0.6% annual return while degrading Sharpe (0.733→0.709) and Calmar (0.396→0.383). Total margin costs consumed 12.1% of final portfolio value. The 12-month lookback's inherent latency means the system remains levered during the first month of a crash. Leverage in momentum creates negative convexity.

## Citation

```bibtex
@article{ppo2026limits,
  title={The Limits of Algorithmic Complexity in Multi-Asset Allocation:
         A Journey from Recurrent State-Space Models to Dual Momentum},
  author={PPO, Lester},
  year={2026},
  note={Available at \url{https://github.com/lesterppo/stock-world-model}}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
