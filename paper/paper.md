---
title: "The Efficiency of Parsimonious Feature Sets in Financial Regime Detection"
subtitle: "Evidence that Simple Macro Factors Outperform Deep Latent State-Space Models for Multi-Asset Allocation"
author: "Lester PPO"
date: "June 2026"
abstract: |
  We investigate whether deep latent state-space models (Recurrent State-Space Models,
  DreamerV2/V3 architecture) provide additional value over simple macro-economic features
  for the task of financial regime detection and multi-asset portfolio allocation. Using
  14 years of daily data across four asset classes (equities, treasuries, gold, commodities),
  we conduct a rigorous 9-year walk-forward validation comparing three approaches:
  (1) RSSM-learned latent states reduced via PCA, (2) raw macro features (VIX, yield spread,
  momentum), and (3) a momentum rotation baseline. Our central finding is that the RSSM's
  20-dimensional learned representation achieves a mean Calmar ratio of 1.83, while simple
  6-dimensional macro features achieve 2.05—a 12% improvement with 99.95% fewer parameters.
  We further find that both approaches meaningfully outperform static 60/40 and risk parity
  benchmarks, but neither beats simple SPY buy-and-hold in absolute returns. The regime
  labels discovered by the clustering are interpretable and correspond to well-known
  economic narratives (COVID panic, Goldilocks, inflationary growth). We conclude that
  for daily-frequency macro regime allocation, domain-engineered feature sets dominate
  representation learning, and that the primary contribution of regime-switching models
  is tail-risk management rather than return enhancement.

  **Keywords:** regime detection, multi-asset allocation, recurrent state-space models,
  K-Means clustering, macro factors, walk-forward validation, Calmar ratio

header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{hyperref}
documentclass: article
fontsize: 11pt
linestretch: 1.15
geometry: margin=1in
---

# 1. Introduction

Financial markets exhibit distinct structural regimes—periods of high and low volatility,
trending and mean-reverting behavior, risk-on and risk-off environments. Identifying these
regimes and dynamically adjusting portfolio allocations accordingly is a central challenge
in quantitative asset management. The conventional approach relies on human-engineered
macro-economic indicators: the VIX volatility index, yield curve spreads, interest rate
levels, and momentum signals. Recent advances in deep learning, particularly recurrent
state-space models (RSSMs) from the Dreamer family [@hafner2019dream], promise to learn
rich latent representations of temporal dynamics directly from data, potentially capturing
non-linear interactions that human-engineered features miss.

This paper empirically tests whether such learned representations provide additional value
over simple macro features for the specific task of regime-based multi-asset portfolio
allocation. We construct a DreamerV2/V3 RSSM trained on 14 years of daily OHLCV data for
the S&P 500 ETF (SPY), extract its 128-dimensional deterministic hidden state $h_t$, and
cluster it into market regimes using K-Means. We then optimize per-regime portfolio weights
across four asset classes (SPY, TLT, GLD, DBC) via coordinate ascent on the Calmar ratio.
We compare this RSSM-based approach against an identical pipeline using only 6 raw macro
features—VIX level, VIX weekly change, yield spread (10Y-2Y), US 10-year yield, 21-day
momentum, and 63-day momentum—with no learned components.

Our central finding is unambiguous: the raw macro features outperform the RSSM-learned
representation across a 9-year walk-forward validation (2016–2024), achieving a mean Calmar
ratio of 2.05 versus 1.83, with 5/9 annual wins. The 130,000-parameter RSSM provides zero
additional benefit over 6 human-engineered features for this task.

# 2. Related Work

Regime-switching models in finance have a long history, from Hamilton's Markov-switching
model [@hamilton1989] to more recent machine learning approaches using hidden Markov models
[@nguyen2018] and clustering on latent representations [@zheng2020]. The Dreamer family of
world models [@hafner2019dream; @hafner2020dreamerv2; @hafner2023dreamerv3] introduced
recurrent state-space models that separate deterministic and stochastic latent states,
successfully applied to reinforcement learning in visual domains. Financial applications
of world models remain rare, with most work focusing on price prediction rather than
state representation learning.

Our contribution is an empirical comparison of learned versus engineered feature
representations for the specific task of regime-based portfolio allocation, with
rigorous walk-forward validation and transaction cost accounting. We find that the
efficiency of simple macro features echoes Occam's razor in quantitative finance.

# 3. Methodology

## 3.1 Data

We use daily data from January 2010 through December 2024 for four exchange-traded funds:
SPY (S&P 500), TLT (20+ Year Treasury Bonds), GLD (Gold), and DBC (Commodities Index).
Data is sourced from Yahoo Finance. Macro-economic indicators (VIX, yield curve) are
constructed from FRED data with a synthetic fallback for geo-blocked regions.

All features are point-in-time aligned: quarterly fundamental data (ROE, Debt Ratio) is
delayed by 45 calendar days to account for SEC filing delays, preventing lookahead bias.

## 3.2 RSSM Architecture

The Recurrent State-Space Model follows the DreamerV2/V3 architecture:

- **MarketEncoder:** A two-stream fusion of a 2-layer GRU over 60-day technical windows
  (Open, Close, Volume) and an MLP over fundamental features (ROE, Debt Ratio), producing
  a 128-dimensional embedding $e_t$.

- **RSSM Core:** A GRU cell updates the deterministic hidden state $h_t \in \mathbb{R}^{128}$
  from the previous stochastic state $z_{t-1}$ and action $a_{t-1}$ (macro features). A
  posterior network $q(z_t | h_t, e_t)$ produces the stochastic state $z_t \in \mathbb{R}^{32}$
  with a prior network $p(z_t | h_t)$ for imagination rollouts.

- **KL Annealing:** The KL divergence between posterior and prior is annealed from 0 to 1
  over 5,000 steps with 0.1 free bits per dimension, preventing posterior collapse.

The RSSM is trained on SPY data from 2010–2021 (2,962 days) using the Adam optimizer
with learning rate $3\times10^{-4}$ and gradient clipping at 1.0. Training converges
with a healthy KL divergence of approximately 3.2 nats.

## 3.3 Regime Detection and Allocation

Both the RSSM and macro approaches follow the same downstream pipeline:

1. **Feature extraction:** For RSSM, the 128-dimensional $h_t$ is extracted for each asset
   independently, concatenated into a 512-dimensional joint state, and reduced to 20
   dimensions via PCA (retaining 60% of variance). For the macro approach, 6 raw features
   are used directly.

2. **K-Means clustering:** Features are standardized and clustered into $K=6$ regimes
   using scikit-learn's K-Means with $n_\text{init}=10$.

3. **Weight optimization:** Per-regime portfolio weights across the 4 assets are optimized
   via coordinate ascent on the Calmar ratio (annualized return / maximum drawdown), with
   discrete weight choices of $\{0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0\}$ per asset,
   normalized to sum to 1.

4. **Walk-forward validation:** A rolling 5-year training window is used to fit the
   K-Means model and optimize weights; the subsequent year serves as the test set.
   This is repeated for years 2016–2024, producing 9 independent out-of-sample tests.

## 3.4 Production Safeguards

The production implementation includes several risk management features:

- **Transaction costs:** 5 basis points per trade plus 1 bp slippage
- **Velocity cap:** Maximum daily position change of 15% per asset
- **OOD guard:** Positions reduced to 30% when the current feature vector exceeds
  the 99th percentile Euclidean distance from training centroids
- **Ensemble:** K-Means models across 5 random seeds, weights averaged for stability

## 3.5 Benchmarks

We compare against four benchmarks: (1) Buy-and-hold SPY, (2) Risk parity (inverse
volatility weighting), (3) Static 60/40 SPY/TLT, and (4) Momentum rotation (equal-weight
top 2 assets by 63-day trailing return).

# 4. Results

## 4.1 Walk-Forward Validation

Table \ref{tab:walkforward} presents the 9-year walk-forward results for the production
macro allocator with transaction costs.

\begin{table}[H]
\centering
\caption{Macro Regime Allocator — 9-Year Walk-Forward (with 6bp transaction costs)}
\label{tab:walkforward}
\begin{tabular}{lrrrrr}
\toprule
Year & Calmar & Sharpe & Cum. Return & Max DD & Turnover \\
\midrule
2016 & +2.39 & +1.26 & +13.4\% & -5.6\% & 1.1\% \\
2017 & +3.46 & +2.17 & +13.9\% & -4.0\% & 0.1\% \\
2018 & -0.29 & -0.32 & -3.9\% & -13.3\% & 0.8\% \\
2019 & +4.91 & +2.02 & +15.7\% & -3.2\% & 1.9\% \\
2020 & +0.72 & +0.79 & +19.2\% & -26.8\% & 1.2\% \\
2021 & +0.29 & +0.29 & +2.4\% & -8.0\% & 8.7\% \\
2022 & +0.19 & +0.27 & +3.1\% & -16.4\% & 2.9\% \\
2023 & +2.58 & +1.49 & +15.8\% & -6.1\% & 1.6\% \\
2024 & +2.67 & +1.25 & +15.7\% & -5.9\% & 0.0\% \\
\midrule
Mean & +1.88 & +1.02 & +10.6\% & -10.0\% & 2.1\% \\
\bottomrule
\end{tabular}
\end{table}

The macro allocator achieves positive Calmar ratios in 7 of 9 years, with particularly
strong performance in low-volatility bull markets (2017, 2019, 2023–2024). The COVID year
(2020) is the worst performer, as the model's lagging macro features could not anticipate
the March 2020 crash.

## 4.2 Ablation Study: RSSM vs. Raw Macro

Table \ref{tab:ablation} presents the core ablation result comparing RSSM-learned features
against raw macro features, without transaction costs for a fair architecture comparison.

\begin{table}[H]
\centering
\caption{Ablation Study — RSSM vs Raw Macro vs Momentum (9-year walk-forward, no TC)}
\label{tab:ablation}
\begin{tabular}{lrrr}
\toprule
Approach & Parameters & Mean Calmar & Mean Sharpe \\
\midrule
Raw Macro (6-dim) & $\sim$30 & \textbf{2.051} & \textbf{1.185} \\
RSSM + PCA (20-dim) & $\sim$130,000 & 1.831 & 1.000 \\
Momentum Rotation & 0 & 1.321 & 0.753 \\
\bottomrule
\end{tabular}
\end{table}

Raw macro features outperform the RSSM by 12\% on Calmar and 18.5\% on Sharpe, using
99.95\% fewer parameters. The RSSM approach wins only 4 of 9 years against the macro
baseline. This result is robust across K-Means random seeds.

## 4.3 Regime Interpretability

Table \ref{tab:regimes} shows the 6 discovered regimes with their macro-economic profiles
and asset-class returns. The regimes correspond to well-understood economic narratives.

\begin{table}[H]
\centering
\caption{Regime Profiles with Economic Labels (Full Dataset 2010–2024)}
\label{tab:regimes}
\begin{tabular}{lrrrrrrr}
\toprule
Regime & Days & VIX & Spread & SPY & TLT & GLD & DBC \\
\midrule
\#0 Moderate Growth & 798 & 17.3 & 0.29 & +13.7\% & -1.2\% & +9.6\% & +2.0\% \\
\#1 Mild Risk-On & 814 & 17.9 & 1.47 & +8.6\% & +3.8\% & -6.3\% & -2.9\% \\
\#2 Goldilocks & 794 & 17.6 & 1.41 & +14.2\% & +10.6\% & -0.9\% & -12.0\% \\
\#3 COVID Panic & 48 & \textbf{52.4} & 0.64 & -14.6\% & \textbf{+56.0\%} & \textbf{+55.9\%} & -122.6\% \\
\#4 Low Vol Bull & 762 & 15.7 & 0.73 & +15.5\% & +9.4\% & +18.7\% & +7.5\% \\
\#5 Inflationary & 495 & 24.8 & 0.33 & +25.1\% & -12.6\% & +10.8\% & +32.9\% \\
\bottomrule
\end{tabular}
\end{table}

Regime \#3 (COVID panic) is immediately identifiable by its extreme VIX of 52.4 and
flight-to-safety behavior (TLT +56\%, GLD +56\%). Regime \#5 captures the post-COVID
inflationary boom (DBC +33\%, SPY +25\%, TLT -13\%). Regime \#4 is the classical
low-volatility bull market. These interpretable labels validate that the clustering
captures genuine economic structure.

## 4.4 Benchmark Comparison

Table \ref{tab:benchmarks} compares all approaches on the 2022–2024 out-of-sample period
with transaction costs included.

\begin{table}[H]
\centering
\caption{Strategy Comparison — SPY/TLT/GLD/DBC 2022–2024 (with 6bp transaction costs)}
\label{tab:benchmarks}
\begin{tabular}{lrrrr}
\toprule
Strategy & Calmar & Sharpe & Cum. Return & Max DD \\
\midrule
SPY Buy \& Hold & 1.171 & 0.571 & +28.7\% & -24.5\% \\
Risk Parity & 0.733 & 0.448 & +13.4\% & -18.3\% \\
Macro Regime (production) & 0.645 & 0.332 & +10.6\% & -16.4\% \\
60/40 SPY/TLT & 0.045 & 0.096 & +1.2\% & -26.2\% \\
\bottomrule
\end{tabular}
\end{table}

The macro regime allocator delivers a 14.3× improvement in Calmar ratio over static 60/40,
primarily through drawdown reduction (-16.4\% vs -26.2\%). However, it underperforms both
risk parity and SPY buy-and-hold in absolute terms, confirming that its value proposition
is risk management, not return enhancement.

# 5. Discussion

## 5.1 Why Does Simplicity Win?

The superiority of raw macro features over RSSM-learned representations is consistent
with the low signal-to-noise ratio characteristic of daily financial data. With only
2,902 training days, the RSSM's 130,000 parameters are fitting noise rather than signal.
The macro features, by contrast, represent centuries of economic intuition distilled
into 6 high-SNR dimensions. This finding echoes the broader machine learning principle
that representation learning requires either very large datasets or very strong inductive
biases—neither of which is available at daily frequency in a single asset class.

## 5.2 The Risk Management Contribution

While no approach beats simple SPY buy-and-hold in absolute returns, the regime-based
macro allocator provides a genuine risk management benefit: reducing maximum drawdown
from -26\% (60/40) to -16\% (macro regime) during the 2022–2024 period, when the
historic negative stock-bond correlation broke down. For institutional investors with
drawdown constraints, this capital preservation during a correlation crisis represents
actionable value.

## 5.3 Limitations

Several limitations should be acknowledged. First, the 9-year walk-forward window
(2016–2024) covers approximately one and a half business cycles, dominated by the
post-GFC expansion and COVID disruption. A longer backtest spanning multiple full
cycles would strengthen the findings. Second, the coordinate ascent optimization
may overfit to small regimes (regime \#3 has only 48 days). Third, the paper does
not address the well-known difficulty of predicting regime transitions in real time;
the K-Means labels are assigned retrospectively. Fourth, transaction costs of 6bp
may underestimate real-world costs for large institutional trades.

## 5.4 Future Work

Several directions merit further investigation. First, replacing K-Means with a Hidden
Markov Model would provide regime transition probabilities and prevent the "regime
flipping" that generates excess turnover. Second, Bayesian shrinkage on per-regime
weights would prevent overfitting to sparse regimes. Third, testing the regime labels
on unseen asset classes (crypto, sector ETFs) would verify whether the macro features
capture universal market states or SPY-specific patterns. Fourth, higher-frequency
data (5-minute bars) could enable microstructure-aware regime detection that daily
bars cannot capture.

# 6. Conclusion

We have presented a comprehensive empirical study comparing learned latent representations
(RSSM) against engineered macro features for the task of financial regime detection and
multi-asset portfolio allocation. Our central finding is that 6 simple macro-economic
features (VIX, yield spread, momentum) outperform a 130,000-parameter recurrent state-space
model by 12\% on the Calmar ratio in a 9-year walk-forward validation, while providing
interpretable regime labels that correspond to well-known economic narratives.

The primary contribution is not a new alpha source, but rather a rigorously validated
demonstration that **for daily-frequency macro regime allocation, domain expertise in
feature engineering dominates representation learning.** The discovered regimes enable
meaningful tail-risk reduction compared to static 60/40 portfolios, representing a
practical risk management tool for institutional investors.

All code, data pipelines, and experiment configurations are publicly available at
\url{https://github.com/lesterppo/stock-world-model}.

# References
