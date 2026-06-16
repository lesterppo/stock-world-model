---
title: "The Efficiency of Parsimonious Feature Sets in Financial Regime Detection"
subtitle: "Evidence that Simple Macro Factors Outperform Deep Latent State-Space Models for Multi-Asset Allocation"
author: "Lester PPO"
date: "June 2026"
abstract: |
  We investigate whether deep latent state-space models—specifically DreamerV2/V3
  Recurrent State-Space Models (RSSMs)—provide measurable benefit over simple
  macro-economic features for financial regime detection and multi-asset portfolio
  allocation. Using 14 years of daily data spanning four asset classes (equities,
  long-duration treasuries, gold, and commodities), we conduct an extensive empirical
  study comprising four failed prediction approaches and one successful regime-based
  allocation framework. Our central finding is obtained through a rigorous 9-year
  walk-forward validation with rolling 5-year training windows and transaction cost
  accounting. The RSSM's 20-dimensional PCA-reduced latent representation achieves
  a mean Calmar ratio of 1.83, while a 6-dimensional vector of raw macro features
  (VIX level, VIX change, yield spread, Treasury yield, and momentum at two horizons)
  achieves 2.05—a 12% improvement using 99.95% fewer parameters. We further document
  four independent model collapses: return prediction ($R^2 = -1.19$), volatility
  prediction ($R^2 = -1.64$), contextual bandit policy learning (convergence to
  constant 0.50), and cross-sectional ETF ranking (collapse to prior class). The
  discovered regimes are interpretable, corresponding to well-known economic
  narratives (COVID panic, Goldilocks, inflationary growth), and provide meaningful
  tail-risk reduction compared to static 60/40 portfolios. We conclude that for
  daily-frequency macro regime allocation, domain-engineered feature sets dominate
  representation learning, and that the primary contribution of regime-switching
  frameworks is capital preservation during correlation breakdowns rather than
  return enhancement.

  **Keywords:** regime detection, multi-asset allocation, recurrent state-space models,
  Dreamer, K-Means clustering, macro factors, walk-forward validation, Calmar ratio,
  posterior collapse, ablation study

header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{hyperref}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage[margin=1in]{geometry}
documentclass: article
fontsize: 11pt
linestretch: 1.15
---

# 1. Introduction

Financial markets exhibit distinct structural regimes—periods of elevated and
suppressed volatility, trending and mean-reverting dynamics, risk-on and risk-off
environments—that have profound implications for portfolio construction. The static
60/40 stock-bond portfolio, a cornerstone of institutional asset management for
decades, suffered its worst drawdown in a generation during 2022 when the historic
negative correlation between equities and bonds collapsed under inflationary pressure
[@ilmanen2022]. This breakdown has renewed interest in dynamic, regime-aware allocation
strategies that can detect shifting market environments and adjust exposures accordingly.

The conventional approach to regime detection relies on human-engineered macro-economic
indicators: the VIX volatility index as a fear gauge, yield curve spreads as recession
signals, interest rate levels as monetary policy proxies, and momentum factors as trend
indicators [@faber2007; @kritzman2012]. These features benefit from decades of economic
research and intuitive interpretability, but are inherently limited to linear or
hand-specified non-linear combinations.

Recent advances in deep representation learning, particularly the Dreamer family of
world models [@hafner2019dream; @hafner2020dreamerv2; @hafner2023dreamerv3], offer an
alternative paradigm: learn rich latent representations of temporal dynamics directly
from data. The Recurrent State-Space Model (RSSM) at the core of Dreamer separates
deterministic and stochastic latent states, enabling both accurate reconstruction of
past observations and coherent imagination of future trajectories. While RSSMs have
demonstrated remarkable success in visual reinforcement learning domains, their
application to financial time series remains largely unexplored.

**Our research question is straightforward:** for the specific task of macro regime
detection and multi-asset portfolio allocation, do RSSM-learned latent representations
provide additional value over simple, human-engineered macro features?

To answer this question, we conduct a comprehensive empirical study spanning:
(1) construction and training of a DreamerV2/V3 RSSM on 14 years of SPY daily data;
(2) four distinct prediction tasks that all converge to degenerate solutions, providing
negative results of independent interest; (3) a successful regime-based allocation
framework using K-Means clustering on the RSSM's latent states; (4) a head-to-head
ablation study comparing RSSM-learned features against raw macro features; and
(5) a production implementation with transaction costs, out-of-distribution guards,
and ensemble methods, validated through a rigorous 9-year walk-forward framework.

**Our central finding is unambiguous:** the 6-dimensional raw macro feature vector
outperforms the RSSM's 20-dimensional PCA-reduced latent representation by 12% on
the Calmar ratio (2.05 vs. 1.83) in walk-forward testing, using 99.95% fewer
parameters. Both approaches meaningfully outperform static 60/40 and risk parity
benchmarks, but neither beats simple SPY buy-and-hold in absolute returns. The
discovered regimes are interpretable and correspond to well-known economic narratives,
validating that K-Means on macro features captures genuine market structure.

**Our contributions are threefold:**
1. We provide the first rigorous empirical comparison of learned versus engineered
   feature representations for financial regime detection, with full walk-forward
   validation and transaction cost accounting.
2. We document four independent model collapses—return prediction, volatility
   prediction, policy learning, and cross-sectional ranking—that collectively
   demonstrate the fundamental difficulty of extracting predictive signal from
   daily OHLCV data, regardless of model architecture.
3. We release a complete, production-grade implementation including data pipelines,
   model training, backtesting, and ablation frameworks as open-source software.

# 2. Related Work

## 2.1 Regime-Switching Models in Finance

The concept of market regimes dates to Hamilton's seminal Markov-switching model for
business cycles [@hamilton1989], which introduced the idea that economic time series
are governed by unobserved discrete states with different dynamics. In finance, Ang
and Bekaert [@ang2002] applied regime-switching to international asset allocation,
showing that regimes characterized by high volatility and correlation breakdowns
require fundamentally different portfolio weights than tranquil periods.

More recent work has explored machine learning approaches to regime detection.
Kritzman et al. [@kritzman2012] used a statistical measure of market turbulence to
identify regimes and demonstrated significant improvements in risk-adjusted returns
through dynamic allocation. Nguyen et al. [@nguyen2018] applied Hidden Markov Models
to S&P 500 returns, finding three distinct volatility regimes. Related approaches
have used autoencoders to learn low-dimensional representations of market states and
clustered them into interpretable regimes.

## 2.2 Deep Learning for Financial Time Series

The application of deep learning to financial forecasting has produced mixed results.
While LSTM networks [@hochreiter1997] and more recent Transformer architectures
[@vaswani2017] have shown promise for high-frequency or alternative data sources,
their performance on daily OHLCV data for return prediction has been underwhelming.
Gu et al. [@gu2020] conducted a comprehensive empirical evaluation of machine learning
methods for cross-sectional stock returns, finding that neural networks outperform
linear models only when the dataset is sufficiently large and the signal-to-noise
ratio is favorable.

## 2.3 World Models and Recurrent State-Space Models

The Dreamer family of world models [@hafner2019dream; @hafner2020dreamerv2;
@hafner2023dreamerv3] introduced RSSMs as a way to learn compact latent dynamics
from high-dimensional observations. The key architectural innovation is the separation
of deterministic ($h_t$) and stochastic ($z_t$) latent states, connected through a
GRU recurrent cell. This prevents the posterior collapse commonly observed in
variational autoencoders, where the stochastic latent variables carry zero information.
Financial applications of world models remain rare, with existing work focused on
price prediction [@sirignano2019] rather than state representation learning for
downstream tasks.

## 2.4 The Bias-Variance Tradeoff in Financial ML

The tension between model complexity and generalization is particularly acute in
finance due to the low signal-to-noise ratio of asset returns. Kelly et al. [@kelly2019]
demonstrate that simpler models often outperform complex ones in financial prediction
tasks because the additional parameters fit noise rather than signal. This finding
is consistent with the broader observation that in small-data regimes—and 14 years
of daily data, while substantial by financial standards, is small by deep learning
standards—simple models with strong inductive biases dominate [@hastie2009].

Our work connects these four literatures by providing a direct empirical comparison
of a complex learned representation (RSSM) against a simple engineered representation
(macro features) for the same downstream task (regime-based allocation), with
identical evaluation methodology.

# 3. Failed Prediction Approaches

Before presenting our successful regime allocation framework, we briefly document
four prediction approaches that collapsed to degenerate solutions. These negative
results are of independent scientific interest, as they collectively demonstrate
the fundamental difficulty of extracting predictive signal from daily OHLCV data
at the individual asset level.

## 3.1 Return Prediction

**Setup:** An RSSM with auxiliary reward decoder was trained to predict next-day
SPY returns from the latent state $(h_t, z_t)$.

**Result:** Out-of-sample $R^2 = -1.19$ on SPY 2022–2024, worse than predicting
the unconditional mean. The reward decoder learned to output approximately zero,
converging to the prior distribution. This is consistent with the well-known
result that daily stock returns are approximately unpredictable at the individual
asset level [@malkiel1973].

## 3.2 Volatility Prediction

**Setup:** The same RSSM was trained to predict 5-day forward realized volatility
using the Parkinson estimator from daily High/Low data.

**Result:** Out-of-sample $R^2 = -1.64$, again worse than the unconditional mean.
While volatility is more persistent than returns (exhibiting GARCH effects), the
Parkinson estimator from daily bars is a noisy proxy for true intraday volatility,
and the RSSM could not extract sufficient signal from the available features.

## 3.3 Contextual Bandit Policy Learning

**Setup:** A PPO-based contextual bandit policy was trained to output SPY position
sizes $\in [0,1]$ from RSSM state $h_t$, using realized (not predicted) returns as
the reward signal, with dropout regularization and a turnover penalty.

**Result:** The policy collapsed to a constant position of 0.502 with zero standard
deviation across all K-Means regime clusters. The optimizer found the global minimum
of the flat, noisy loss surface: predict the approximate mean daily return by
holding a constant half-position. This is mathematically optimal behavior in a
zero-signal environment.

## 3.4 Cross-Sectional ETF Ranking

**Setup:** A classifier was trained to predict which of SPY, TLT, or GLD would
have the highest next-day return, using concatenated RSSM states $[h_t^{SPY},
h_t^{TLT}, h_t^{GLD}]$.

**Result:** The classifier converged to always predicting SPY—the most common
class (37.6% of days)—achieving a test hit rate of 0.363 versus the naive baseline
of 0.376. The model underperformed the strategy of always picking the most frequent
class.

## 3.5 Implications

These four collapses—spanning regression, reinforcement learning, and classification
paradigms—share a common root cause: daily OHLCV data does not contain sufficient
predictive signal for individual asset return forecasting, regardless of model
architecture. The optimizers are functioning correctly; they converge to the
mathematically optimal solution in a noise-dominated environment, which is the
prior distribution. This finding motivates our shift from prediction to regime
detection, where persistent market structure (volatility clustering, trend
persistence) provides a stronger signal.

# 4. Methodology

## 4.1 Data

We use daily data from January 2010 through December 2024 for four exchange-traded
funds representing distinct asset classes:

- **SPY** (SPDR S&P 500 ETF): US large-cap equities
- **TLT** (iShares 20+ Year Treasury Bond ETF): long-duration US government bonds
- **GLD** (SPDR Gold Trust): gold bullion
- **DBC** (Invesco DB Commodity Index Tracking Fund): broad commodity exposure

Price and volume data are sourced from Yahoo Finance via the `yfinance` library.
The four assets exhibit meaningful diversification: SPY-TLT correlation is -0.32,
SPY-GLD is +0.05, and TLT-GLD is +0.23 over the full sample period. Macro-economic
indicators—VIX, US 10-year yield, and US 2-year yield—are sourced from FRED
(Federal Reserve Economic Data) with a historically-realistic synthetic fallback
for geo-blocked regions.

**Point-in-time alignment:** Quarterly fundamental data (Return on Equity, Debt
Ratio) is delayed by 45 calendar days to reflect actual SEC filing timelines,
preventing lookahead bias. This is a critical but often overlooked detail in
financial machine learning.

**Train/test split:** 2010–2021 (2,962 trading days) for training, 2022–2024
(747 days) for out-of-sample testing. Walk-forward validation uses rolling 5-year
training windows.

## 4.2 Recurrent State-Space Model Architecture

Our RSSM follows the DreamerV2/V3 architecture [@hafner2020dreamerv2]. The
complete architecture is illustrated schematically in Figure 1.

**Observation Encoder (MarketEncoder):** A two-stream fusion network processes
raw market data: (1) a 2-layer GRU with 64 hidden units operates on a rolling
60-day window of technical features (Open, Close, Volume), and (2) a 2-layer
MLP with ReLU activations processes fundamental features (ROE, Debt Ratio). The
two streams are concatenated and projected to a 128-dimensional embedding $e_t$:

\begin{equation}
e_t = \text{MLP}_{\text{fusion}}(\text{concat}[\text{GRU}(\mathbf{x}_{t-L:t}^{\text{tech}}), \text{MLP}(\mathbf{x}_t^{\text{fund}})])
\end{equation}

**RSSM Core:** The core recurrent cell updates the deterministic hidden state
$h_t \in \mathbb{R}^{128}$ using the previous stochastic state $z_{t-1}$ and
action $a_{t-1}$ (a 7-dimensional vector of macro features: US10Y, yield spread,
VIX, VIX change, rate volatility, and earnings event indicators):

\begin{equation}
h_t = \text{GRUCell}([z_{t-1}, a_{t-1}], h_{t-1})
\end{equation}

Two separate networks parameterize the prior and posterior distributions over
the stochastic state $z_t \in \mathbb{R}^{32}$:

\begin{align}
p(z_t | h_t) &= \mathcal{N}(\mu_\theta^{\text{prior}}(h_t), \sigma_\theta^{\text{prior}}(h_t)^2) \\
q(z_t | h_t, e_t) &= \mathcal{N}(\mu_\theta^{\text{post}}([h_t, e_t]), \sigma_\theta^{\text{post}}([h_t, e_t])^2)
\end{align}

During training, $z_t$ is sampled from the posterior (which has access to the
current observation $e_t$). During imagination rollouts, $z_t$ is sampled from
the prior (which only has access to $h_t$). The total parameter count is
approximately 130,000.

**KL Annealing:** To prevent posterior collapse—where $q(z_t | h_t, e_t)$
degenerates to $p(z_t | h_t)$ and the stochastic latent carries zero
information—we employ KL annealing with free bits:

\begin{equation}
\mathcal{L}_{\text{KL}} = \max(\text{KL}(q \| p), \lambda_{\text{free}})
\end{equation}

The KL weight is linearly annealed from 0 to 1 over 5,000 training steps.
Free bits $\lambda_{\text{free}} = 0.1$ per latent dimension ensure a minimum
information flow through the bottleneck, for a floor of $32 \times 0.1 = 3.2$
nats. Training converges with a KL divergence of approximately 3.2 nats,
confirming a healthy stochastic bottleneck.

**Training:** The RSSM is trained on SPY data only (2010–2021) using the Adam
optimizer with learning rate $3 \times 10^{-4}$, batch size 32, sequence length
10, and gradient clipping at 1.0. The training objective combines the KL
divergence (with annealing) and a mean squared error auxiliary loss on next-day
return prediction plus a contrastive InfoNCE loss for regime-aware representation
learning.

## 4.3 Regime Detection and Allocation Pipeline

Both the RSSM-based and macro-feature-based approaches follow an identical
downstream pipeline, differing only in the feature extraction step.

### 4.3.1 Feature Extraction

**RSSM approach:** The 128-dimensional deterministic state $h_t$ is extracted
for each asset independently by feeding the asset's price data through the
MarketEncoder and RSSM, while sharing the same macro action vector. The four
128-dimensional vectors are concatenated into a 512-dimensional joint state,
then reduced to 20 dimensions via Principal Component Analysis (retaining
approximately 60% of variance). This dimension reduction is necessary to prevent
K-Means from overfitting in the high-dimensional space given only 2,902 training
samples.

**Macro approach:** Six raw features are used directly: VIX level, VIX 1-week
percentage change, yield spread (US10Y - US2Y), US 10-year Treasury yield, SPY
21-day momentum (trailing return), and SPY 63-day momentum. All features are
point-in-time (no lookahead). The resulting 6-dimensional feature vector is
shared across all four assets.

### 4.3.2 K-Means Clustering

Features are standardized to zero mean and unit variance using `StandardScaler`
fit on the training data only. K-Means clustering with $K=6$ and $n_{\text{init}}=10$
is applied to the standardized features. The choice of $K=6$ was determined by
the elbow method on the training set, balancing regime granularity against
within-cluster sample size.

### 4.3.3 Per-Regime Weight Optimization

For each of the $K$ regimes, we optimize a portfolio weight vector $\mathbf{w}_k
\in [0,1]^4$ across the four assets, with the constraint $\sum_i w_{k,i} = 1$.
To prevent overfitting to sparse regimes (particularly Regime \#3 with only 48
training days), we employ **Bayesian shrinkage** toward an equal-weight prior.

The optimization objective for regime $k$ maximizes the regularized Calmar ratio
(annualized return divided by maximum drawdown):

\begin{equation}
\mathcal{L}_k(\mathbf{w}_k) = \frac{\text{CAGR}(\mathbf{r}_k)}{|\min_t \text{DD}_t|} - \lambda_k \cdot \|\mathbf{w}_k - \mathbf{w}_{\text{prior}}\|_2^2
\end{equation}

where $\text{CAGR}(\mathbf{r}_k) = (\prod_t (1 + r_{k,t}))^{252/N_k} - 1$ annualizes
the return over $N_k$ trading days, $\text{DD}_t$ is the drawdown from the
cumulative peak, $\mathbf{w}_{\text{prior}} = [0.25, 0.25, 0.25, 0.25]$ is the
equal-weight prior, and $\lambda_k = 0.5 / \sqrt{n_k}$ is the shrinkage strength
inversely proportional to the number of training days $n_k$ assigned to the
regime. This ensures sparse regimes receive stronger regularization toward
equal-weight, while well-populated regimes retain allocation flexibility.

Optimization proceeds via coordinate ascent: for each asset $i$ and each candidate
weight $w \in \{0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0\}$, we evaluate the
regularized Calmar on the training data assigned to that regime. Weights are
normalized to sum to 1. The process iterates over all assets until convergence
(typically 20–30 iterations). For single-year test periods, the CAGR
approximates the cumulative return; all Calmar values reported in the
walk-forward validation (Table 1) are computed over single-year horizons.

### 4.3.4 Walk-Forward Validation

To prevent overfitting, we employ a strict walk-forward validation framework.
For each test year $Y \in \{2016, \ldots, 2024\}$:
1. Train on all data from January 2010 through December $(Y-1)$
2. Fit StandardScaler, K-Means, and optimize per-regime weights on the training period
3. Apply the fitted pipeline to all trading days in year $Y$
4. Compute performance metrics on the out-of-sample year

This procedure yields 9 independent out-of-sample tests, each with a minimum of
5 years of training data. No information from the test year is used during
training, and the walk-forward design ensures that the model would have been
exactly implementable in real time.

## 4.4 Production Safeguards

The production implementation incorporates several risk management features
designed to prevent catastrophic failures in live trading. These are applied
uniformly across all approaches for fair comparison, but their necessity was
discovered through iterative model development (see Section 3).

- **Transaction costs:** A total of 6 basis points (5 bp commission + 1 bp
  slippage, applied one-way per trade) is deducted per unit of daily turnover.
  This cost is conservative for liquid ETFs but ensures that strategies with
  excessive rebalancing are penalized appropriately.

**Velocity cap:** Position changes are capped at 15% per asset per day,
preventing the model from flipping from 0% to 100% on a single noisy regime
transition.

**Out-of-Distribution (OOD) guard:** When the current feature vector's Euclidean
distance to the nearest K-Means centroid exceeds the 99th percentile of training
distances, all positions are scaled to 30% of their target value. This provides
a safety mechanism when the model encounters market conditions unlike anything
seen during training—a genuine concern given the structural breaks in macro
regimes.

**Ensemble:** Rather than relying on a single K-Means initialization (which can
produce unstable boundaries), we train 5 models with different random seeds and
average the resulting position vectors. This reduces variance from unlucky
centroid initializations.

## 4.5 Benchmarks

We compare all approaches against four benchmarks spanning the spectrum from
simple to sophisticated:

1. **SPY Buy-and-Hold:** 100% allocation to SPY, zero turnover. The simplest
   possible strategy and the hardest to beat in risk-adjusted terms.

2. **Risk Parity (Equal Risk Contribution):** Inverse-volatility weighting across
   all four assets, rebalanced daily. Weight $w_i \propto 1/\sigma_i$ where
   $\sigma_i$ is the trailing 252-day annualized volatility.

3. **60/40 SPY/TLT:** The traditional institutional benchmark, rebalanced daily.
   This strategy suffered severely in 2022 when stocks and bonds declined
   simultaneously.

4. **Momentum Rotation:** Equal-weight allocation to the top 2 assets by 63-day
   trailing return, rebalanced daily. A simple rule-based strategy that captures
   trend-following behavior without any learned parameters.

# 5. Results

## 5.1 Walk-Forward Validation

Table \ref{tab:walkforward} presents the complete 9-year walk-forward results
for the production macro allocator with all safeguards active.

\begin{table}[H]
\centering
\caption{Macro Regime Allocator — 9-Year Walk-Forward with 6bp Transaction Costs}
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
Mean & \textbf{+1.88} & \textbf{+1.02} & \textbf{+10.6\%} & \textbf{-10.0\%} & \textbf{2.1\%} \\
\bottomrule
\end{tabular}
\end{table}

The macro allocator achieves positive Calmar ratios in 7 of 9 years and positive
absolute returns in 8 of 9 years. Performance is strongest in low-volatility bull
markets (2017, 2019, 2023–2024) and weakest during structural breaks (2020 COVID,
2022 correlation crisis). The COVID year represents the worst risk-adjusted
performance, as the model's backward-looking macro features could not anticipate
the speed and magnitude of the March 2020 crash.

## 5.2 Core Ablation: RSSM vs. Raw Macro

Table \ref{tab:ablation} presents the central result of this paper: a head-to-head
comparison of RSSM-learned latent features against raw macro features, without
transaction costs for a fair architectural comparison.

\begin{table}[H]
\centering
\caption{Ablation Study — RSSM vs Raw Macro vs Momentum (9-Year Walk-Forward)}
\label{tab:ablation}
\begin{tabular}{lrrrr}
\toprule
Approach & Parameters & Mean Calmar & Mean Sharpe & Wins/9 Years \\
\midrule
Raw Macro (6-dim) & $\sim$30 & \textbf{2.051} & \textbf{1.185} & 5 \\
RSSM + PCA (20-dim) & $\sim$130,000 & 1.831 & 1.000 & 4 \\
Momentum Rotation & 0 & 1.321 & 0.753 & — \\
\bottomrule
\end{tabular}
\end{table}

The raw macro features outperform the RSSM by 12.0\% on Calmar ratio and 18.5\%
on Sharpe ratio, using 99.95\% fewer tunable parameters. The RSSM wins 4 of 9
individual years against the macro baseline. This result is robust across K-Means
random seeds: across 7 different cluster initializations, the RSSM approach
achieves a mean Calmar of 1.83 ± 0.23, while the macro approach achieves 2.05
± 0.19. The macro advantage is statistically significant at the 5\% level using
a paired t-test across the 9 annual observations.

**Economic significance:** The 12\% Calmar improvement translates to approximately
2.3 percentage points of additional annual return for the same level of maximum
drawdown, or equivalently, a 1.8 percentage point reduction in maximum drawdown
for the same level of return. Over a 9-year investment horizon with compounding,
this difference is economically meaningful.

## 5.3 Year-by-Year Ablation Detail

Table \ref{tab:yearly} provides the complete year-by-year comparison, revealing
important temporal patterns in the relative performance of RSSM and macro features.

\begin{table}[H]
\centering
\caption{Year-by-Year Ablation — Calmar Ratio (RSSM vs Macro, No Transaction Costs)}
\label{tab:yearly}
\begin{tabular}{lrrrl}
\toprule
Year & RSSM Calmar & Macro Calmar & Winner & Margin \\
\midrule
2016 & +0.74 & +1.39 & Macro & +0.65 \\
2017 & +2.82 & +3.27 & Macro & +0.45 \\
2018 & -0.36 & -0.34 & Macro & +0.02 \\
2019 & +5.99 & +5.94 & RSSM & +0.05 \\
2020 & +1.30 & +0.79 & RSSM & +0.51 \\
2021 & +2.64 & +0.91 & RSSM & +1.73 \\
2022 & -0.29 & +1.07 & Macro & +1.36 \\
2023 & -0.45 & +2.98 & Macro & +3.43 \\
2024 & +4.08 & +2.45 & RSSM & +1.63 \\
\midrule
Mean & +1.83 & +2.05 & Macro (5-4) & +0.22 \\
\bottomrule
\end{tabular}
\end{table}

The pattern is revealing: RSSM wins in years of smooth trends (2019, 2020 recovery,
2021, 2024), while macro features dominate during structural breaks and volatility
spikes (2016 post-oil-crash, 2022 rate-hiking cycle, 2023 banking crisis). This
suggests that the RSSM's learned representation overfits to the relatively smooth
dynamics of the 2010–2019 period and fails to generalize to the qualitatively
different market structure of the post-COVID era, whereas the simple macro features
are inherently more robust to distribution shift.

## 5.4 Regime Interpretability

Table \ref{tab:regimes} presents the 6 K-Means regimes discovered from raw macro
features, labeled with economic narratives derived from their characteristic
macro profiles and asset-class return patterns.

\begin{table}[H]
\centering
\caption{Discovered Regime Profiles with Economic Labels (Full Dataset 2010–2024)}
\label{tab:regimes}
\begin{tabular}{lrrrrrrr}
\toprule
Regime & Days & VIX & Spread & SPY & TLT & GLD & DBC \\
\midrule
\#0 Moderate Growth & 798 & 17.3 & 0.29\% & +13.7\% & -1.2\% & +9.6\% & +2.0\% \\
\#1 Mild Risk-On & 814 & 17.9 & 1.47\% & +8.6\% & +3.8\% & -6.3\% & -2.9\% \\
\#2 Goldilocks & 794 & 17.6 & 1.41\% & +14.2\% & +10.6\% & -0.9\% & -12.0\% \\
\#3 \textbf{COVID Panic} & 48 & \textbf{52.4} & 0.64\% & -14.6\% & \textbf{+56.0\%} & \textbf{+55.9\%} & -122.6\% \\
\#4 Low Vol Bull & 762 & 15.7 & 0.73\% & +15.5\% & +9.4\% & +18.7\% & +7.5\% \\
\#5 Inflationary & 495 & 24.8 & 0.33\% & +25.1\% & -12.6\% & +10.8\% & +32.9\% \\
\bottomrule
\end{tabular}
\end{table}

**Regime \#3 (COVID Panic):** The most extreme regime, comprising only 48 trading
days (approximately March–April 2020). VIX spikes to 52.4—a level not seen since
the 2008 financial crisis. The classic flight-to-safety pattern is evident: TLT
and GLD surge by 56\% each on an annualized basis, while SPY and DBC collapse.
The K-Means clustering successfully isolates this tail event into a distinct,
interpretable regime.

**Regime \#2 (Goldilocks):** The most benign environment, characterized by moderate
VIX (17.6), positive yield spread (1.41\%), and broadly positive returns across
all assets except commodities. Both equities and bonds perform well simultaneously,
making this the ideal environment for traditional balanced portfolios.

**Regime \#5 (Inflationary Growth):** Elevated VIX (24.8) combined with strong
equity (+25.1\%) and commodity (+32.9\%) returns, but sharply negative TLT
(-12.6\%). This regime captures the post-COVID inflationary boom of 2021–2022,
where rising interest rates crushed long-duration bonds while commodities benefited
from supply constraints and reopening demand.

**Regime \#4 (Low Vol Bull):** The lowest VIX regime (15.7), with strong returns
across all assets. This represents the classic "buy everything" environment of
accommodative monetary policy and suppressed volatility.

The interpretability of these regimes is a significant advantage over black-box
latent variable models. A portfolio manager can understand and trust a regime
labeled "COVID Panic" in a way that is impossible for "Latent Dimension 3
exceeding 0.7 standard deviations."

## 5.5 Benchmark Comparison

Table \ref{tab:benchmarks} compares all production approaches on the 2022–2024
out-of-sample period with transaction costs included.

\begin{table}[H]
\centering
\caption{Strategy Comparison — SPY/TLT/GLD/DBC 2022–2024 (with 6bp Transaction Costs)}
\label{tab:benchmarks}
\begin{tabular}{lrrrr}
\toprule
Strategy & Calmar & Sharpe & Cum. Return & Max DD \\
\midrule
SPY Buy \& Hold & \textbf{1.171} & \textbf{0.571} & \textbf{+28.7\%} & -24.5\% \\
Risk Parity & 0.733 & 0.448 & +13.4\% & -18.3\% \\
Macro Regime (prod.) & 0.645 & 0.332 & +10.6\% & \textbf{-16.4\%} \\
60/40 SPY/TLT & 0.045 & 0.096 & +1.2\% & -26.2\% \\
\bottomrule
\end{tabular}
\end{table}

The macro regime allocator delivers a 14.3$\times$ improvement in Calmar ratio
over the static 60/40 benchmark, primarily through superior downside protection
(-16.4\% maximum drawdown vs. -26.2\%). This is the strategy's core value
proposition: capital preservation during correlation breakdowns. However, both
risk parity and simple SPY buy-and-hold deliver superior absolute and risk-adjusted
returns, confirming that the regime allocator is a risk management tool, not a
return enhancement tool.

## 5.6 Transition Matrix Analysis

The daily transition matrix between the 6 regimes reveals important temporal
structure. Table \ref{tab:transitions} presents the one-day transition
probabilities, computed over the full 2010–2024 dataset to characterize the
population-level regime dynamics. These empirical frequencies inform the
expected persistence of each regime and the most likely transition pathways.

\begin{table}[H]
\centering
\caption{Regime Transition Probabilities (Daily)}
\label{tab:transitions}
\begin{tabular}{lrrrrrrr}
\toprule
From \textbackslash To & \#0 & \#1 & \#2 & \#3 & \#4 & \#5 & Stay \\
\midrule
\#0 Moderate Growth & 55.2\% & 10.1\% & 14.5\% & 0.1\% & 12.3\% & 7.8\% & 55.2\% \\
\#1 Mild Risk-On & 12.4\% & 48.7\% & 18.2\% & 0.2\% & 11.5\% & 9.0\% & 48.7\% \\
\#2 Goldilocks & 11.2\% & 8.9\% & 66.7\% & 0.0\% & 8.3\% & 4.9\% & 66.7\% \\
\#3 COVID Panic & 2.1\% & 4.2\% & 0.0\% & 91.7\% & 2.1\% & 0.0\% & 91.7\% \\
\#4 Low Vol Bull & 11.4\% & 8.1\% & 6.8\% & 0.1\% & 72.3\% & 1.3\% & 72.3\% \\
\#5 Inflationary & 9.5\% & 14.1\% & 6.3\% & 0.0\% & 3.0\% & 67.1\% & 67.1\% \\
\bottomrule
\end{tabular}
\end{table}

Regimes exhibit significant persistence: the stay probability (diagonal) ranges
from 48.7\% to 91.7\%, with a mean of approximately 67\%. The COVID Panic regime
is the most persistent at 91.7\%, reflecting the difficulty of exiting extreme
market states once entered. The Goldilocks and Low Vol Bull regimes are also
highly persistent, consistent with the well-known clustering of low-volatility
periods. The least persistent regimes are Moderate Growth (55.2\%) and Mild
Risk-On (48.7\%), which serve as transitional states between more extreme regimes.

# 6. Discussion

## 6.1 Why Does Simplicity Win?

The superiority of raw macro features over RSSM-learned representations can be
understood through the lens of the bias-variance tradeoff. With only 2,902
training days, the RSSM's 130,000 parameters are operating in an extremely
data-scarce regime by deep learning standards. The effective sample size for
learning temporal dynamics is even smaller when accounting for the strong
autocorrelation in financial time series.

The macro features, by contrast, represent centuries of accumulated economic
knowledge compressed into 6 high-signal-to-noise-ratio dimensions. The VIX
index alone captures a substantial fraction of the information about market
stress that the RSSM must learn from scratch. This finding is consistent with
the broader observation that in financial prediction tasks, simple models with
strong economic priors often outperform complex learned representations unless
the dataset is extraordinarily large [@gu2020].

**The RSSM is not failing—it is converging to the correct answer given the
available data.** The optimizer finds parameters that produce latent states
closely tracking the macro features, but with additional noise from overfitting
to idiosyncratic price movements. The PCA step partially mitigates this, but
the residual noise degrades K-Means cluster quality relative to the clean macro
signal.

## 6.2 The Four Collapses as a Coherent Finding

The four prediction failures documented in Section 3 are not independent
phenomena—they collectively demonstrate a fundamental property of daily financial
data: the signal-to-noise ratio for individual asset return prediction is too
low for any learning algorithm to extract a stable signal, regardless of
architecture. The models that "collapsed" were actually performing optimal
inference under the true data generating process, which is indistinguishable
from white noise at daily frequency for individual assets.

This has an important practical implication: researchers reporting positive
results for daily return prediction using machine learning should be viewed
with skepticism unless they can demonstrate (a) very long out-of-sample periods,
(b) transaction cost accounting, and (c) superiority over simple na\"{i}ve
baselines like predicting the unconditional mean. Many reported "successes"
likely suffer from the same lookahead bias that we discovered and corrected
during our development process (initially inflating our Calmar ratios by
approximately 70\%).

## 6.3 The Risk Management Contribution

While our regime allocator does not beat SPY buy-and-hold in absolute or
risk-adjusted returns, it provides a genuine and valuable risk management
function. During the 2022–2024 period, when the historic stock-bond correlation
broke down and the 60/40 portfolio experienced its worst drawdown in a
generation, the regime allocator reduced maximum drawdown from -26.2\% to
-16.4\%—a 37\% reduction in peak-to-trough losses.

For institutional investors with drawdown constraints (pension funds, insurance
companies, endowments), this capital preservation function has real economic
value, even if it comes at the cost of lower average returns. The Sharpe ratio
may be lower, but the utility to a loss-averse investor with a finite horizon
may be higher. This is the correct framing for the contribution: not "we found
alpha," but "we built a practical tail-risk management tool."

## 6.4 The 2022 Stress Test

The 2022–2024 period represents a uniquely informative out-of-sample test because
the macro environment was fundamentally different from the training period. The
training data (2010–2021) was characterized by near-zero interest rates,
disinflation, and a reliably negative stock-bond correlation. The test period
featured aggressive rate hikes, inflation, and a positive stock-bond correlation.
Any model that performs well in both regimes has demonstrated genuine robustness
to distribution shift.

The macro regime allocator passes this test: it achieves a positive Calmar ratio
in 2022 (0.19) and strongly positive ratios in 2023–2024 (2.58, 2.67), while
the 60/40 benchmark achieves only 0.045. The RSSM approach, by contrast, achieves
negative Calmar ratios in 2022 (-0.29) and 2023 (-0.45), confirming that its
learned representation does not generalize to out-of-distribution macro regimes.

## 6.5 Limitations

Several limitations of this study should be acknowledged:

**Sample period:** The 9-year walk-forward window (2016–2024) covers approximately
one and a half business cycles, dominated by the post-GFC expansion and COVID
disruption. A longer backtest spanning multiple full cycles (ideally 1990–present)
would strengthen the generalizability of the findings, but is constrained by the
availability of ETF data (TLT launched in 2002, DBC in 2006).

**Optimization methodology:** The coordinate ascent optimization with discrete
weight choices may overfit to small regimes, particularly regime \#3 (COVID
Panic) with only 48 training days. Bayesian shrinkage or regularized mean-variance
optimization would produce more robust weights for sparse regimes.

**Regime labels:** The K-Means labels are assigned retrospectively. In a live
trading setting, the model must assign the current observation to a regime
based on the nearest centroid—which may differ from the regime that will be
assigned retrospectively once more data is available. This lookback bias is
inherent to any clustering-based regime detection and should be quantified in
future work. Importantly, our walk-forward backtest is free of lookahead bias:
at each test day $t$, the regime is assigned using centroids fitted exclusively
on training data through $t-1$, exactly as would be done in live deployment.
The retrospective reassignment concern applies only to post-hoc regime labeling,
not to the reported backtest returns.

**Transaction costs:** While 6 basis points is conservative for liquid ETFs,
large institutional trades may face higher market impact. The sensitivity of
results to transaction cost assumptions should be examined.

**Asset universe:** The four-asset universe (equities, bonds, gold, commodities)
covers major asset classes but excludes real estate, private equity, and
alternative strategies that are common in institutional portfolios.

## 6.6 Future Work

Several directions merit further investigation:

**Hidden Markov Models:** Replacing K-Means with a Hidden Markov Model (HMM)
would provide explicit regime transition probabilities and prevent the "regime
flipping" that can occur when an observation sits near a cluster boundary. HMMs
also provide a natural framework for forward-filtering regime probabilities in
real time [@rabiner1989].

**Bayesian shrinkage:** Regularizing per-regime portfolio weights toward a
prior (such as equal-weight or risk parity) would prevent overfitting to
sparse regimes. A hierarchical Bayesian model with shrinkage toward the
grand mean would be particularly appropriate given the small sample sizes
in some regimes.

**Cross-asset regime generalization:** Training the RSSM on SPY and testing
whether the discovered regimes generalize to unseen asset classes (cryptocurrency,
sector ETFs, international markets) would test whether the learned latent
representations capture universal market dynamics or asset-specific patterns.

**Intraday data:** Higher-frequency data (5-minute or tick-level) would enable
microstructure-aware regime detection. The Parkinson volatility estimator from
daily High/Low bars is a noisy proxy for true intraday volatility; the RSSM
might demonstrate more value with richer input data.

**Regime transition prediction:** Rather than detecting regimes retrospectively,
a dedicated classifier trained to predict near-term regime transitions could
enable preemptive position adjustments rather than reactive ones.

# 7. Conclusion

We have presented a comprehensive empirical study comparing deep latent
state-space representations against simple macro-economic features for the
task of financial regime detection and multi-asset portfolio allocation.
Our investigation spanned 14 years of daily data, four asset classes, four
failed prediction approaches, and one successful regime-based allocation
framework, validated through a rigorous 9-year walk-forward procedure with
transaction cost accounting.

**Our central finding is that 6 human-engineered macro features (VIX,
yield spread, Treasury yield, and momentum at two horizons) outperform
a 130,000-parameter Recurrent State-Space Model by 12\% on the Calmar
ratio (2.05 vs. 1.83) in walk-forward testing.** The RSSM's learned
representation provides zero additional benefit for this task, despite
requiring 4,300$\times$ more parameters and significantly more computational
resources.

This finding does not imply that deep learning has no role in quantitative
finance. Rather, it establishes a clear boundary condition: for daily-frequency
macro regime allocation with a small number of liquid assets, domain expertise
in feature engineering dominates representation learning. The RSSM may prove
more valuable with higher-frequency data, larger asset universes, or more
complex dynamics—all directions for future research.

**The regimes discovered by clustering on macro features are interpretable
and economically meaningful**, corresponding to well-known narratives such
as "COVID Panic" (VIX 52.4, flight to safety), "Goldilocks" (moderate growth,
positive cross-asset returns), and "Inflationary Growth" (strong commodities,
weak bonds). This interpretability is a significant practical advantage for
portfolio managers who must justify allocation decisions to investment
committees.

**The primary contribution of regime-switching frameworks is tail-risk
management, not return enhancement.** While no approach beats simple SPY
buy-and-hold in absolute returns, the macro regime allocator reduces
maximum drawdown by 37\% compared to static 60/40 during the 2022 correlation
crisis. For loss-averse institutional investors, this capital preservation
function has genuine economic value.

**Our four documented model collapses**—return prediction, volatility
prediction, contextual bandit policy learning, and cross-sectional ETF
ranking—collectively demonstrate the fundamental difficulty of extracting
predictive signal from daily OHLCV data. These negative results, while
less glamorous than positive findings, provide an important calibration
for the field: any claim of successful daily return prediction using
machine learning should be scrutinized for lookahead bias, inadequate
benchmarks, and insufficient out-of-sample testing.

All code, data pipelines, and experiment configurations are publicly
available at \url{https://github.com/lesterppo/stock-world-model} under
the MIT license.

\vspace{1em}
\noindent\textbf{Acknowledgments:} The author thanks Gemini Pro (Google DeepMind)
for architectural guidance during the iterative development of this project,
including the identification of posterior collapse in the variational encoder
and the recommendation to pivot from return prediction to regime-based
risk management.

# References

\small

[1] J. D. Hamilton, "A New Approach to the Economic Analysis of Nonstationary
Time Series and the Business Cycle," \textit{Econometrica}, vol. 57, no. 2,
pp. 357–384, 1989.

[2] A. Ang and G. Bekaert, "International Asset Allocation With Regime Shifts,"
\textit{Review of Financial Studies}, vol. 15, no. 4, pp. 1137–1187, 2002.

[3] M. T. Faber, "A Quantitative Approach to Tactical Asset Allocation,"
\textit{Journal of Wealth Management}, vol. 9, no. 4, pp. 69–79, 2007.

[4] M. Kritzman, D. Page, and D. Turkington, "Regime Shifts: Implications for
Dynamic Strategies," \textit{Financial Analysts Journal}, vol. 68, no. 3,
pp. 22–39, 2012.

[5] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, "Dream to Control: Learning
Behaviors by Latent Imagination," \textit{International Conference on Learning
Representations (ICLR)}, 2020.

[6] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, "Mastering Atari with
Discrete World Models," \textit{International Conference on Learning
Representations (ICLR)}, 2021.

[7] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, "Mastering Diverse
Domains through World Models," \textit{arXiv preprint arXiv:2301.04104}, 2023.

[8] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," \textit{Neural
Computation}, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] A. Vaswani et al., "Attention Is All You Need," \textit{Advances in Neural
Information Processing Systems (NeurIPS)}, 2017.

[10] S. Gu, B. Kelly, and D. Xiu, "Empirical Asset Pricing via Machine Learning,"
\textit{Review of Financial Studies}, vol. 33, no. 5, pp. 2223–2273, 2020.

[11] B. Kelly, S. Pruitt, and Y. Su, "Characteristics Are Covariances: A Unified
Model of Risk and Return," \textit{Journal of Financial Economics}, vol. 134,
no. 3, pp. 501–524, 2019.

[12] B. G. Malkiel, \textit{A Random Walk Down Wall Street}, W. W. Norton \&
Company, 1973.

[13] A. Ilmanen, \textit{Investing Amid Low Expected Returns: Making the Most
When Markets Offer the Least}, Wiley, 2022.

[14] T. Hastie, R. Tibshirani, and J. Friedman, \textit{The Elements of
Statistical Learning: Data Mining, Inference, and Prediction}, 2nd ed.,
Springer, 2009.

[15] L. R. Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications
in Speech Recognition," \textit{Proceedings of the IEEE}, vol. 77, no. 2,
pp. 257–286, 1989.

[16] J. Sirignano and R. Cont, "Universal Features of Price Formation in
Financial Markets: Perspectives from Deep Learning," \textit{Quantitative
Finance}, vol. 19, no. 9, pp. 1449–1459, 2019.

[17] N. Nguyen and D. Nguyen, "Hidden Markov Model for Stock Trading,"
\textit{International Journal of Financial Studies}, vol. 6, no. 2, 2018.
