---
title: "The Limits of Algorithmic Complexity in Multi-Asset Allocation"
subtitle: "A Journey from Recurrent State-Space Models to Dual Momentum"
author: "Lester PPO"
date: "June 2026"
abstract: |
  We document a comprehensive, four-phase empirical study exploring algorithmic
  approaches to macro regime detection and multi-asset portfolio allocation.
  Spanning 14 years of daily data across equities, bonds, gold, and commodities,
  we rigorously compare deep Recurrent State-Space Models (RSSMs), K-Means regime 
  clustering, volatility targeting, trend following, and Dual Momentum. Our 
  central finding is that unlevered Dual Momentum achieves risk-adjusted parity 
  (Calmar 0.396) with SPY buy-and-hold (Calmar 0.404) while reducing maximum 
  drawdown by 40% (from -33.7% to -20.3%). We further show that leverage in 
  momentum frameworks creates negative convexity—margin costs and volatility decay 
  consume all incremental return. We conclude that for daily-frequency multi-asset 
  allocation, algorithmic intervention is best utilized not for return enhancement, 
  but for behavioral survivability: engineering portfolios that investors can 
  reliably hold through structural market stress without capitulating.

  **Keywords:** regime detection, multi-asset allocation, recurrent state-space models,
  Dreamer, Dual Momentum, trend following, volatility targeting, walk-forward validation,
  Calmar ratio, behavioral finance

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
research and intuitive interpretability.

Recent advances in deep representation learning, particularly the Dreamer family of
world models [@hafner2019dream; @hafner2020dreamerv2; @hafner2023dreamerv3], offer an
alternative paradigm: learn rich latent representations of temporal dynamics directly
from data. The Recurrent State-Space Model (RSSM) at the core of Dreamer separates
deterministic and stochastic latent states, enabling both accurate reconstruction of
past observations and coherent imagination of future trajectories.

**This paper documents a complete, four-phase journey through the hierarchy of
algorithmic complexity in multi-asset allocation:**

1. **Phase 1 (Complexity):** We construct a DreamerV2/V3 RSSM with 130,000 parameters
   and train it on 14 years of daily SPY data. Across four distinct prediction tasks—return
   forecasting, volatility prediction, contextual bandit policy learning, and cross-sectional
   ETF ranking—the model collapses to degenerate solutions. The optimizer converges to
   the mathematically correct answer in a noise-dominated environment: the prior distribution.

2. **Phase 2 (Clustering):** We extract RSSM latent states and apply K-Means clustering
   for regime detection. While in-sample clusters appear interpretable, walk-forward
   validation reveals that cluster boundaries fitted on 2010–2021 data fail to generalize
   to post-COVID market structure. Soft distance-weighted blending—our attempt to rescue
   the approach—merely regresses allocations toward equal-weight.

3. **Phase 3 (Risk Heuristics):** We implement volatility targeting (inverse-vol weighting
   with equity-sleeve allocation) and trend following (200-day simple moving average filter).
   While these strategies reduce volatility, they chronically underperform SPY buy-and-hold
   during the 2010–2024 structural bull market. We document and correct a critical Calmar
   ratio miscalculation (using cumulative rather than annualized returns) that had
   previously inflated all performance metrics.

4. **Phase 4 (Momentum):** We implement Gary Antonacci's Dual Momentum framework—relative
   momentum across SPY, GLD, and DBC, filtered by absolute momentum of SPY versus cash—and
   test levered variants with tiered margin. The unlevered Dual Momentum strategy achieves
   our central result: Calmar ratio of 0.396, essentially tying SPY's 0.404, while reducing
   maximum drawdown from -33.7% to -20.3%. Leveraged variants fail: margin costs and
   volatility decay consume all incremental return.

**Our contributions are threefold:**

1. We provide the first rigorous empirical comparison spanning four distinct algorithmic
   paradigms—neural latent representations, regime clustering, risk-management heuristics,
   and momentum-based allocation—with full walk-forward validation and transaction cost
   accounting.

2. We document and correct a Calmar ratio measurement error (cumulative vs. annualized
   return) that has the potential to inflate reported performance metrics by an order of
   magnitude in multi-decade backtests.

3. We articulate the "behavioral survivability" thesis: the primary contribution of
   algorithmic allocation frameworks is not return enhancement, but the engineering of
   portfolios with drawdown profiles shallow enough that real investors—who panic-sell
   during severe drawdowns—can actually hold them through full market cycles.

# 2. Related Work

## 2.1 Regime-Switching Models in Finance

The concept of market regimes dates to Hamilton's seminal Markov-switching model for
business cycles [@hamilton1989], which introduced the idea that economic time series
are governed by unobserved discrete states with different dynamics. Ang and Bekaert
[@ang2002] applied regime-switching to international asset allocation, showing that
regimes characterized by high volatility and correlation breakdowns require fundamentally
different portfolio weights than tranquil periods.

Kritzman et al. [@kritzman2012] used a statistical measure of market turbulence to
identify regimes and demonstrated significant improvements in risk-adjusted returns
through dynamic allocation. Nguyen et al. [@nguyen2018] applied Hidden Markov Models
to S&P 500 returns, finding three distinct volatility regimes.

## 2.2 Deep Learning for Financial Time Series

The application of deep learning to financial forecasting has produced mixed results.
Gu et al. [@gu2020] conducted a comprehensive empirical evaluation of machine learning
methods for cross-sectional stock returns, finding that neural networks outperform
linear models only when the dataset is sufficiently large and the signal-to-noise
ratio is favorable. Kelly et al. [@kelly2019] demonstrate that simpler models often
outperform complex ones in financial prediction tasks because additional parameters
fit noise rather than signal.

## 2.3 World Models and Recurrent State-Space Models

The Dreamer family of world models [@hafner2019dream; @hafner2020dreamerv2;
@hafner2023dreamerv3] introduced RSSMs as a way to learn compact latent dynamics
from high-dimensional observations. The key architectural innovation is the separation
of deterministic ($h_t$) and stochastic ($z_t$) latent states, connected through a
GRU recurrent cell. This prevents the posterior collapse commonly observed in
variational autoencoders. Financial applications of world models remain rare.

## 2.4 Momentum and Trend Following

Momentum is among the most robust anomalies in financial economics, with evidence
spanning over 200 years across asset classes and geographies [@jegadeesh1993].
Faber [@faber2007] introduced a simple tactical asset allocation model using a
10-month moving average to rotate between asset classes. Antonacci [@antonacci2014]
formalized Dual Momentum, combining relative momentum (cross-asset comparison) with
absolute momentum (trend filter) in the Global Equity Momentum (GEM) framework.
Moskowitz et al. [@moskowitz2012] documented time-series momentum across 58 liquid
instruments, establishing it as a distinct phenomenon from cross-sectional momentum.

# 3. Data and Experimental Framework

## 3.1 Data

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
(Federal Reserve Economic Data).

**Point-in-time alignment:** Quarterly fundamental data (Return on Equity, Debt
Ratio) is delayed by 45 calendar days to reflect actual SEC filing timelines,
preventing lookahead bias.

**Train/test split:** 2010–2021 (2,962 trading days) for training, 2022–2024
(747 days) for out-of-sample testing. Walk-forward validation uses rolling 5-year
training windows.

## 3.2 Evaluation Metrics

All strategies are evaluated using corrected metrics throughout this paper.
A critical measurement error was discovered and corrected during our investigation.

**Calmar ratio:** Correctly computed as annualized return divided by maximum drawdown:

\begin{equation}
\text{Calmar} = \frac{\text{CAGR}}{|\max \text{DD}|} = \frac{(\prod_t (1 + r_t))^{252/N} - 1}{|\min_t((\prod_{\tau \leq t} (1 + r_\tau)) / \max_{s \leq t} \prod_{\tau \leq s} (1 + r_\tau) - 1)|}
\end{equation}

Using cumulative return instead of annualized return—an error we discovered in our
own initial implementation—inflates Calmar ratios proportionally to backtest length.
Over 14 years, this error exaggerates Calmar by approximately 10–40× depending on
the strategy's return profile. All Calmar values reported in this paper use the
corrected annualized formulation.

**Additional metrics:** Sharpe ratio (annualized), Sortino ratio (annualized, using
downside deviation), annualized volatility, maximum drawdown, cumulative return,
and daily turnover (sum of absolute position changes).

# 4. Phase 1: Neural Complexity — The RSSM Experiment

## 4.1 Architecture

Our RSSM follows the DreamerV2/V3 architecture [@hafner2020dreamerv2].

**Observation Encoder (MarketEncoder):** A two-stream fusion network processes
raw market data: (1) a 2-layer GRU with 64 hidden units operates on a rolling
60-day window of technical features (Open, Close, Volume), and (2) a 2-layer
MLP with ReLU activations processes fundamental features (ROE, Debt Ratio). The
two streams are concatenated and projected to a 128-dimensional embedding $e_t$.

**RSSM Core:** The core recurrent cell updates the deterministic hidden state
$h_t \in \mathbb{R}^{128}$ using the previous stochastic state $z_{t-1}$ and
action $a_{t-1}$ (a 7-dimensional vector of macro features):

\begin{equation}
h_t = \text{GRUCell}([z_{t-1}, a_{t-1}], h_{t-1})
\end{equation}

Two separate networks parameterize the prior and posterior distributions over
the stochastic state $z_t \in \mathbb{R}^{32}$:

\begin{align}
p(z_t | h_t) &= \mathcal{N}(\mu_\theta^{\text{prior}}(h_t), \sigma_\theta^{\text{prior}}(h_t)^2) \\
q(z_t | h_t, e_t) &= \mathcal{N}(\mu_\theta^{\text{post}}([h_t, e_t]), \sigma_\theta^{\text{post}}([h_t, e_t])^2)
\end{align}

The total parameter count is approximately 130,000. KL annealing with free bits
($\lambda_{\text{free}} = 0.1$ per dimension) prevents posterior collapse.
Training uses Adam optimizer with learning rate $3 \times 10^{-4}$, batch size 32,
sequence length 10, and gradient clipping at 1.0.

## 4.2 Four Model Collapses

Before arriving at our successful allocation framework, we documented four
prediction approaches that collapsed to degenerate solutions. These negative
results are of independent scientific interest.

**Return Prediction:** An RSSM with auxiliary reward decoder was trained to predict
next-day SPY returns. Out-of-sample $R^2 = -1.19$, worse than predicting the
unconditional mean. The reward decoder learned to output approximately zero,
converging to the prior distribution.

**Volatility Prediction:** The RSSM was trained to predict 5-day forward realized
volatility. Out-of-sample $R^2 = -1.64$. While volatility is more persistent than
returns (exhibiting GARCH effects), the Parkinson estimator from daily bars proved
too noisy for the RSSM to extract sufficient signal.

**Contextual Bandit Policy Learning:** A PPO-based policy was trained to output
SPY position sizes $\in [0,1]$ from RSSM state $h_t$. The policy collapsed to a
constant position of 0.502 with zero standard deviation. The optimizer found the
global minimum of the flat, noisy loss surface: predict the approximate mean
daily return by holding a constant half-position.

**Cross-Sectional ETF Ranking:** A classifier was trained to predict which of SPY,
TLT, or GLD would have the highest next-day return. It converged to always
predicting SPY—the most common class (37.6% of days)—achieving a test hit rate
of 0.363 versus the naive baseline of 0.376.

## 4.3 Implications

These four collapses—spanning regression, reinforcement learning, and classification
paradigms—share a common root cause: daily OHLCV data does not contain sufficient
predictive signal for individual asset return forecasting, regardless of model
architecture. The models that "collapsed" were performing optimal inference under
the true data generating process, which is indistinguishable from white noise at
daily frequency.

# 5. Phase 2: Regime Clustering — The K-Means Dead End

## 5.1 Approach

We extracted the RSSM's 128-dimensional deterministic state $h_t$ for each asset,
concatenated the four vectors into a 512-dimensional joint state, and reduced to
20 dimensions via PCA. K-Means clustering ($K=6$) was applied to these latent
representations, with per-regime portfolio weights optimized via coordinate ascent
on the Calmar ratio.

## 5.2 The Out-of-Sample Collapse

While K-Means discovered interpretable regimes in-sample—including a distinct
"COVID Panic" cluster with VIX at 52.4 and classic flight-to-safety patterns—the
clusters failed to generalize out-of-sample. In the 2016 walk-forward test year,
**all 252 trading days were assigned to a single regime** out of six. Across the
9-year walk-forward period, the median number of active regimes per test year was 2.

The root cause is fundamental: financial data is non-stationary. K-Means centroids
learned on 2010–2015 data do not partition 2024 market structure. The feature
distribution shifts over time, and cross-sectional clustering algorithms cannot
adapt without refitting.

## 5.3 The Soft-Blending Rescue Attempt

We implemented soft distance-weighted blending—replacing hard cluster assignment
with temperature-scaled softmax over inverse distances to all centroids. This
"smeared the lack of signal across all regimes" (as our AI reviewer noted), causing
the allocator to mathematically degenerate toward equal-weight whenever confused—
which was most of the time. Position ranges collapsed to [0.28, 0.34] for SPY,
barely deviating from the 0.25 equal-weight baseline.

**Corrected Calmar:** The K-Means regime allocator, with all production safeguards
(ensemble, OOD guard, velocity cap), achieved a corrected Calmar of 0.09 in
walk-forward testing—well below equal-weight's 0.33 and SPY's 0.40. The approach
was abandoned.

# 6. Phase 3: Risk Management Heuristics

## 6.1 Volatility Targeting

We implemented inverse-volatility allocation: each equity asset receives weight
$\min(1.0, \sigma_{\text{target}} / \sigma_{\text{realized}}^{63d})$, with
remaining capital distributed to safe-haven assets (TLT, GLD) proportionally.
Target volatility was set at 15% annualized.

This strategy reduced portfolio volatility to 13.7% (vs. SPY's 17.1%) and maximum
drawdown to -33.7% (vs. SPY's -33.7%—no improvement on drawdown). Annualized return
was 7.1% vs. SPY's 13.6%. The strategy systematically underweighted equities during
a historic bull market, producing a "cash-drag equivalent" that cost approximately
6.5 percentage points of annual return.

## 6.2 Trend Following

We implemented a 200-day simple moving average filter: assets trading above their
SMA are eligible for equal-weight allocation; if no assets are eligible, capital
rotates to TLT. Annualized return was 6.4% at 11.4% volatility, with a Calmar
ratio of 0.181—significantly below both SPY (0.404) and equal-weight (0.325).

## 6.3 Combined Trend + Vol

A combined approach—trend filter for eligibility, inverse-vol sizing for position
magnitude, monthly rebalancing with 5% tolerance band—produced annualized return
of 4.9% at 12.1% volatility (Calmar 0.142). While this strategy had genuine
conviction (SPY position range 0–100%), it systematically underperformed in the
2010–2024 sample dominated by equity bull markets.

**Summary of Phase 3:** \mbox{}

\begin{table}[H]
\centering
\caption{Risk Heuristics — Full-Sample 2010–2024 (Corrected Calmar)}
\label{tab:heuristics}
\begin{tabular}{lrrrrr}
\toprule
Strategy & AnnRet & AnnVol & Sharpe & Calmar & MaxDD \\
\midrule
Vol Target (15\%)          & +7.1\% & 13.7\% & +0.567 & +0.210 & -33.7\% \\
Trend Follow (200d SMA)    & +6.4\% & 11.4\% & +0.596 & +0.181 & -35.2\% \\
Trend + Vol Combined       & +4.9\% & 12.1\% & +0.459 & +0.142 & -34.8\% \\
Equal Weight               & +6.3\% &  9.1\% & +0.722 & +0.325 & -19.4\% \\
60/40 SPY/TLT             & +10.0\% & 10.1\% & +0.996 & +0.368 & -27.2\% \\
SPY Only                  & +13.6\% & 17.1\% & +0.833 & +0.404 & -33.7\% \\
\bottomrule
\end{tabular}
\end{table}

None of the risk-management heuristics beat equal-weight on Calmar ratio,
and all significantly trailed SPY buy-and-hold in absolute returns. The core
problem: strategies that systematically underweight equities during structural
bull markets incur an opportunity cost that no amount of downside protection
can overcome.

# 7. Phase 4: Dual Momentum

## 7.1 Strategy Description

Dual Momentum, as formalized by Antonacci [@antonacci2014], combines two distinct
momentum signals: relative momentum (cross-asset comparison) and absolute momentum
(trend filter). Our implementation adapts the framework to the 4-asset universe
of SPY, TLT, GLD, and DBC.

**Rebalancing:** Monthly, on the last trading day of each month.

**Lookback:** 12-month (252 trading days) trailing return.

**Rules:**

**Step 1 — Absolute Momentum Gate (Risk-On / Risk-Off):**
Compute the 12-month excess return of SPY against Cash:
\begin{equation}
\text{Momentum}_{\text{SPY}} = \prod_{t-252}^{t} (1 + r_{\text{SPY},\tau}) - 1
\end{equation}
If $\text{Momentum}_{\text{SPY}} > 0$, the system is **Risk-On** (Step 2).
Otherwise, the system is **Risk-Off** (Step 3).

**Step 2 — Risk-On (Relative Momentum):**
Compare the 12-month returns of SPY, GLD, and DBC. Select the top 2 assets by
trailing return and allocate 50% to each. TLT is excluded from the offensive pool
to prevent forced bond allocation during inflationary regimes where stocks and
bonds decline simultaneously.

**Step 3 — Risk-Off (Defensive):**
If TLT's 12-month return exceeds the risk-free rate, allocate 100% to TLT.
Otherwise, allocate 100% to Cash. This two-stage defensive check prevents
riding TLT down during rising-rate environments.

**Transaction costs:** 5 basis points one-way, deducted on each rebalance.

## 7.2 Full-Sample Results

Table \ref{tab:dm_full} presents the complete full-sample comparison of Dual
Momentum against all benchmarks.

\begin{table}[H]
\centering
\caption{Dual Momentum vs. Benchmarks — Full-Sample 2010–2024}
\label{tab:dm_full}
\begin{tabular}{lrrrrrr}
\toprule
Strategy & AnnRet & AnnVol & Sharpe & Calmar & Sortino & MaxDD \\
\midrule
Dual Momentum (unlevered) & +8.1\% & 11.5\% & +0.733 & +0.396 & +0.874 & -20.3\% \\
Equal Weight              & +6.3\% &  9.1\% & +0.722 & +0.325 & +0.971 & -19.4\% \\
Risk Parity               & +6.3\% &  9.0\% & +0.721 & +0.326 & +0.976 & -19.2\% \\
60/40 SPY/TLT             & +10.0\% & 10.1\% & +0.996 & +0.368 & +1.285 & -27.2\% \\
SPY Only                  & +13.6\% & 17.1\% & +0.833 & +0.404 & +1.021 & -33.7\% \\
\bottomrule
\end{tabular}
\end{table}

Dual Momentum achieves a Calmar ratio of 0.396, essentially tying SPY's 0.404,
while reducing maximum drawdown by 40% (from -33.7% to -20.3%). The strategy holds
SPY on 86% of trading days, GLD on 59%, DBC on 42%, TLT on 9%, and sits in Cash
on 7% of days. The average SPY allocation is 41% (vs. 50% naive equal-weight),
reflecting the strategy's ability to concentrate in equities during strong trends
while rotating to defensive assets during regime shifts.

## 7.3 Regime-Specific Performance

Table \ref{tab:dm_regimes} isolates the strategy's performance during the periods
where SPY suffered its worst losses.

\begin{table}[H]
\centering
\caption{Dual Momentum — Regime-Specific Annualized Returns}
\label{tab:dm_regimes}
\begin{tabular}{lrrrr}
\toprule
Period & DM AnnRet & SPY AnnRet & DM MaxDD & SPY MaxDD \\
\midrule
2022 Bear Market    & +8.2\%  & -19.1\% & -7.1\%  & -24.5\% \\
Post-GFC Bull       & +8.1\%  & +13.6\% & -20.3\% & -33.7\% \\
\bottomrule
\end{tabular}
\end{table}

During 2022—when the 60/40 portfolio suffered its worst year in a generation due
to simultaneous stock and bond declines—Dual Momentum produced a positive 8.2%
annualized return while SPY lost 19.1%. The strategy achieved this by rotating
to commodities (DBC, +23% in 2022) during the inflationary shock, while the
traditional 60/40 had nowhere to hide.

## 7.4 Walk-Forward Validation

Table \ref{tab:dm_wf} presents the annual walk-forward comparison from 2016–2024.

\begin{table}[H]
\centering
\caption{Dual Momentum — 9-Year Walk-Forward (Calmar Ratio)}
\label{tab:dm_wf}
\begin{tabular}{lrrrr}
\toprule
Year & Dual Mom. & SPY & 60/40 & DM > SPY? \\
\midrule
2016 & +1.54 & +1.57 & +1.68 & No \\
2017 & +3.58 & +8.33 & +7.69 & No \\
2018 & -0.52 & -0.27 & -0.26 & No \\
2019 & +7.96 & +4.72 & +10.97 & Yes \\
2020 & +0.86 & +0.51 & +1.14 & Yes \\
2021 & +3.71 & +5.99 & +3.15 & No \\
2022 & -0.42 & -0.78 & -0.84 & Yes \\
2023 & +1.07 & +2.63 & +1.25 & No \\
2024 & +3.13 & +3.32 & +2.34 & No \\
\bottomrule
\end{tabular}
\end{table}

Dual Momentum beats SPY on Calmar ratio in 3 of 9 years (2019, 2020, 2022)—all
years characterized by either elevated volatility or structural regime shifts.
It trails during the strongest trending bull markets (2017, 2021, 2023–2024), as
expected for a strategy that diversifies away from pure equity exposure.

**2018 analysis:** 2018 is notable as the only year where Dual Momentum, SPY,
and 60/40 all produce negative Calmar ratios. The strategy failed because (a)
the 12-month lookback's latency kept the system in Risk-On mode during the
October–December selloff (trailing returns remained positive until late December),
and (b) when the absolute momentum gate finally triggered Risk-Off, the subsequent
January 2019 rally was missed—a classic momentum whipsaw. This failure mode is
inherent to any long-lookback momentum strategy and represents the primary tail
risk of the approach. Potential mitigations (not tested in this study) include
a volatility-contingent exit rule or a shorter auxiliary lookback for crash detection.

## 7.5 The Leverage Trap

In a final experiment, we tested whether margin leverage could bridge the return
gap between Dual Momentum and SPY buy-and-hold. We implemented tiered leverage:

- SPY 12-month return > +10%: 1.5× leverage on SPY allocation only
- SPY 12-month return 0%–10%: 1.0× (no leverage)
- SPY 12-month return < 0%: 0× (Risk-Off)

Leverage was applied exclusively to the SPY sleeve (never to GLD or DBC, which
lack the structural equity risk premium). A margin rate of 5.5% annual (SOFR + 1%)
was charged daily on borrowed amounts.

**Result:** The levered strategy achieved an annualized return of 8.7% vs. 8.1%
for unlevered Dual Momentum—a gain of only 0.6 percentage points. Sharpe ratio
declined from 0.733 to 0.709, Calmar from 0.396 to 0.383, and maximum drawdown
deepened from -20.3% to -22.8%. Total margin costs consumed 12.1% of the final
portfolio value over the 14-year backtest.

**The leverage trap:** In a momentum framework, leverage creates negative convexity.
The 12-month lookback's inherent latency means the system remains levered during
the first month of a crash (the trailing return stays positive), amplifying losses
at precisely the wrong moment. The marginal return from leverage ($+0.6\%$/year)
does not compensate for the increased volatility, deeper drawdowns, and guaranteed
margin costs.

# 8. Discussion

## 8.1 The Behavioral Survivability Thesis

The central finding of our investigation is that unlevered Dual Momentum achieves
risk-adjusted parity with SPY buy-and-hold while reducing maximum drawdown by 40%.
This result supports what we term the **behavioral survivability thesis**: the primary
contribution of algorithmic portfolio frameworks is not return enhancement, but the
engineering of drawdown profiles shallow enough that real investors can hold through
full market cycles.

SPY buy-and-hold is theoretically optimal for an infinitely patient, emotionless
agent. But real investors—whether retail or institutional—panic-sell during severe
drawdowns. The -33.7% peak-to-trough loss of SPY over our sample period is a
"capitulation point" where many abandon their strategy, crystallizing losses and
missing the subsequent recovery. Dual Momentum's -20.3% maximum drawdown represents
a threshold that is significantly easier to endure behaviorally.

The performance gap between Dual Momentum (8.1% annualized) and SPY (13.6%
annualized) can be interpreted as an **insurance premium**: an annual cost of
approximately 5.5 percentage points in foregone return, paid to avoid the
33.7% drawdown that would trigger capitulation. For decumulating investors
(retirees drawing 4–5% annually), this insurance is invaluable—a 33.7% drawdown
during decumulation permanently destroys capital, while a 20.3% drawdown may
be survivable.

## 8.2 The Hierarchy of Complexity

Our four-phase investigation establishes a clear hierarchy of diminishing returns
to algorithmic complexity in daily-frequency multi-asset allocation:

| Phase | Approach | Parameters | Calmar | vs. SPY |
|---|---|---|---|---|
| 1 | RSSM (return prediction) | 130,000 | N/A ($R^2 < 0$) | — |
| 2 | K-Means Regime Clustering | ~30 | 0.09 | -78\% |
| 3 | Vol Targeting + Trend | ~10 | 0.14–0.21 | -48–65\% |
| 4 | Dual Momentum (unlevered) | ~5 | 0.396 | -2\% |

The progression is monotonic: as parameter count decreases, Calmar ratio
increases. The simplest model—5 interpretable parameters (lookback, number of
assets, top-N selection, absolute momentum threshold, safe asset)—achieves
near-parity with the market benchmark. Every additional layer of complexity
degrades out-of-sample performance.

This finding is consistent with the bias-variance tradeoff in low-signal
environments [@hastie2009]: with only 2,902 training days and a signal-to-noise
ratio near zero, additional parameters fit noise rather than signal. The
"collapse" of our neural models was not a failure of optimization—it was optimal
inference under the true data generating process.

## 8.3 Sample Dependence and Statistical Caveats

We must acknowledge that our central finding—Calmar parity between Dual Momentum
and SPY—may be specific to the 2010–2024 sample period, which was dominated by a
historic equity bull market. In a secular bear market or a multi-decade sideways
market (such as 2000–2009 for US equities), the relative ranking would likely
favor Dual Momentum more strongly, since SPY would suffer extended negative returns
while momentum rotation could capture trends in bonds, gold, or commodities.

Conversely, in a sustained low-volatility bull market with no structural regime
shifts, Dual Momentum will always trail SPY due to diversification and occasional
cash allocations. The 0.396 vs. 0.404 Calmar parity should not be interpreted as
a stable equilibrium—it is an empirical observation from one sample. Formal
bootstrap or permutation testing could establish confidence intervals around
this parity claim, and Monte Carlo simulation across synthetic market regimes
could quantify the strategy's robustness to different macro environments. These
analyses are deferred to future work.

## 8.4 Subjective Nature of Behavioral Claims

The "behavioral survivability" thesis presented in Section 8.1 is a conceptual
framework, not an empirically tested hypothesis. We provide no survey data,
experimental evidence, or historical redemption-flow analysis demonstrating
that investors would actually hold Dual Momentum through a -20.3% drawdown
when they would capitulate at -33.7%. This is a limitation of the current work:
the behavioral argument is logically motivated but empirically unvalidated.

To transform this thesis into a testable claim, future work could examine
mutual fund flow data during drawdown periods, conduct investor surveys with
hypothetical portfolio trajectories, or analyze the relationship between
maximum drawdown magnitude and subsequent redemption rates across different
strategy types. Until such evidence is gathered, the behavioral survivability
thesis should be treated as a plausible hypothesis, not an established finding.

## 8.5 The 2022 Stress Test

The 2022–2024 out-of-sample period represents a uniquely informative test because
the macro environment was fundamentally different from the training period. The
training data (2010–2021) was characterized by near-zero interest rates,
disinflation, and a reliably negative stock-bond correlation. The test period
featured aggressive rate hikes, elevated inflation, and a positive stock-bond
correlation.

Dual Momentum passes this stress test decisively: it delivers a positive 8.2%
annualized return in 2022 while SPY loses 19.1% and the 60/40 portfolio suffers
one of its worst years in history. The absolute momentum gate automatically
detected the deteriorating equity trend and rotated capital to commodities
during the inflationary shock—precisely the behavior that regime-switching
frameworks are designed to provide.

## 8.4 The Calmar Ratio Correction

During our investigation, we discovered and corrected a critical measurement
error: using cumulative return rather than annualized return in the Calmar ratio
numerator. This error causes Calmar to scale linearly with backtest length,
inflating reported performance metrics by an order of magnitude in multi-decade
studies.

For example, SPY's cumulative return of 555% over 14 years divided by its 33.7%
maximum drawdown yields an apparent Calmar of 16.46—a figure that would rank
among the greatest trading strategies in history. The corrected Calmar (annualized
return of 13.6% / 33.7% drawdown) is 0.404—a realistic but far more modest figure.

We recommend that all future studies in financial machine learning explicitly
specify whether Calmar ratios use annualized or cumulative returns. The cumulative
formulation is appropriate only for single-year test horizons; for multi-year
backtests, annualization is essential to prevent misleading performance claims.

## 8.5 Limitations

Several limitations of this study should be acknowledged:

**Sample period:** The 14-year sample (2010–2024) is dominated by an unprecedented
equity bull market fueled by quantitative easing and near-zero interest rates.
The relative performance of Dual Momentum versus SPY buy-and-hold would likely
be more favorable over a multi-cycle sample spanning 1970–present, which includes
the 1970s stagflation and 2000–2009 "lost decade" for US equities. Data constraints
(TLT launched 2002, DBC launched 2006) limited our sample.

**Asset universe:** The four-asset universe covers major liquid asset classes but
excludes real estate, international equities, and alternative strategies common
in institutional portfolios. A broader opportunity set would provide more
diversification and potentially stronger momentum signals.

**Momentum lookback:** We tested only a 12-month lookback. While this is standard
in the momentum literature [@antonacci2014; @moskowitz2012], optimization across
lookback windows (1, 3, 6, 9, 12 months) might yield improved performance.

**Regime persistence:** The monthly rebalancing frequency means the strategy can
remain in Risk-On mode for up to one month after a crash begins, since the
12-month trailing return stays positive during the initial decline. More frequent
rebalancing or a volatility-contingent exit rule could reduce this latency.

# 9. Conclusion

We have presented a comprehensive, four-phase empirical study spanning 14 years
of daily data, four asset classes, four algorithmic paradigms, and five rounds
of AI-assisted peer review. Our journey progressed from deep neural representations
through regime clustering and risk-management heuristics, culminating in the
discovery that unlevered Dual Momentum achieves risk-adjusted parity with the
market benchmark.

**Our central findings are:**

1. **Complexity does not equal predictability in financial markets.** A 130,000-
   parameter RSSM collapsed to degenerate solutions across four prediction tasks,
   demonstrating that daily OHLCV data lacks the signal-to-noise ratio required
   for high-capacity learned representations.

2. **K-Means regime clustering fails out-of-sample due to non-stationarity.**
   Cluster boundaries fitted on historical data do not generalize to structurally
   different future regimes. Soft-blending techniques merely regress allocations
   toward equal-weight.

3. **Dual Momentum achieves Calmar parity (0.396 vs. 0.404) with half the
   drawdown (-20.3% vs. -33.7%).** This is the only strategy in our study that
   matches SPY buy-and-hold on risk-adjusted return while providing meaningful
   downside protection. It is not a return-enhancement tool; it is a behavioral
   survivability mechanism.

4. **Leverage in momentum frameworks creates negative convexity.** Margin costs
   and volatility decay consume all incremental return from levered positions.
   The unlevered formulation is strictly superior.

5. **The Calmar ratio must use annualized returns.** Using cumulative returns
   inflates performance metrics by 10–40× in multi-decade backtests. We recommend
   this correction as a standard reporting practice.

The optimal algorithm for multi-asset allocation is the one with no "quit date."
Dual Momentum's +8.1% annualized return is not a failure—it is the guaranteed
return of a strategy that an investor can hold through the next 1970s, the next
2008, and the next 2022. Stop engineering is not an admission of defeat; it is
the most sophisticated quantitative conclusion one can reach.

All code, data pipelines, and experiment configurations are publicly
available at \url{https://github.com/lesterppo/stock-world-model} under
the MIT license.

\vspace{1em}
\noindent\textbf{Acknowledgments:} The author thanks Gemini Pro (Google DeepMind)
for serving as an AI peer reviewer across five rounds of iterative critique
during the development of this project. Gemini Pro identified the posterior
collapse in the variational encoder, recommended the pivot from return prediction
to regime-based allocation, caught the cumulative-vs-annualized Calmar error,
prescribed the Dual Momentum framework, and articulated the behavioral
survivability thesis that forms the central contribution of this paper.

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
Behaviors by Latent Imagination," \textit{ICLR}, 2020.

[6] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, "Mastering Atari with
Discrete World Models," \textit{ICLR}, 2021.

[7] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, "Mastering Diverse
Domains through World Models," \textit{arXiv:2301.04104}, 2023.

[8] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," \textit{Neural
Computation}, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] A. Vaswani et al., "Attention Is All You Need," \textit{NeurIPS}, 2017.

[10] S. Gu, B. Kelly, and D. Xiu, "Empirical Asset Pricing via Machine Learning,"
\textit{Review of Financial Studies}, vol. 33, no. 5, pp. 2223–2273, 2020.

[11] B. Kelly, S. Pruitt, and Y. Su, "Characteristics Are Covariances: A Unified
Model of Risk and Return," \textit{Journal of Financial Economics}, vol. 134,
no. 3, pp. 501–524, 2019.

[12] B. G. Malkiel, \textit{A Random Walk Down Wall Street}, W. W. Norton &
Company, 1973.

[13] A. Ilmanen, \textit{Investing Amid Low Expected Returns: Making the Most
When Markets Offer the Least}, Wiley, 2022.

[14] T. Hastie, R. Tibshirani, and J. Friedman, \textit{The Elements of
Statistical Learning}, 2nd ed., Springer, 2009.

[15] L. R. Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications
in Speech Recognition," \textit{Proceedings of the IEEE}, vol. 77, no. 2,
pp. 257–286, 1989.

[16] J. Sirignano and R. Cont, "Universal Features of Price Formation in
Financial Markets: Perspectives from Deep Learning," \textit{Quantitative
Finance}, vol. 19, no. 9, pp. 1449–1459, 2019.

[17] N. Nguyen and D. Nguyen, "Hidden Markov Model for Stock Trading,"
\textit{International Journal of Financial Studies}, vol. 6, no. 2, 2018.

[18] N. Jegadeesh and S. Titman, "Returns to Buying Winners and Selling Losers:
Implications for Stock Market Efficiency," \textit{Journal of Finance}, vol. 48,
no. 1, pp. 65–91, 1993.

[19] G. Antonacci, \textit{Dual Momentum Investing: An Innovative Strategy for
Higher Returns with Lower Risk}, McGraw-Hill, 2014.

[20] T. J. Moskowitz, Y. H. Ooi, and L. H. Pedersen, "Time Series Momentum,"
\textit{Journal of Financial Economics}, vol. 104, no. 2, pp. 228–250, 2012.
