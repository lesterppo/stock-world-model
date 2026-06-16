#!/usr/bin/env python3
"""
Stock World Model — Phase 3: Stress Testing Engine

Runs the trained world model + controller through out-of-sample stress scenarios:
  - Scenario A: 2020-03 style — extreme volatility spike, all-correlation crash
  - Scenario B: 2022 style — sustained rate hiking, growth-to-value rotation

Compares the latent-imagination-trained controller against baselines:
  1. Random policy
  2. Buy-and-hold (always long)
  3. Our PPO-trained controller

Metrics: Sharpe ratio, max drawdown, cumulative return, survival score.

Usage:
    python train_phase3.py \
        --phase1 checkpoints/phase1_final.pt \
        --phase2 checkpoints/phase2_final.pt
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MacroStockEncoder, StockDynamicsModel, RewardDecoder
from controller import TradingController
from losses import RiskAdjustedLoss
from data import (
    TECH_FEATURE_COLS,
    FUND_FEATURE_COLS,
    MACRO_ACTION_COLS,
    MICRO_ACTION_COLS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Stress Scenario Generator
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StressScenario:
    """Parameters defining a stress test regime."""
    name: str
    n_days: int
    price_drift: float          # daily mean return
    price_vol: float            # daily std return
    macro_us10y_mean: float
    macro_us10y_vol: float
    macro_yield_spread: float    # 10Y-2Y spread (negative = inversion)
    macro_vix_base: float
    macro_vix_spike_factor: float  # multiplier during crisis days
    micro_earnings_surprise_mean: float
    micro_earnings_surprise_vol: float
    earnings_day_prob: float
    description: str


# Realistic stress scenarios based on historical data
STRESS_SCENARIOS = {
    "covid_2020": StressScenario(
        name="COVID-19 Crash (2020-03)",
        n_days=120,  # enough for lookback + test steps
        price_drift=-0.008,         # -0.8%/day during panic
        price_vol=0.06,             # 6% daily vol (normal: ~1-2%)
        macro_us10y_mean=0.7,       # rates plummeted (flight to safety)
        macro_us10y_vol=0.05,
        macro_yield_spread=0.3,     # positive spread (both rates low)
        macro_vix_base=60.0,        # VIX at 60+
        macro_vix_spike_factor=1.5,  # spikes to 85 during worst days
        micro_earnings_surprise_mean=-0.08,  # earnings misses
        micro_earnings_surprise_vol=0.05,
        earnings_day_prob=0.05,
        description=(
            "COVID-19 panic: 4 circuit breakers in 2 weeks. "
            "All-correlation → 1.0 crash. VIX spikes to 85. "
            "Tests: does the model detect extreme uncertainty and go to cash?"
        ),
    ),
    "rate_hike_2022": StressScenario(
        name="Fed Rate Hiking Cycle (2022)",
        n_days=252,                   # full trading year
        price_drift=-0.001,           # -0.1%/day for growth stocks
        price_vol=0.025,              # 2.5% daily vol (elevated)
        macro_us10y_mean=3.5,         # rates rising to 4%+
        macro_us10y_vol=0.03,
        macro_yield_spread=-0.3,      # INVERTED: 2Y > 10Y
        macro_vix_base=28.0,          # elevated but not panic
        macro_vix_spike_factor=1.3,
        micro_earnings_surprise_mean=-0.02,  # slight misses (compression)
        micro_earnings_surprise_vol=0.03,
        earnings_day_prob=0.06,
        description=(
            "Sustained rate hiking: yield curve inverts, growth stocks "
            "lose 30-50%. Tests: does the model detect the macro gravity "
            "field and rotate away from high-P/E stocks?"
        ),
    ),
    "bull_market": StressScenario(
        name="Bull Market Baseline (control)",
        n_days=120,
        price_drift=+0.0015,          # +0.15%/day = ~38%/yr
        price_vol=0.012,              # normal vol
        macro_us10y_mean=2.0,
        macro_us10y_vol=0.01,
        macro_yield_spread=1.0,       # steep curve = expansion
        macro_vix_base=16.0,          # low VIX = complacency
        macro_vix_spike_factor=1.1,
        micro_earnings_surprise_mean=+0.02,
        micro_earnings_surprise_vol=0.02,
        earnings_day_prob=0.06,
        description="Bull market: steep yield curve, low VIX, positive earnings surprises.",
    ),
}


def generate_stress_data(
    scenario: StressScenario,
    tech_dim: int = 3,
    fund_dim: int = 2,
    macro_dim: int = 5,
    micro_dim: int = 2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic stress scenario as latent-space tensors.

    Returns:
        tech_seq:    [n_days, tech_dim] — technical features
        fund_vec:    [n_days, fund_dim] — fundamental features
        macro_acts:  [n_days, macro_dim] — macro actions
        micro_acts:  [n_days, micro_dim] — micro actions
    """
    rng = np.random.default_rng(seed)
    n = scenario.n_days

    # ── Price simulation ─────────────────────────────────────────────────
    price = 200.0
    opens, closes, volumes = [], [], []
    for i in range(n):
        # Inject VIX-correlated vol spikes
        day_vol = scenario.price_vol
        if rng.random() < 0.05:  # 5% chance of extreme day
            day_vol *= 2.5
        daily_ret = rng.normal(scenario.price_drift, day_vol)
        price *= (1.0 + daily_ret)
        closes.append(price)
        opens.append(price * rng.uniform(0.98, 1.02))
        volumes.append(rng.uniform(5e6, 2e8))

    tech_seq = np.column_stack([opens, closes, volumes]).astype(np.float32)

    # ── Fundamentals (slow-moving, PIT-aligned) ──────────────────────────
    roe = 0.20
    debt = 0.45
    fund_vec = np.zeros((n, fund_dim), dtype=np.float32)
    for i in range(n):
        if i % 30 == 0 and i > 0:  # quarterly update
            roe += rng.normal(scenario.micro_earnings_surprise_mean * 0.5, 0.02)
            debt += rng.normal(0.0, 0.02)
        fund_vec[i, 0] = roe
        fund_vec[i, 1] = debt

    # ── Macro actions ────────────────────────────────────────────────────
    macro_acts = np.zeros((n, macro_dim), dtype=np.float32)
    us10y = scenario.macro_us10y_mean
    vix = scenario.macro_vix_base
    for i in range(n):
        us10y += rng.normal(0.0, scenario.macro_us10y_vol)
        us10y = max(0.1, us10y)
        us2y = us10y - scenario.macro_yield_spread + rng.normal(0.0, 0.02)

        # VIX with occasional spikes
        if rng.random() < 0.03:  # 3% chance of spike day
            vix *= scenario.macro_vix_spike_factor
        vix += rng.normal(0.0, vix * 0.05)  # mean-reverting noise
        vix = max(10.0, vix)

        spread = us10y - us2y
        vix_change = rng.normal(0.0, 0.03)
        us10y_vol = scenario.macro_us10y_vol

        macro_acts[i] = [us10y, spread, vix, vix_change, us10y_vol]

    # ── Micro actions ────────────────────────────────────────────────────
    micro_acts = np.zeros((n, micro_dim), dtype=np.float32)
    for i in range(n):
        is_earnings = 1 if rng.random() < scenario.earnings_day_prob else 0
        surprise = (
            rng.normal(scenario.micro_earnings_surprise_mean, scenario.micro_earnings_surprise_vol)
            if is_earnings
            else 0.0
        )
        micro_acts[i] = [is_earnings, surprise]

    return (
        torch.tensor(tech_seq),
        torch.tensor(fund_vec),
        torch.tensor(macro_acts),
        torch.tensor(micro_acts),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Policy Functions
# ══════════════════════════════════════════════════════════════════════════════


class RandomPolicy:
    """Baseline: random actions ~ Uniform(-1, 1)."""
    def get_action(self, z_t, deterministic=False):
        B = z_t.shape[0]
        action = torch.empty(B, 1).uniform_(-1, 1)
        log_prob = torch.zeros(B, 1)  # not used
        value = torch.zeros(B, 1)
        return action, log_prob, value


class BuyAndHoldPolicy:
    """Baseline: always fully long (action = 1.0)."""
    def get_action(self, z_t, deterministic=False):
        B = z_t.shape[0]
        action = torch.ones(B, 1)
        log_prob = torch.zeros(B, 1)
        value = torch.zeros(B, 1)
        return action, log_prob, value


class ShortOnlyPolicy:
    """Baseline: always fully short (action = -1.0)."""
    def get_action(self, z_t, deterministic=False):
        B = z_t.shape[0]
        action = -torch.ones(B, 1)
        log_prob = torch.zeros(B, 1)
        value = torch.zeros(B, 1)
        return action, log_prob, value


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Runner
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    policy_name: str
    scenario_name: str
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    final_equity: float
    mean_position: float
    n_trades: int
    equity_curve: list[float] = field(default_factory=list)


@torch.no_grad()
def run_backtest(
    policy,
    v_encoder: MacroStockEncoder,
    m_dynamics: StockDynamicsModel,
    reward_decoder: RewardDecoder,
    tech_seq: torch.Tensor,
    fund_vec: torch.Tensor,
    macro_acts: torch.Tensor,
    micro_acts: torch.Tensor,
    lookback: int = 60,
    policy_name: str = "Policy",
    device: torch.device = torch.device("cpu"),
) -> BacktestResult:
    """
    Run a full backtest of a policy through latent imagination.

    Steps:
    1. Slide a lookback window across the stress data
    2. At each step: encode z_t → get action → compute reward → transition
    3. Track equity curve, compute metrics
    """
    n_days = len(tech_seq)
    if n_days <= lookback:
        raise ValueError(f"Need more than {lookback} days, got {n_days}")

    n_steps = n_days - lookback
    initial_equity = 1.0
    equity = initial_equity
    equity_curve = [equity]
    daily_returns = []

    for t in range(n_steps):
        # Current history window
        hist_start = t
        hist_end = t + lookback
        tech_window = tech_seq[hist_start:hist_end].unsqueeze(0).to(device)  # [1, L, tech_dim]
        fund_t = fund_vec[hist_end - 1].unsqueeze(0).to(device)              # [1, fund_dim]

        # Encode state
        _, _, z_t = v_encoder(tech_window, fund_t)  # [1, latent_dim]

        # Get action
        action, _, _ = policy.get_action(z_t, deterministic=True)
        position = action.item()

        # Get reward (predicted next-day return × position)
        pred_return = reward_decoder(z_t).item()
        daily_ret = position * pred_return
        daily_returns.append(daily_ret)

        # Update equity
        equity *= (1.0 + daily_ret)
        equity_curve.append(equity)

        # Transition (if we have next day's data)
        if t < n_steps - 1:
            macro_t = macro_acts[hist_end - 1].unsqueeze(0).to(device)
            micro_t = micro_acts[hist_end - 1].unsqueeze(0).to(device)
            z_t, _, _ = m_dynamics(z_t, macro_t, micro_t)

    # Compute metrics
    daily_returns = np.array(daily_returns)
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns) + 1e-8
    sharpe = mean_ret / std_ret * np.sqrt(252)  # annualized

    cum_return = equity - initial_equity

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - np.array(equity_curve)) / peak
    max_dd = np.max(drawdowns)

    return BacktestResult(
        policy_name=policy_name,
        scenario_name="",
        cumulative_return=cum_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        final_equity=equity,
        mean_position=float(action.mean()) if hasattr(action, 'mean') else 0.0,
        n_trades=n_steps,
        equity_curve=equity_curve,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Stress Testing Engine")
    parser.add_argument("--phase1", type=str, required=True, help="Phase 1 checkpoint")
    parser.add_argument("--phase2", type=str, required=True, help="Phase 2 controller checkpoint")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=30,
                        help="Lookback window for stress test (shorter than Phase 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Load Phase 1 ─────────────────────────────────────────────────────────
    ckpt1 = torch.load(args.phase1, map_location=device, weights_only=False)
    v_encoder = MacroStockEncoder(tech_dim=3, fund_dim=2, latent_dim=args.latent_dim).to(device)
    m_dynamics = StockDynamicsModel(latent_dim=args.latent_dim, macro_dim=5, micro_dim=2).to(device)
    reward_decoder = RewardDecoder(latent_dim=args.latent_dim).to(device)
    v_encoder.load_state_dict(ckpt1["v_encoder_state"])
    m_dynamics.load_state_dict(ckpt1["m_dynamics_state"])
    reward_decoder.load_state_dict(ckpt1["reward_decoder_state"])
    v_encoder.eval()
    m_dynamics.eval()
    reward_decoder.eval()
    print("Phase 1 loaded.")

    # ── Load Phase 2 Controller ──────────────────────────────────────────────
    ckpt2 = torch.load(args.phase2, map_location=device, weights_only=False)
    controller = TradingController(latent_dim=args.latent_dim).to(device)
    controller.load_state_dict(ckpt2["controller_state"])
    controller.eval()
    print(f"Phase 2 controller loaded (epoch {ckpt2.get('epoch', '?')}).")

    # ── Policies ─────────────────────────────────────────────────────────────
    policies = {
        "SWM Controller": controller,
        "Buy & Hold": BuyAndHoldPolicy(),
        "Always Short": ShortOnlyPolicy(),
        "Random": RandomPolicy(),
    }

    # ── Run stress tests ─────────────────────────────────────────────────────
    all_results: list[BacktestResult] = []

    print(f"\n{'='*70}")
    print(f"STRESS TESTING ENGINE")
    print(f"{'='*70}\n")

    for scenario_key, scenario in STRESS_SCENARIOS.items():
        print(f"Scenario: {scenario.name}")
        print(f"  {scenario.description[:100]}...")
        print(f"  Days: {scenario.n_days}, Price drift: {scenario.price_drift:+.4f}/day")

        # Generate data
        tech_seq, fund_vec, macro_acts, micro_acts = generate_stress_data(
            scenario,
            seed=args.seed + hash(scenario_key) % 10000,
        )
        print(f"  Generated {len(tech_seq)} days of stress data")

        # Run each policy
        for policy_name, policy in policies.items():
            result = run_backtest(
                policy, v_encoder, m_dynamics, reward_decoder,
                tech_seq, fund_vec, macro_acts, micro_acts,
                lookback=args.lookback,
                policy_name=policy_name,
                device=device,
            )
            result.scenario_name = scenario.name
            all_results.append(result)

        print()

    # ── Summary Table ────────────────────────────────────────────────────────
    print(f"{'='*90}")
    print(f"{'Policy':<20} {'Scenario':<30} {'Cum Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Final Eq':>10}")
    print(f"{'='*90}")

    for result in all_results:
        print(
            f"{result.policy_name:<20} "
            f"{result.scenario_name:<30} "
            f"{result.cumulative_return:>+10.4f} "
            f"{result.sharpe_ratio:>+8.2f} "
            f"{result.max_drawdown:>8.2%} "
            f"{result.final_equity:>10.4f}"
        )

    print(f"{'='*90}")

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("\n--- Analysis ---")

    # Group by scenario
    scenarios_seen = set()
    for result in all_results:
        if result.scenario_name not in scenarios_seen:
            scenarios_seen.add(result.scenario_name)
            scenario_results = [r for r in all_results if r.scenario_name == result.scenario_name]
            best = max(scenario_results, key=lambda r: r.sharpe_ratio)
            worst = min(scenario_results, key=lambda r: r.sharpe_ratio)
            print(f"\n{result.scenario_name}:")
            print(f"  Best:  {best.policy_name} (Sharpe: {best.sharpe_ratio:+.2f}, MaxDD: {best.max_drawdown:.2%})")
            print(f"  Worst: {worst.policy_name} (Sharpe: {worst.sharpe_ratio:+.2f}, MaxDD: {worst.max_drawdown:.2%})")

            # Did SWM beat buy-and-hold?
            swm = [r for r in scenario_results if r.policy_name == "SWM Controller"][0]
            bah = [r for r in scenario_results if r.policy_name == "Buy & Hold"][0]
            if swm.sharpe_ratio > bah.sharpe_ratio:
                print(f"  ✓ SWM Controller beats Buy & Hold by {swm.sharpe_ratio - bah.sharpe_ratio:+.2f} Sharpe")
            else:
                print(f"  ✗ SWM Controller trails Buy & Hold by {swm.sharpe_ratio - bah.sharpe_ratio:+.2f} Sharpe")

            if swm.max_drawdown < bah.max_drawdown:
                print(f"  ✓ SWM drawdown ({swm.max_drawdown:.2%}) < Buy & Hold ({bah.max_drawdown:.2%})")
            else:
                print(f"  ✗ SWM drawdown ({swm.max_drawdown:.2%}) >= Buy & Hold ({bah.max_drawdown:.2%})")

    # ── Overall ranking ──────────────────────────────────────────────────────
    print(f"\n--- Overall Ranking (avg Sharpe across scenarios) ---")
    policy_sharpes = {}
    for result in all_results:
        policy_sharpes.setdefault(result.policy_name, []).append(result.sharpe_ratio)
    for name, sharpes in sorted(policy_sharpes.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg = np.mean(sharpes)
        print(f"  {name:<20}: {avg:+.2f} avg Sharpe")

    print(f"\nStress testing complete.")


if __name__ == "__main__":
    main()
