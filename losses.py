"""
Stock World Model — Loss Functions

Phase 1 Losses:
  - TransitionLoss: KL divergence between M-Dynamics prediction and V-Encoder encoding.
  - RewardPredictionLoss: MSE between predicted and actual next-day return.

Phase 2/3 Losses:
  - RiskAdjustedLoss: trajectory-level Sharpe ratio + maximum drawdown penalty.

Phase 4 Losses:
  - InfoNCE: contrastive loss for cross-regime generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Self-Supervised World Model Learning
# ══════════════════════════════════════════════════════════════════════════════


def kl_divergence_gaussian(
    mu_pred: torch.Tensor,
    logvar_pred: torch.Tensor,
    mu_true: torch.Tensor,
    logvar_true: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence between two diagonal Gaussians.
    KL( N(mu_pred, sigma_pred) || N(mu_true, sigma_true) )
    """
    var_pred = torch.exp(logvar_pred)
    var_true = torch.exp(logvar_true)
    kl = 0.5 * (
        logvar_true
        - logvar_pred
        + (var_pred + (mu_pred - mu_true) ** 2) / var_true
        - 1.0
    )
    return kl.sum(dim=-1).mean()  # mean over batch, sum over latent dims


def transition_loss(
    z_pred_mu: torch.Tensor,
    z_pred_logvar: torch.Tensor,
    z_true_mu: torch.Tensor,
    z_true_logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Self-supervised transition loss: how well does M-Dynamics predict
    the next state distribution?

    Uses KL divergence between predicted and true posterior distributions.
    """
    return kl_divergence_gaussian(z_pred_mu, z_pred_logvar, z_true_mu, z_true_logvar)


def reward_prediction_loss(
    pred_return: torch.Tensor,
    true_return: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary reward prediction loss: MSE between predicted and actual
    next-day return. Anchors the latent space to have financial semantics.
    """
    return F.mse_loss(pred_return.squeeze(-1), true_return)


def phase1_loss(
    z_pred_mu: torch.Tensor,
    z_pred_logvar: torch.Tensor,
    z_true_mu: torch.Tensor,
    z_true_logvar: torch.Tensor,
    pred_return: torch.Tensor,
    true_return: torch.Tensor,
    kl_weight: float = 1.0,
    reward_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined Phase 1 loss: self-supervised world model training.

    Loss = kl_weight * KL(pred || true) + reward_weight * MSE(pred_return, true_return)

    Returns:
        total_loss: scalar
        metrics: dict of component losses for logging
    """
    kl = transition_loss(z_pred_mu, z_pred_logvar, z_true_mu, z_true_logvar)
    reward = reward_prediction_loss(pred_return, true_return)
    total = kl_weight * kl + reward_weight * reward
    metrics = {"kl_loss": kl.item(), "reward_loss": reward.item(), "total_loss": total.item()}
    return total, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2/3: Risk-Adjusted Trajectory Loss (for Controller training)
# ══════════════════════════════════════════════════════════════════════════════


class RiskAdjustedLoss(nn.Module):
    """
    Trajectory-level risk-adjusted loss for training the trading controller.
    Maximizes simulated Sharpe ratio while penalizing max drawdown.
    """

    def __init__(self, risk_free_rate: float = 0.0, mdd_lambda: float = 2.0):
        super().__init__()
        self.rf = risk_free_rate
        self.mdd_lambda = mdd_lambda

    def forward(self, simulated_returns: torch.Tensor):
        """
        Args:
            simulated_returns: [B, TrajectoryLength] — simulated daily returns
        Returns:
            total_loss: scalar
            sharpe: mean Sharpe ratio
            max_dd: mean max drawdown
        """
        # 1. Simulated Sharpe ratio
        mean_ret = torch.mean(simulated_returns, dim=-1)
        std_ret = torch.std(simulated_returns, dim=-1) + 1e-6
        sharpe = (mean_ret - self.rf) / std_ret

        # 2. Max drawdown on cumulative log-equity curve
        log_cum_equity = torch.cumsum(simulated_returns, dim=-1)
        running_max = torch.cummax(log_cum_equity, dim=-1)[0]
        drawdowns = running_max - log_cum_equity
        max_drawdown = torch.max(drawdowns, dim=-1)[0]

        # 3. Composite objective: maximize Sharpe, minimize drawdown
        total_loss = -torch.mean(sharpe) + self.mdd_lambda * torch.mean(max_drawdown)
        return total_loss, torch.mean(sharpe), torch.mean(max_drawdown)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Contrastive Loss (InfoNCE) for Cross-Regime Generalization
# ══════════════════════════════════════════════════════════════════════════════


def info_nce_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss (SimCLR-style).

    Given a batch of embeddings where consecutive pairs are positives
    (embeddings[0] & embeddings[1] are a positive pair, [2] & [3] are a
    positive pair, etc.), computes the normalized temperature-scaled
    cross-entropy loss.

    Positive pairs: embeddings from the same stock under similar macro
    regimes (temporally close, similar macro conditions).
    Negative pairs: all other samples in the batch (in-batch negatives).

    Args:
        embeddings: [2*B, ProjDim] — L2-normalized embeddings
                    where indices (2i, 2i+1) form a positive pair
        temperature: softmax temperature (lower = harder contrast)
    Returns:
        loss: scalar — average InfoNCE loss
    """
    # Compute cosine similarity matrix: [2*B, 2*B]
    sim_matrix = embeddings @ embeddings.T / temperature

    # Positive pairs: (2i, 2i+1) and (2i+1, 2i)
    N = embeddings.shape[0]
    labels = torch.arange(N, device=embeddings.device)
    # For sample i, its positive is i^1 (flip last bit)
    labels = labels ^ 1  # XOR with 1: 0→1, 1→0, 2→3, 3→2, ...

    # Mask out self-similarity
    mask = torch.eye(N, device=embeddings.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def build_contrastive_pairs(
    z_mu: torch.Tensor,
    macro_actions: torch.Tensor,
) -> torch.Tensor:
    """
    Build positive pairs using macro-regime similarity.

    For each sample in the batch, its positive pair is the other sample
    with the most similar macro conditions. This works correctly even
    when the DataLoader shuffles samples randomly.

    Args:
        z_mu:          [B, LatentDim] — encoded market states
        macro_actions: [B, MacroDim]  — macro conditions for each sample
    Returns:
        embeddings: [B, LatentDim] — reordered so that (2i, 2i+1) are
                    macro-similar positive pairs for InfoNCE
    """
    B = z_mu.shape[0]
    if B < 4:
        # Not enough for meaningful contrast — return as-is
        return z_mu

    # Normalize macro vectors for distance computation
    macro_norm = F.normalize(macro_actions, dim=-1)

    # Compute pairwise macro similarity: [B, B]
    macro_sim = macro_norm @ macro_norm.T

    # Build positive pairs greedily: for each unpaired sample,
    # find the most similar unpaired neighbor
    paired = torch.zeros(B, dtype=torch.bool, device=z_mu.device)
    pairs = []

    for i in range(B):
        if paired[i]:
            continue
        # Find most similar unpaired neighbor
        sims = macro_sim[i].clone()
        sims[paired] = float("-inf")
        sims[i] = float("-inf")  # exclude self
        j = sims.argmax().item()
        if not paired[j]:
            pairs.append((i, j))
            paired[i] = True
            paired[j] = True
        else:
            # Fallback: pair with any unpaired
            unpaired = (~paired).nonzero(as_tuple=True)[0]
            if len(unpaired) > 0:
                j = unpaired[0].item()
                pairs.append((i, j))
                paired[i] = True
                paired[j] = True

    # Interleave pairs: [a0, b0, a1, b1, ...]
    B_paired = len(pairs) * 2
    embeddings = torch.empty(B_paired, z_mu.shape[-1], device=z_mu.device)
    for idx, (i, j) in enumerate(pairs):
        embeddings[2 * idx] = z_mu[i]
        embeddings[2 * idx + 1] = z_mu[j]

    return embeddings


def contrastive_loss(
    z_mu: torch.Tensor,
    macro_actions: torch.Tensor,
    contrastive_head: nn.Module,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Full contrastive loss pipeline.

    1. Build macro-similarity-based positive pairs
    2. Pass through contrastive projection head
    3. Compute InfoNCE loss

    Returns:
        loss: scalar InfoNCE loss
    """
    embeddings = build_contrastive_pairs(z_mu, macro_actions)
    if embeddings.shape[0] < 4:
        return torch.tensor(0.0, device=z_mu.device)
    proj = contrastive_head(embeddings)
    return info_nce_loss(proj, temperature=temperature)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: RSSM Losses — KL annealing + free bits + RSSM training loss
# ══════════════════════════════════════════════════════════════════════════════


class KLAnnealer:
    """
    KL annealing schedule for RSSM training.

    Prevents posterior collapse by gradually increasing KL weight from 0 to 1.
    The model first learns good representations (low KL weight), then the KL
    penalty ramps up to enforce a meaningful stochastic bottleneck.

    Options:
      - monotonic: linear ramp from 0 → 1 over anneal_steps
      - cyclical:  repeated ramps (better for non-stationary data)

    Also supports "free bits": a minimum KL per latent dimension that must
    be satisfied before the KL penalty applies. This ensures the latent
    space actually carries information.
    """

    def __init__(
        self,
        anneal_steps: int = 5000,
        mode: str = "monotonic",
        free_bits: float = 0.5,
    ):
        self.anneal_steps = anneal_steps
        self.mode = mode
        self.free_bits = free_bits
        self.step = 0

    def __call__(self) -> float:
        """Return current KL weight."""
        if self.mode == "monotonic":
            weight = min(1.0, self.step / self.anneal_steps)
        elif self.mode == "cyclical":
            cycle = self.step % self.anneal_steps
            weight = min(1.0, cycle / (self.anneal_steps / 2))
        else:
            weight = 1.0
        self.step += 1
        return weight


def rssm_kl_loss(
    post_mu: torch.Tensor,
    post_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    """
    KL divergence for RSSM: KL(q(z_t | h_t, e_t) || p(z_t | h_t)).

    With free bits: the KL is only penalized above `free_bits` nats per dimension.
    This prevents the posterior from collapsing to the prior on dimensions
    that carry little information.

    Args:
        post_mu, post_logvar: posterior params [T, B, D] or [B, D]
        prior_mu, prior_logvar: prior params     [T, B, D] or [B, D]
        free_bits: minimum KL nats per dimension before penalty applies
    """
    var_post = torch.exp(post_logvar)
    var_prior = torch.exp(prior_logvar)

    kl_per_dim = 0.5 * (
        prior_logvar - post_logvar
        + (var_post + (post_mu - prior_mu) ** 2) / var_prior
        - 1.0
    )

    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    return kl_per_dim.sum(dim=-1).mean()  # average over batch & time


def rssm_reward_loss(
    pred_return: torch.Tensor,
    true_return: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and actual next-day return."""
    return F.mse_loss(pred_return.squeeze(-1), true_return)


def rssm_phase5_loss(
    rssm_out: dict,
    pred_return: torch.Tensor,
    true_return: torch.Tensor,
    annealer: KLAnnealer,
    reward_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined Phase 5 RSSM training loss.

    Loss = kl_weight * KL(q||p) + reward_weight * MSE(pred_return, true_return)

    Args:
        rssm_out: dict from RSSM.observe_rollout() with post_mu, post_logvar,
                  prior_mu, prior_logvar
        pred_return: [T, B, 1] predicted returns
        true_return: [T, B] true returns
        annealer: KLAnnealer instance
    Returns:
        total_loss, metrics dict
    """
    kl_weight = annealer()
    kl = rssm_kl_loss(
        rssm_out["post_mu"], rssm_out["post_logvar"],
        rssm_out["prior_mu"], rssm_out["prior_logvar"],
        free_bits=annealer.free_bits,
    )
    reward = rssm_reward_loss(pred_return, true_return)
    total = kl_weight * kl + reward_weight * reward
    metrics = {
        "kl_loss": kl.item(),
        "reward_loss": reward.item(),
        "total_loss": total.item(),
        "kl_weight": kl_weight,
    }
    return total, metrics
