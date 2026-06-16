"""
Stock World Model — Core Neural Architecture

Phase 1-4 (original):
  V-Encoder: Two-stream fusion of technical (GRU) + fundamental (MLP) features,
              with variational output for distributional state encoding.
  M-Dynamics: Dual-track conditional transition model predicting next latent state.
  Reward Decoder: Auxiliary head for next-day return prediction.
  ContrastiveHead: InfoNCE projection for cross-regime generalization.

Phase 5 (RSSM upgrade — Gemini Pro's fix for posterior collapse):
  RSSM: Recurrent State Space Model (DreamerV2/V3 architecture).
        Separates deterministic hidden state h_t from stochastic latent z_t
        to prevent the GRU from bypassing the information bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroStockEncoder(nn.Module):
    """
    Variational V-Encoder: encodes (technical + fundamental) → latent state z_t.
    Outputs both mu and logvar to support KL-divergence training against M-Dynamics.
    """
    def __init__(self, tech_dim: int, fund_dim: int, latent_dim: int = 64):
        super().__init__()
        # Technical stream: GRU over daily OHLCV + indicators
        self.tech_encoder = nn.GRU(
            input_size=tech_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        # Fundamental stream: MLP over sparse quarterly financials
        self.fund_encoder = nn.Sequential(
            nn.Linear(fund_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        # Fusion: concatenate then compress
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Variational head: outputs log-variance for the encoded state
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, tech_sequence: torch.Tensor, fund_vector: torch.Tensor):
        """
        Args:
            tech_sequence: [B, TimeSteps, TechDim] — daily candles + indicators
            fund_vector:   [B, FundDim]            — current-quarter financials (ffill'd)
        Returns:
            z_mu:     [B, LatentDim]
            z_logvar: [B, LatentDim]
            z_sample: [B, LatentDim] — reparameterized sample (for downstream use)
        """
        # Technical encoding
        _, h_n = self.tech_encoder(tech_sequence)
        z_tech = h_n[-1]  # [B, 64]

        # Fundamental encoding
        z_fund = self.fund_encoder(fund_vector)  # [B, 64]

        # Fusion
        z_combined = torch.cat([z_tech, z_fund], dim=-1)  # [B, 128]
        z_mu = self.fusion_layer(z_combined)              # [B, latent_dim]
        z_logvar = self.fc_logvar(z_mu)                    # [B, latent_dim]

        # Reparameterization trick
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z_sample = z_mu + eps * std

        return z_mu, z_logvar, z_sample


class StockDynamicsModel(nn.Module):
    """
    M-Dynamics: predicts next latent state z_{t+1} conditioned on
    current z_t and dual-track action vector (macro environment + micro shocks).
    """
    def __init__(self, latent_dim: int = 64, macro_dim: int = 5, micro_dim: int = 3):
        super().__init__()
        # Macro environment encoder (continuous systemic regime)
        self.macro_net = nn.Sequential(
            nn.Linear(macro_dim, 32),
            nn.ReLU(),
        )
        # Micro shock encoder (sparse impulse: earnings day, surprise %)
        self.micro_net = nn.Sequential(
            nn.Linear(micro_dim, 32),
            nn.ReLU(),
        )
        # Core transition network: z_t + macro_emb + micro_emb → next state
        input_dim = latent_dim + 32 + 32
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Output heads: Gaussian parameters
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        macro_action: torch.Tensor,
        micro_action: torch.Tensor,
    ):
        """
        Args:
            z_t:          [B, LatentDim] — current market state from V-Encoder
            macro_action: [B, MacroDim]  — systemic environment (rates, VIX, spread)
            micro_action: [B, MicroDim]  — idiosyncratic shocks (earnings surprise)
        Returns:
            z_next:  [B, LatentDim] — sampled next state
            mu:      [B, LatentDim]
            logvar:  [B, LatentDim]
        """
        e_macro = self.macro_net(macro_action)   # [B, 32]
        e_micro = self.micro_net(micro_action)     # [B, 32]
        combined = torch.cat([z_t, e_macro, e_micro], dim=-1)
        x = self.transition_net(combined)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mu + eps * std

        return z_next, mu, logvar


class RewardDecoder(nn.Module):
    """
    Auxiliary decoder: predicts next-day return from latent state z_t.
    Anchors the latent space to have financial meaning (not just arbitrary compression).
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # scalar: predicted next-day return
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: [B, LatentDim] — encoded market state
        Returns:
            pred_return: [B, 1] — predicted next-day return
        """
        return self.net(z_t)


class ContrastiveHead(nn.Module):
    """
    Phase 4: InfoNCE projection head for V-Encoder.

    Maps latent states z_t to an L2-normalized embedding space where
    contrastive learning pulls together states from similar macro regimes
    and pushes apart states from different regimes.

    Architecture follows SimCLR: 2-layer MLP with BatchNorm and ReLU.
    Contrastive loss is computed on the normalized output.
    """

    def __init__(self, latent_dim: int = 64, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: [B, LatentDim] — encoded state (use z_mu, not z_sample)
        Returns:
            proj: [B, ProjDim] — L2-normalized contrastive embedding
        """
        proj = self.net(z_t)
        return F.normalize(proj, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: RSSM (Recurrent State Space Model) — DreamerV2/V3 Architecture
# ══════════════════════════════════════════════════════════════════════════════


class MarketEncoder(nn.Module):
    """
    Observation encoder: compresses raw market features into embedding e_t.

    This is the same two-stream fusion as the old MacroStockEncoder,
    but without the variational head. The RSSM handles the stochastic
    component separately.
    """
    def __init__(self, tech_dim: int, fund_dim: int, embed_dim: int = 128):
        super().__init__()
        self.tech_encoder = nn.GRU(
            input_size=tech_dim, hidden_size=64,
            num_layers=2, batch_first=True,
        )
        self.fund_encoder = nn.Sequential(
            nn.Linear(fund_dim, 32), nn.ReLU(), nn.Linear(32, 64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tech_seq: torch.Tensor, fund_vec: torch.Tensor) -> torch.Tensor:
        """
        Returns e_t: [B, embed_dim] — encoded observation embedding
        """
        _, h_n = self.tech_encoder(tech_seq)
        z_tech = h_n[-1]
        z_fund = self.fund_encoder(fund_vec)
        return self.fusion(torch.cat([z_tech, z_fund], dim=-1))


class RSSM(nn.Module):
    """
    Recurrent State Space Model (DreamerV2/V3-style).

    Key innovation: separates deterministic hidden state h_t from
    stochastic latent state z_t. This prevents the GRU from finding
    a deterministic shortcut (posterior collapse).

    Components:
      - Recurrent:  h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
      - Prior:      p(z_t | h_t)        → μ_p, log σ²_p
      - Posterior:  q(z_t | h_t, e_t)   → μ_q, log σ²_q

    During training: use posterior q(z_t | h_t, e_t)  (has observation)
    During imagination: use prior p(z_t | h_t)         (no observation)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        action_dim: int = 7,    # macro(5) + micro(2)
        hidden_dim: int = 128,
        latent_dim: int = 64,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        # Deterministic recurrent cell
        rnn_input_dim = latent_dim + action_dim  # z_{t-1} + a_{t-1}
        self.rnn = nn.GRUCell(input_size=rnn_input_dim, hidden_size=hidden_dim)

        # Prior: p(z_t | h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        # Posterior: q(z_t | h_t, e_t)
        posterior_input_dim = hidden_dim + embed_dim
        self.posterior_fc = nn.Sequential(
            nn.Linear(posterior_input_dim, hidden_dim), nn.ReLU(),
        )
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar).clamp(min=self.min_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def initial_state(self, batch_size: int, device: torch.device):
        """Initialize h_0 and z_0 as zeros."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def observe_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action_prev: torch.Tensor,
        e_t: torch.Tensor,
    ) -> dict:
        """
        One RSSM step with observation (training mode).

        Args:
            h_prev:      [B, hidden_dim] — previous deterministic state
            z_prev:      [B, latent_dim] — previous stochastic state
            action_prev: [B, action_dim] — previous action (macro+micro)
            e_t:         [B, embed_dim]  — current observation embedding
        Returns dict with:
            h_t, z_t: new states
            prior_mu, prior_logvar: p(z_t | h_t)
            post_mu, post_logvar:  q(z_t | h_t, e_t)
        """
        # Deterministic update
        rnn_input = torch.cat([z_prev, action_prev], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)

        # Prior (no observation)
        p_feat = self.prior_fc(h_t)
        prior_mu = self.prior_mu(p_feat)
        prior_logvar = self.prior_logvar(p_feat)

        # Posterior (with observation)
        post_input = torch.cat([h_t, e_t], dim=-1)
        q_feat = self.posterior_fc(post_input)
        post_mu = self.posterior_mu(q_feat)
        post_logvar = self.posterior_logvar(q_feat)

        # Sample from posterior
        z_t = self._sample(post_mu, post_logvar)

        return {
            "h_t": h_t,
            "z_t": z_t,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "post_mu": post_mu,
            "post_logvar": post_logvar,
        }

    def imagine_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One RSSM step WITHOUT observation (imagination mode).

        Uses the prior p(z_t | h_t) since no observation is available.
        """
        rnn_input = torch.cat([z_prev, action_prev], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)
        p_feat = self.prior_fc(h_t)
        prior_mu = self.prior_mu(p_feat)
        prior_logvar = self.prior_logvar(p_feat)
        z_t = self._sample(prior_mu, prior_logvar)
        return h_t, z_t

    def observe_rollout(
        self,
        e_seq: torch.Tensor,
        a_seq: torch.Tensor,
        h_0: torch.Tensor = None,
        z_0: torch.Tensor = None,
    ) -> dict:
        """
        Run RSSM over a full sequence with observations.

        Args:
            e_seq: [T, B, embed_dim]  — encoded observations
            a_seq: [T, B, action_dim] — actions (t-1 to t for each step)
            h_0:   [B, hidden_dim]    — initial hidden (default: zeros)
            z_0:   [B, latent_dim]    — initial latent  (default: zeros)
        Returns:
            dict with stacked tensors [T, B, ...]
        """
        T, B, _ = e_seq.shape
        if h_0 is None:
            h_0 = torch.zeros(B, self.hidden_dim, device=e_seq.device)
        if z_0 is None:
            z_0 = torch.zeros(B, self.latent_dim, device=e_seq.device)

        h_list, z_list = [], []
        prior_mu_list, prior_logvar_list = [], []
        post_mu_list, post_logvar_list = [], []

        h_t, z_t = h_0, z_0
        for t in range(T):
            out = self.observe_step(
                h_t, z_t,
                a_seq[t],   # action at step t leads to state at t+1
                e_seq[t],   # observation at step t
            )
            h_t, z_t = out["h_t"], out["z_t"]
            h_list.append(h_t)
            z_list.append(z_t)
            prior_mu_list.append(out["prior_mu"])
            prior_logvar_list.append(out["prior_logvar"])
            post_mu_list.append(out["post_mu"])
            post_logvar_list.append(out["post_logvar"])

        return {
            "h": torch.stack(h_list),
            "z": torch.stack(z_list),
            "prior_mu": torch.stack(prior_mu_list),
            "prior_logvar": torch.stack(prior_logvar_list),
            "post_mu": torch.stack(post_mu_list),
            "post_logvar": torch.stack(post_logvar_list),
        }

    def imagine_rollout(
        self,
        h_0: torch.Tensor,
        z_0: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run RSSM imagination (no observations).

        Args:
            h_0:   [B, hidden_dim]
            z_0:   [B, latent_dim]
            a_seq: [T, B, action_dim]
        Returns:
            h_seq: [T, B, hidden_dim]
            z_seq: [T, B, latent_dim]
        """
        T = a_seq.shape[0]
        h_list, z_list = [], []
        h_t, z_t = h_0, z_0
        for t in range(T):
            h_t, z_t = self.imagine_step(h_t, z_t, a_seq[t])
            h_list.append(h_t)
            z_list.append(z_t)
        return torch.stack(h_list), torch.stack(z_list)


class RSSMRewardDecoder(nn.Module):
    """
    Reward decoder for RSSM: predicts next-day return from (h_t, z_t).
    Uses BOTH deterministic and stochastic state components.
    """
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))


class EnsembleRSSM(nn.Module):
    """
    Ensemble of RSSM models for epistemic uncertainty quantification.

    Trains N independent RSSM instances. During imagination, disagreement
    among ensemble members indicates epistemic uncertainty (model doesn't
    know the true dynamics), which the controller can use to reduce risk.
    """
    def __init__(
        self,
        n_models: int = 3,
        embed_dim: int = 128,
        action_dim: int = 7,
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            RSSM(embed_dim, action_dim, hidden_dim, latent_dim)
            for _ in range(n_models)
        ])

    def imagine_rollout(
        self,
        h_0: torch.Tensor,
        z_0: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_mean: [T, B, hidden]
            z_mean: [T, B, latent]
            z_var:  [T, B, latent] — variance ACROSS ensemble (epistemic)
        """
        T, B = a_seq.shape[0], a_seq.shape[1]
        all_z = []
        for model in self.models:
            _, z_seq = model.imagine_rollout(h_0, z_0, a_seq)
            all_z.append(z_seq.unsqueeze(0))  # [1, T, B, latent]

        all_z = torch.cat(all_z, dim=0)       # [N, T, B, latent]
        z_mean = all_z.mean(dim=0)            # [T, B, latent]
        z_var = all_z.var(dim=0)              # [T, B, latent]

        # h from first model (representative)
        h_seq, _ = self.models[0].imagine_rollout(h_0, z_0, a_seq)
        return h_seq, z_mean, z_var

    def epistemic_uncertainty(
        self,
        h_0: torch.Tensor,
        z_0: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Returns scalar epistemic uncertainty per step [T, B]."""
        _, _, z_var = self.imagine_rollout(h_0, z_0, a_seq)
        return z_var.mean(dim=-1)  # [T, B]
