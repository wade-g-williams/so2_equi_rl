"""FERM-SAC. Subclass of SACAgent with one shared CNN encoder (actor +
critic), a Polyak momentum key encoder, and a CURL-style InfoNCE loss
over two random-crop views (142x142 -> 128x128, pad=7). Shared encoder
trains from the critic TD loss and the InfoNCE loss; actor loss is
detached on both paths.

Two Polyak rates: cfg.tau (slow) for the critic target, cfg.curl_tau
(fast) for the key encoder. Key side is not bootstrapped, so it can
chase the online encoder harder.
"""

import copy
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_equi_rl.agents.sac import SACAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.sac_ferm import SACFERMConfig
from so2_equi_rl.networks import CNNActor, CNNCritic, CNNEncoder
from so2_equi_rl.utils import augmentation as aug_mod
from so2_equi_rl.utils import tile_state

# Fixed offset on cfg.seed so the FERM aug RNG doesn't overlap DrQ (1337) or RAD (2022).
_AUG_SEED_OFFSET = 3407


class SACFERMAgent(SACAgent):
    """Twin-Q SAC with a shared CNN encoder trained by both TD and InfoNCE."""

    def __init__(
        self,
        cfg: SACFERMConfig,
        encoder_cls: Type[nn.Module],
        actor_cls: Type[nn.Module],
        critic_cls: Type[nn.Module],
    ) -> None:
        # CNN-only. InfoNCE expects flat features, not GeometricTensor.
        # Fail fast so a wrong encoder_cls doesn't surface as a cryptic
        # shape error inside the bilinear product.
        if encoder_cls is not CNNEncoder:
            raise TypeError(
                f"SACFERMAgent is CNN-only; got encoder_cls={encoder_cls.__name__}"
            )
        if actor_cls is not CNNActor:
            raise TypeError(
                f"SACFERMAgent is CNN-only; got actor_cls={actor_cls.__name__}"
            )
        if critic_cls is not CNNCritic:
            raise TypeError(
                f"SACFERMAgent is CNN-only; got critic_cls={critic_cls.__name__}"
            )

        # Skip super().__init__(). Base SAC builds two encoders, FERM needs
        # one shared plus a momentum key. Reuse the scalar setup helpers.
        self._init_hyperparams(cfg)
        self._init_action_decoder(cfg)

        self.curl_tau = float(cfg.curl_tau)
        self.curl_lambda = float(cfg.curl_lambda)
        self.curl_temperature = float(cfg.curl_temperature)
        self.ferm_pad = int(cfg.ferm_pad)

        # Shared CNN encoder injected into both actor and critic. critic.parameters()
        # picks it up, so critic_optim already covers encoder grads from TD.
        enc_kwargs = {
            "obs_channels": cfg.obs_channels,
            "n_hidden": cfg.n_hidden,
            "group_order": cfg.group_order,
        }
        self.q_encoder = encoder_cls(**enc_kwargs).to(self.device)

        self.actor = actor_cls(
            encoder=self.q_encoder,
            action_dim=cfg.action_dim,
            detach_encoder=True,  # actor loss never reaches the shared encoder
        ).to(self.device)
        self.critic = critic_cls(
            encoder=self.q_encoder,
            action_dim=cfg.action_dim,
        ).to(self.device)

        # Bellman target. Polyak at cfg.tau matches base SAC.
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Momentum key encoder. Polyak at cfg.curl_tau (faster than the critic target).
        self.k_encoder = copy.deepcopy(self.q_encoder)
        for p in self.k_encoder.parameters():
            p.requires_grad_(False)

        # CURL bilinear projection. logits[i, j] = <q_i, W @ k_j>. Orthogonal
        # init keeps initial logits O(1) so cross-entropy doesn't saturate at step 0.
        self.W = nn.Parameter(
            torch.empty(
                self.q_encoder.output_dim,
                self.q_encoder.output_dim,
                device=self.device,
            )
        )
        nn.init.orthogonal_(self.W)

        self._init_alpha(cfg)

        # Actor optim covers heads only. Encoder lives in critic_optim and
        # encoder_optim; actor's detach zeros encoder grads on that path
        # anyway, so an Adam buffer for them would just be wasted memory.
        self._actor_head_params = [
            p
            for name, p in self.actor.named_parameters()
            if not name.startswith("encoder.")
        ]
        self.actor_optim = torch.optim.Adam(self._actor_head_params, lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        # Separate encoder_optim so InfoNCE and TD can run different LRs.
        # The two losses have different gradient scales.
        self.encoder_optim = torch.optim.Adam(
            list(self.q_encoder.parameters()) + [self.W],
            lr=cfg.encoder_lr,
        )

        # CPU generator. random_crop uses torch.randint which needs cpu.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        # Critic, actor, alpha, InfoNCE, then two Polyak averages.
        batch = batch.to(self.device, non_blocking=True)

        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        obs_tiled = tile_state(batch.obs, batch.state)
        next_obs_tiled = tile_state(batch.next_obs, batch.next_state)

        # Critic step on raw (non-augmented) obs. Encoder rotation-awareness
        # comes from the InfoNCE pretext below, not from aug'ing the Q-path.
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs_tiled)
            q1_next, q2_next = self.critic_target(next_obs_tiled, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            y = reward + self.gamma * (1.0 - done) * min_q_next

        # critic backward flows into q_encoder through the critic path. That's
        # one of the two shared-encoder training signals; InfoNCE is the other.
        q1, q2 = self.critic(obs_tiled, batch.action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        # Actor step. Two detaches keep actor-loss gradient off the shared
        # encoder: actor.detach_encoder cuts the policy path, critic(detach)
        # cuts the Q-path used to score new_action.
        new_action, log_prob, _ = self.actor.sample(obs_tiled)
        q1_new, q2_new = self.critic(obs_tiled, new_action, detach_encoder=True)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._actor_head_params, self.grad_clip_norm)
        self.actor_optim.step()

        # Alpha step.
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # InfoNCE step. Paper §E FERM: two independent random crops
        # (142x142 -> 128x128) for the query and key. Pulls (q, k) from the
        # same row together and pushes other pairings apart.
        B = batch.obs.shape[0]

        # random_crop uses a CPU generator; move raw obs to cpu for the
        # crop, then back to device for encoding.
        obs_cpu = batch.obs.cpu()
        q_obs = aug_mod.random_crop(
            obs_cpu, pad=self.ferm_pad, generator=self._aug_gen
        ).to(self.device)
        k_obs = aug_mod.random_crop(
            obs_cpu, pad=self.ferm_pad, generator=self._aug_gen
        ).to(self.device)
        q_obs_tiled = tile_state(q_obs, batch.state)
        k_obs_tiled = tile_state(k_obs, batch.state)

        # Key side is frozen via requires_grad=False; no_grad also drops key activations.
        q_feat = self.q_encoder(q_obs_tiled).view(B, -1)
        with torch.no_grad():
            k_feat = self.k_encoder(k_obs_tiled).view(B, -1)

        # Bilinear logits (B, B). Diagonal = positives, off-diagonal = negatives.
        # Row-max subtract for numerical stability, doesn't change cross-entropy.
        Wk = torch.matmul(self.W, k_feat.t())  # (d, B)
        logits = torch.matmul(q_feat, Wk) / self.curl_temperature  # (B, B)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = torch.arange(B, device=self.device)
        infonce_loss = F.cross_entropy(logits, labels)

        # zero_grad clears stale q_encoder grads from the critic backward earlier.
        self.encoder_optim.zero_grad()
        (self.curl_lambda * infonce_loss).backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.q_encoder.parameters()) + [self.W],
                self.grad_clip_norm,
            )
        self.encoder_optim.step()

        # Two Polyak updates. critic_target at cfg.tau, k_encoder at cfg.curl_tau.
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_target.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

            for p, p_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
                p_k.mul_(1.0 - self.curl_tau).add_(p.data, alpha=self.curl_tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "alpha": self.alpha.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        # q_encoder is saved separately even though actor/critic state_dicts
        # carry an "encoder.*" prefix pointing at the same weights, so the
        # checkpoint format stays stable if a refactor breaks that aliasing.
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "q_encoder": self.q_encoder.state_dict(),
            "k_encoder": self.k_encoder.state_dict(),
            "W": self.W.detach().cpu(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        # Actor and critic share q_encoder by reference; loading q_encoder
        # last pins the final shared weights.
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        self.q_encoder.load_state_dict(d["q_encoder"])
        self.k_encoder.load_state_dict(d["k_encoder"])
        with torch.no_grad():
            self.W.copy_(d["W"].to(self.device))
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        self.actor_optim.load_state_dict(d["actor_optim"])
        self.critic_optim.load_state_dict(d["critic_optim"])
        self.encoder_optim.load_state_dict(d["encoder_optim"])
        self.alpha_optim.load_state_dict(d["alpha_optim"])
