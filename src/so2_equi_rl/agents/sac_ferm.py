"""FERM-SAC. Subclass of SACAgent that shares one CNN encoder between
actor and critic, pairs it with a momentum key encoder, and trains the
shared encoder jointly with the critic TD loss and a CURL-style InfoNCE
contrastive loss over two SO(2)-rotated views of the same obs.

Four network instances, two Polyak-averaged targets:
  q_encoder:       online shared encoder (actor + critic).
  k_encoder:       Polyak EMA of q_encoder at cfg.curl_tau.
  critic:          critic heads on top of q_encoder.
  critic_target:   deepcopy of critic, Polyak EMA at cfg.tau.

Two different tau rates: cfg.tau (slow) for the bootstrapped TD target,
cfg.curl_tau (fast) for the InfoNCE keys. The contrastive target isn't
bootstrapped, so the key encoder can chase the online encoder harder
without feedback blow-up.

Actor is built with detach_encoder=True, and the actor step calls
critic(..., detach_encoder=True), so the shared encoder never sees actor
loss on either path. It trains only from the critic TD loss and the
InfoNCE loss.

Aug is SO(2) rotations, two independent per-row thetas per batch (one
query view, one key view). The task has workspace rotation symmetry and
the rest of the codebase (DrQ, RAD) uses rotations too, so cross-variant
comparisons stay controlled.
"""

import copy
import math
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

# Fixed offset on top of cfg.seed so FERM's InfoNCE aug RNG doesn't
# overlap DrQ's (1337) or RAD's (2022). Distinct integer so two variants
# sharing cfg.seed don't replay the same theta sequence.
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
        # CNN-only validation: the contrastive loss operates on flat
        # feature vectors, not the GeometricTensor EquiEncoder produces.
        # Fail fast at construction so a wrong encoder_cls doesn't surface
        # as a cryptic shape error deep in the InfoNCE bilinear product.
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

        # Not calling super().__init__(). Base SAC builds two independent
        # encoders (one per actor, one per critic). FERM needs one shared
        # encoder plus a separate momentum key, so the network plumbing
        # differs end-to-end. Re-doing the shared SAC setup here is cleaner
        # than monkey-patching base SAC's layout. Phase 3's trainer refactor
        # is the right place to factor this out.

        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.action_dim = cfg.action_dim
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.grad_clip_norm = cfg.grad_clip_norm

        # Action decoder bounds, identical to base SAC. Pre-unpacked so
        # decode/encode don't reindex on every call.
        self.dpos = cfg.dpos
        self.drot = cfg.drot
        self.p_low = float(cfg.p_range[0])
        self.p_high = float(cfg.p_range[1])
        self.p_span = self.p_high - self.p_low

        # FERM knobs.
        self.curl_tau = float(cfg.curl_tau)
        self.curl_lambda = float(cfg.curl_lambda)
        self.curl_temperature = float(cfg.curl_temperature)
        self.ferm_aug_mode = cfg.ferm_aug_mode
        # ferm_group_order resolves to cfg.group_order in __post_init__,
        # so None should not leak this far. Cast defensively.
        self.ferm_group_order = int(
            cfg.ferm_group_order
            if cfg.ferm_group_order is not None
            else cfg.group_order
        )

        # One shared CNN encoder injected into both actor and critic.
        # critic.parameters() includes q_encoder, so critic_optim already
        # covers encoder gradients from the TD loss.
        enc_kwargs = {
            "obs_channels": cfg.obs_channels,
            "n_hidden": cfg.n_hidden,
            "group_order": cfg.group_order,  # CNNEncoder ignores; kept for kwarg uniformity
        }
        self.q_encoder = encoder_cls(**enc_kwargs).to(self.device)

        self.actor = actor_cls(
            encoder=self.q_encoder,
            action_dim=cfg.action_dim,
            detach_encoder=True,  # actor loss must never shape the shared encoder
        ).to(self.device)
        self.critic = critic_cls(
            encoder=self.q_encoder,
            action_dim=cfg.action_dim,
        ).to(self.device)

        # Bellman target: full deepcopy of critic (incl. its own encoder
        # replica). Polyak at cfg.tau matches base SAC target semantics.
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Momentum key encoder: separate deepcopy of q_encoder, Polyak at
        # cfg.curl_tau (faster than critic_target because InfoNCE isn't
        # bootstrapped). Frozen; only updated by Polyak in update().
        self.k_encoder = copy.deepcopy(self.q_encoder)
        for p in self.k_encoder.parameters():
            p.requires_grad_(False)

        # CURL bilinear projection. logits[i, j] = <q_i, W @ k_j>.
        # Square matrix over the encoder's flat feature dim. Orthogonal
        # init keeps initial logits O(1); a plain random init would
        # saturate cross-entropy at step 0.
        self.W = nn.Parameter(
            torch.empty(
                self.q_encoder.output_dim,
                self.q_encoder.output_dim,
                device=self.device,
            )
        )
        nn.init.orthogonal_(self.W)

        # SAC temperature (alpha) setup, identical to base SAC.
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(cfg.init_alpha), device=self.device)
        )
        self.target_entropy = (
            cfg.target_entropy
            if cfg.target_entropy is not None
            else -float(cfg.action_dim)
        )

        # Split actor params into heads only. The shared encoder is
        # already owned by critic_optim + encoder_optim, and the actor's
        # detach zeros encoder grads on that path anyway, so leaving
        # encoder params in actor_optim would just waste an Adam momentum
        # buffer on them. Saved as a list for grad clipping.
        self._actor_head_params = [
            p
            for name, p in self.actor.named_parameters()
            if not name.startswith("encoder.")
        ]
        self.actor_optim = torch.optim.Adam(self._actor_head_params, lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        # encoder_optim: shared encoder + CURL projection. Separate from
        # critic_optim so InfoNCE and TD can get different learning rates;
        # the two losses have different gradient scales and shouldn't
        # share a step size.
        self.encoder_optim = torch.optim.Adam(
            list(self.q_encoder.parameters()) + [self.W],
            lr=cfg.encoder_lr,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # Dedicated CPU Generator for the InfoNCE aug RNG. Same rationale
        # as DrQ/RAD: sample_so2_angles produces CPU tensors, rotate_obs
        # moves theta onto obs.device later. A cuda generator would crash
        # torch.randint. CPU also keeps this decoupled from the global
        # torch RNG that drives network init and the actor sampler.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        # One FERM-SAC update step: critic, actor, alpha, InfoNCE, then
        # two independent Polyak averages. Returns scalar losses/metrics.
        batch = batch.to(self.device, non_blocking=True)

        # Broadcast reward/done to (B, 1) so Q-value arithmetic is
        # shape-clean (Q-heads output (B, 1); raw buffer tensors are (B,)).
        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        # Tile scalar gripper state onto the heightmap for both timesteps.
        obs_tiled = tile_state(batch.obs, batch.state)
        next_obs_tiled = tile_state(batch.next_obs, batch.next_state)

        # Critic step. Standard SAC Bellman target on raw (non-augmented)
        # obs. The encoder's rotation-awareness comes from the InfoNCE
        # pretext task below, not from aug'ing the Q-path, so FERM's
        # contribution stays cleanly separable from RAD/DrQ in ablations.
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs_tiled)
            q1_next, q2_next = self.critic_target(next_obs_tiled, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            y = reward + self.gamma * (1.0 - done) * min_q_next

        # detach_encoder defaults to False here, so critic_loss.backward()
        # flows into q_encoder through the critic path. That's one of the
        # shared encoder's two training signals; InfoNCE is the other.
        q1, q2 = self.critic(obs_tiled, batch.action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        # Actor step. Two detaches together keep actor-loss gradient off
        # the shared encoder. actor.detach_encoder=True cuts the policy
        # path inside actor.sample, and critic(..., detach_encoder=True)
        # cuts the Q-path used to score new_action. actor_loss.backward()
        # then only updates the actor heads.
        new_action, log_prob, _ = self.actor.sample(obs_tiled)
        q1_new, q2_new = self.critic(obs_tiled, new_action, detach_encoder=True)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._actor_head_params, self.grad_clip_norm)
        self.actor_optim.step()

        # Alpha step. Detached log-prob so the gradient only touches
        # log_alpha. Identical to base SAC.
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # InfoNCE step. Two independent SO(2) rotations per row: theta_q
        # for the query view, theta_k for the key view. The contrastive
        # target pulls the (q, k) pair from the same row together and
        # pushes all other pairings apart, which shapes the encoder
        # toward rotation-invariance.
        B = batch.obs.shape[0]
        theta_q = aug_mod.sample_so2_angles(
            B,
            mode=self.ferm_aug_mode,
            group_order=self.ferm_group_order,
            generator=self._aug_gen,
        )
        theta_k = aug_mod.sample_so2_angles(
            B,
            mode=self.ferm_aug_mode,
            group_order=self.ferm_group_order,
            generator=self._aug_gen,
        )

        # Rotate raw obs first, then tile. Rotating the tiled tensor
        # would rotate the state plane too, which is SO(2)-invariant and
        # shouldn't be touched; pre-tile rotation keeps it clean.
        q_obs = aug_mod.rotate_obs(batch.obs, theta_q)
        k_obs = aug_mod.rotate_obs(batch.obs, theta_k)
        q_obs_tiled = tile_state(q_obs, batch.state)
        k_obs_tiled = tile_state(k_obs, batch.state)

        # q_feat trains q_encoder. k_feat is frozen: k_encoder params
        # have requires_grad=False, and no_grad here also drops the
        # key-side activations to save memory.
        q_feat = self.q_encoder(q_obs_tiled).view(B, -1)
        with torch.no_grad():
            k_feat = self.k_encoder(k_obs_tiled).view(B, -1)

        # Bilinear logits, shape (B, B). Diagonal entries are the
        # positive (i, i) pairs; off-diagonal are the B-1 negatives for
        # each row. Row-max subtract keeps softmax numerically stable;
        # it's a constant shift, doesn't change the cross-entropy.
        Wk = torch.matmul(self.W, k_feat.t())  # (d, B)
        logits = torch.matmul(q_feat, Wk) / self.curl_temperature  # (B, B)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = torch.arange(B, device=self.device)
        infonce_loss = F.cross_entropy(logits, labels)

        # encoder_optim.zero_grad() clears q_encoder grads (stale from
        # the critic backward earlier) and W grads (usually None here).
        # Clean grad state before the InfoNCE backward.
        self.encoder_optim.zero_grad()
        (self.curl_lambda * infonce_loss).backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.q_encoder.parameters()) + [self.W],
                self.grad_clip_norm,
            )
        self.encoder_optim.step()

        # Polyak updates. Two targets, two rates. critic_target at
        # cfg.tau (slow, standard SAC), k_encoder at cfg.curl_tau (fast,
        # contrastive EMA). Both run every step.
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
        # Snapshot everything trainable. q_encoder is saved separately
        # even though actor/critic state_dicts already carry an
        # "encoder.*" prefix pointing at the same weights. Saving it
        # explicitly keeps the checkpoint format robust if a later
        # refactor breaks that aliasing.
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
        # Actor and critic share q_encoder by reference, so the load
        # order doesn't matter. Loading q_encoder last pins the final
        # shared weights directly.
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
