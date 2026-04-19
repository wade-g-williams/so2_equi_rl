"""Twin-Q SAC with entropy tuning and a physical-unit action decoder.

encoder_cls/actor_cls/critic_cls are injected so one update() body covers
the equivariant runs and the CNN baseline.

Buffer stores unscaled [-1, 1] actions, decode_action only runs at the
env.step boundary. irrep(1) geometry is defined on the unscaled space.
"""

import copy
import math
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from so2_equi_rl.agents.base import Agent, ActionPair
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.sac import SACConfig
from so2_equi_rl.utils import tile_state


class SACAgent(Agent):
    """Twin-Q SAC with entropy tuning and a physical-unit action decoder."""

    def __init__(
        self,
        cfg: SACConfig,
        encoder_cls: Type[nn.Module],
        actor_cls: Type[nn.Module],
        critic_cls: Type[nn.Module],
    ) -> None:
        self._init_hyperparams(cfg)
        self._init_action_decoder(cfg)

        # Actor and critic each get their own encoder; twin-Qs share one per critic.
        enc_kwargs = {
            "obs_channels": cfg.obs_channels,
            "n_hidden": cfg.n_hidden,
            "group_order": cfg.group_order,  # ignored by CNNEncoder
        }
        actor_encoder = encoder_cls(**enc_kwargs)
        critic_encoder = encoder_cls(**enc_kwargs)

        self.actor = actor_cls(
            encoder=actor_encoder,
            action_dim=cfg.action_dim,
        ).to(self.device)
        self.critic = critic_cls(
            encoder=critic_encoder,
            action_dim=cfg.action_dim,
        ).to(self.device)

        # deepcopy after .to() so we don't rebuild the R2Conv kernel basis on the wrong device.
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self._init_alpha(cfg)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _init_hyperparams(self, cfg: SACConfig) -> None:
        # Auto-pick cuda when available so default construction doesn't crash on cpu-only boxes.
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.action_dim = cfg.action_dim
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.grad_clip_norm = cfg.grad_clip_norm

    def _init_action_decoder(self, cfg: SACConfig) -> None:
        # Pre-unpacked so decode/encode skip reindexing on every call.
        self.dpos = cfg.dpos
        self.drot = cfg.drot
        self.p_low = float(cfg.p_range[0])
        self.p_high = float(cfg.p_range[1])
        self.p_span = self.p_high - self.p_low

    def _init_alpha(self, cfg: SACConfig) -> None:
        # Learnable temperature in log-space so the optimizer sees an unconstrained scalar.
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(cfg.init_alpha), device=self.device)
        )
        self.target_entropy = (
            cfg.target_entropy
            if cfg.target_entropy is not None
            else -float(cfg.action_dim)
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def decode_action(self, unscaled: Tensor) -> Tensor:
        # [-1, 1] to physical units. p is asymmetric onto the gripper range, deltas are symmetric.
        p = self.p_low + 0.5 * (unscaled[:, 0:1] + 1.0) * self.p_span
        dxyz = unscaled[:, 1:4] * self.dpos
        dtheta = unscaled[:, 4:5] * self.drot
        return torch.cat([p, dxyz, dtheta], dim=1)

    def encode_action(self, physical: Tensor) -> Tensor:
        # Inverse of decode_action, used to push planner demos into the buffer.
        # Clamp guards against out-of-range planner commands breaking the [-1, 1] invariant.
        p = 2.0 * (physical[:, 0:1] - self.p_low) / self.p_span - 1.0
        dxyz = physical[:, 1:4] / self.dpos
        dtheta = physical[:, 4:5] / self.drot
        unscaled = torch.cat([p, dxyz, dtheta], dim=1)
        return unscaled.clamp(-1.0, 1.0)

    def select_action(
        self,
        state: Tensor,
        obs: Tensor,
        deterministic: bool = False,
    ) -> ActionPair:
        # deterministic=True picks tanh(mean) for eval rollouts.
        with torch.no_grad():
            state = state.to(self.device)
            obs = obs.to(self.device)
            tiled = tile_state(obs, state)
            if deterministic:
                _, _, mean_tanh = self.actor.sample(tiled)
                unscaled = mean_tanh
            else:
                unscaled, _, _ = self.actor.sample(tiled)
            physical = self.decode_action(unscaled)
        # Move to cpu before returning so env.step and the buffer don't pay a device sync.
        return ActionPair(unscaled=unscaled.cpu(), physical=physical.cpu())

    def update(self, batch: Transition) -> Dict[str, float]:
        # One SAC step: critic, actor, alpha, target.
        batch = batch.to(self.device, non_blocking=True)

        # Q-values are (B, 1); reward and done arrive as (B,) so broadcast up.
        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        obs_tiled = tile_state(batch.obs, batch.state)
        next_obs_tiled = tile_state(batch.next_obs, batch.next_state)

        # Bellman target.
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs_tiled)
            q1_next, q2_next = self.critic_target(next_obs_tiled, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            y = reward + self.gamma * (1.0 - done) * min_q_next

        # Critic step.
        q1, q2 = self.critic(obs_tiled, batch.action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        # Actor step. rsample for the reparameterization trick.
        new_action, log_prob, _ = self.actor.sample(obs_tiled)
        q1_new, q2_new = self.critic(obs_tiled, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optim.step()

        # Temperature step. Detach log_prob so the gradient only touches log_alpha.
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # No grad clip on log_alpha, vacuous on a 1-D tensor.
        self.alpha_optim.step()

        # Polyak target update.
        with torch.no_grad():
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_target.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        self.actor_optim.load_state_dict(d["actor_optim"])
        self.critic_optim.load_state_dict(d["critic_optim"])
        self.alpha_optim.load_state_dict(d["alpha_optim"])
