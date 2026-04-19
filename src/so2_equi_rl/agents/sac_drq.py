"""DrQ-SAC. Subclass of SACAgent that averages the Bellman target over K
augmented copies of next_obs and drives the critic and actor losses over
M augmented copies of obs.

Aug mode, K, M, and aug group order are paper-faithful (discrete C_N,
K = M = 2). Theta is independent per (copy, row), matching the paper's
row-by-row augmentTransition loop but fused into one (n*B, ...) tensor op
so the DrQ update is one rotate_obs call per side instead of 2*K+2*M
python-level rotations.
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from so2_equi_rl.agents.sac import SACAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.sac_drq import SACDrQConfig
from so2_equi_rl.utils import augmentation as aug_mod
from so2_equi_rl.utils import tile_state

# Fixed offset on top of cfg.seed so the aug RNG is decoupled from the
# global torch RNG (network init, env, buffer sampling) while still being
# reproducible from one int.
_AUG_SEED_OFFSET = 1337


class SACDrQAgent(SACAgent):
    """Twin-Q SAC with DrQ-style augmentation on both sides of the update."""

    def __init__(
        self,
        cfg: SACDrQConfig,
        encoder_cls: Type[nn.Module],
        actor_cls: Type[nn.Module],
        critic_cls: Type[nn.Module],
    ) -> None:
        super().__init__(cfg, encoder_cls, actor_cls, critic_cls)

        self.drq_k = int(cfg.drq_k)
        self.drq_m = int(cfg.drq_m)
        self.drq_aug_mode = cfg.drq_aug_mode
        # drq_group_order resolves to cfg.group_order in SACDrQConfig.__post_init__,
        # so None should not leak this far. Cast defensively.
        self.drq_group_order = int(
            cfg.drq_group_order if cfg.drq_group_order is not None else cfg.group_order
        )

        # Dedicated CPU Generator. augmentation.sample_so2_angles builds
        # theta via torch.randint/torch.rand without forwarding a device,
        # so its output is CPU; rotate_obs then moves theta to obs.device
        # for us. A cuda generator would crash on torch.randint. CPU is
        # also safer across torch versions and avoids one extra allocation.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def _augment_copies(
        self,
        obs: Tensor,
        state: Tensor,
        n_copies: int,
        action: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # Build n_copies independent augmented copies of (obs, state) and,
        # if action is given, the same copies of action rotated by the
        # matching theta. Fused into one tensor pass via .repeat + one
        # rotate_obs call, so we pay one GPU kernel instead of a loop.
        B = obs.shape[0]
        total = n_copies * B

        # .repeat(n, 1, 1, 1) tiles along dim 0 so row i of obs appears at
        # positions i, B + i, 2B + i, ... in the stacked tensor. State is
        # invariant under the rotation and just needs the same tiling.
        obs_rep = obs.repeat(n_copies, 1, 1, 1)
        state_rep = state.repeat(n_copies, 1)

        # One fresh theta per (copy, row), independent across copies to
        # match the paper's augmentTransition-per-d loop.
        theta = aug_mod.sample_so2_angles(
            total,
            mode=self.drq_aug_mode,
            group_order=self.drq_group_order,
            generator=self._aug_gen,
        )

        obs_rot = aug_mod.rotate_obs(obs_rep, theta)
        if action is None:
            return obs_rot, state_rep, None

        action_rep = action.repeat(n_copies, 1)
        action_rot = aug_mod.rotate_action_dxy(action_rep, theta)
        return obs_rot, state_rep, action_rot

    def update(self, batch: Transition) -> Dict[str, float]:
        # One DrQ-SAC update step: target averaging over K aug copies of
        # next_obs, critic + actor over M aug copies of obs. Alpha and
        # Polyak target update are identical to base SAC.
        batch = batch.to(self.device, non_blocking=True)

        B = batch.obs.shape[0]
        K = self.drq_k
        M = self.drq_m

        # Q-values are (B, 1); reward and done arrive as (B,) so broadcast up.
        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        # Target side: K augmented copies of next_obs. next_action is
        # sampled by the actor on the rotated obs, so no action rotation
        # is needed here.
        next_obs_k, next_state_k, _ = self._augment_copies(
            batch.next_obs, batch.next_state, K
        )
        next_obs_k_tiled = tile_state(next_obs_k, next_state_k)

        with torch.no_grad():
            next_action_k, next_log_prob_k, _ = self.actor.sample(next_obs_k_tiled)
            q1_next_k, q2_next_k = self.critic_target(next_obs_k_tiled, next_action_k)
            min_q_next_k = (
                torch.min(q1_next_k, q2_next_k) - self.alpha * next_log_prob_k
            )  # (K*B, 1)

            # Bellman target per copy, then average across K. Repeat the
            # scalar r, done tensors K-fold so the arithmetic is broadcast-free.
            reward_k = reward.repeat(K, 1)
            done_k = done.repeat(K, 1)
            y_k = reward_k + self.gamma * (1.0 - done_k) * min_q_next_k
            y = y_k.view(K, B, 1).mean(dim=0)  # (B, 1) averaged across K

        # Current side: M augmented copies of (obs, action). The same theta
        # rotates obs and the (dx, dy) columns of the unscaled action so
        # the (obs, action) pair stays on the SO(2)-equivariant manifold.
        obs_m, state_m, action_m = self._augment_copies(
            batch.obs, batch.state, M, action=batch.action
        )
        obs_m_tiled = tile_state(obs_m, state_m)

        # Critic step. Every M copy is scored against the same averaged y,
        # so broadcast y M-fold along dim 0.
        q1, q2 = self.critic(obs_m_tiled, action_m)  # (M*B, 1)
        y_broadcast = y.repeat(M, 1)
        critic_loss = F.mse_loss(q1, y_broadcast) + F.mse_loss(q2, y_broadcast)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        # Actor step. Re-sample on the same M-augmented obs so the policy
        # sees the same aug distribution the critic was just trained on.
        new_action, log_prob, _ = self.actor.sample(obs_m_tiled)
        q1_new, q2_new = self.critic(obs_m_tiled, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optim.step()

        # Temperature step. Detached log-prob so this gradient only touches
        # log_alpha, not the actor.
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Polyak average on the target critic, same schedule as base SAC.
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
