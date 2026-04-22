"""DrQ-SAC. Subclass of SACAgent that averages the Bellman target over K
shift-augmented copies of next_obs and the critic+actor losses over M
shift-augmented copies of obs. Per-row shift is independent across
(copy, row); copies are fused into one (n*B, ...) tensor op.

Paper §E + Kostrikov et al. 2020: DrQ uses random ±4 pixel shift.
Pixel shift doesn't change world-frame delta actions, so action is NOT
augmented.
"""

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from so2_equi_rl.agents.sac import SACAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.sac_drq import SACDrQConfig
from so2_equi_rl.utils import augmentation as aug_mod
from so2_equi_rl.utils import tile_state

# Fixed offset on cfg.seed so the aug RNG is decoupled from the global
# torch RNG and from RAD/FERM, but still reproducible from one int.
_AUG_SEED_OFFSET = 1337


class SACDrQAgent(SACAgent):
    """Twin-Q SAC with paper-faithful DrQ shift augmentation on both sides."""

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
        self.drq_pad = int(cfg.drq_pad)

        # CPU generator. random_shift uses torch.randint which needs cpu.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def _shift_copies(
        self,
        obs: Tensor,
        state: Tensor,
        n_copies: int,
    ) -> Tuple[Tensor, Tensor]:
        # n_copies independent shifted copies fused into one tensor op.
        # .repeat(n, 1, 1, 1) tiles along dim 0 so row i appears at positions
        # i, B+i, 2B+i, ...
        obs_rep = obs.repeat(n_copies, 1, 1, 1).cpu()
        state_rep = state.repeat(n_copies, 1)

        obs_shifted = aug_mod.random_shift(
            obs_rep, pad=self.drq_pad, generator=self._aug_gen
        )
        return obs_shifted.to(state.device), state_rep

    def update(self, batch: Transition) -> Dict[str, float]:
        # K shifted copies of next_obs for the target average, M shifted
        # copies of obs for the critic and actor losses. Alpha and Polyak
        # match base SAC. Action passes through unchanged, since pixel shift is
        # not an action-space transformation.
        batch = batch.to(self.device, non_blocking=True)

        B = batch.obs.shape[0]
        K = self.drq_k
        M = self.drq_m

        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        # Target side: K copies. next_action is sampled on the shifted obs.
        next_obs_k, next_state_k = self._shift_copies(
            batch.next_obs, batch.next_state, K
        )
        next_obs_k_tiled = tile_state(next_obs_k, next_state_k)

        with torch.no_grad():
            next_action_k, next_log_prob_k, _ = self.actor.sample(next_obs_k_tiled)
            q1_next_k, q2_next_k = self.critic_target(next_obs_k_tiled, next_action_k)
            min_q_next_k = (
                torch.min(q1_next_k, q2_next_k) - self.alpha * next_log_prob_k
            )  # (K*B, 1)

            reward_k = reward.repeat(K, 1)
            done_k = done.repeat(K, 1)
            y_k = reward_k + self.gamma * (1.0 - done_k) * min_q_next_k
            y = y_k.view(K, B, 1).mean(dim=0)

        # Current side: M shifted copies of obs. Action tiled M-fold unchanged.
        obs_m, state_m = self._shift_copies(batch.obs, batch.state, M)
        obs_m_tiled = tile_state(obs_m, state_m)
        action_m = batch.action.repeat(M, 1)

        # Critic step. Broadcast y M-fold so every copy hits the same averaged target.
        q1, q2 = self.critic(obs_m_tiled, action_m)  # (M*B, 1)
        y_broadcast = y.repeat(M, 1)
        critic_loss = F.mse_loss(q1, y_broadcast) + F.mse_loss(q2, y_broadcast)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        # Actor step. Same M-shifted obs so the policy sees the critic's distribution.
        new_action, log_prob, _ = self.actor.sample(obs_m_tiled)
        q1_new, q2_new = self.critic(obs_m_tiled, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optim.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

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
