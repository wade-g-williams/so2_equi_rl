"""DrQ-DQN. Averages the Bellman target over K shift-augmented copies of
next_obs and the TD loss over M shift-augmented copies of obs.

Paper sec E: random +/-4 pixel shift. Action is not augmented since pixel
shift doesn't change world-frame delta actions.
"""

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from so2_equi_rl.agents.dqn import DQNAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.dqn_drq import DQNDrQConfig
from so2_equi_rl.utils import augmentation as aug_mod
from so2_equi_rl.utils import tile_state

# Offset so aug RNG is decoupled from DQN-RAD (2023), SAC-DrQ (1337),
# RAD-SAC (2022), and FERM (3407).
_AUG_SEED_OFFSET = 4242


class DQNDrQAgent(DQNAgent):
    """DQN with DrQ shift averaging on both sides of the Bellman update."""

    def __init__(
        self,
        cfg: DQNDrQConfig,
        net_cls: Type[nn.Module],
    ) -> None:
        super().__init__(cfg, net_cls)

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
        obs_rep = obs.repeat(n_copies, 1, 1, 1).cpu()
        state_rep = state.repeat(n_copies, 1)
        obs_shifted = aug_mod.random_shift(
            obs_rep, pad=self.drq_pad, generator=self._aug_gen
        )
        return obs_shifted.to(state.device), state_rep

    def update(self, batch: Transition) -> Dict[str, float]:
        # K copies of next_obs for the target average, M copies of obs for
        # the TD loss average. Polyak at cfg.tau.
        batch = batch.to(self.device, non_blocking=True)

        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        B = batch.obs.shape[0]
        K = self.drq_k
        M = self.drq_m

        # Target side: K shifted copies of next_obs. Greedy over the full
        # action grid, so no action indexing needed.
        next_obs_k, next_state_k = self._shift_copies(
            batch.next_obs, batch.next_state, K
        )
        next_obs_k_tiled = tile_state(next_obs_k, next_state_k)

        with torch.no_grad():
            q_all_next_k = self.target_net(
                next_obs_k_tiled
            )  # (K*B, n_xy, n_z, n_theta, n_p)
            q_next_k = (
                q_all_next_k.reshape(K * B, -1).max(dim=1)[0].unsqueeze(-1)
            )  # (K*B, 1)

            reward_k = reward.repeat(K, 1)
            done_k = done.repeat(K, 1)
            y_k = reward_k + self.gamma * (1.0 - done_k) * q_next_k
            y = y_k.view(K, B, 1).mean(dim=0)  # (B, 1)

        # Current side: M shifted copies of obs. Action tiled M-fold to
        # index each copy's Q output.
        obs_m, state_m = self._shift_copies(batch.obs, batch.state, M)
        obs_m_tiled = tile_state(obs_m, state_m)
        action_m = batch.action.repeat(M, 1)

        q_all = self.policy_net(obs_m_tiled)  # (M*B, n_xy, n_z, n_theta, n_p)

        action_idx = action_m.long()
        p_id = action_idx[:, 0]
        xy_id = action_idx[:, 1]
        z_id = action_idx[:, 2]
        theta_id = action_idx[:, 3]
        mb_arange = torch.arange(M * B, device=self.device)
        q_pred = q_all[mb_arange, xy_id, z_id, theta_id, p_id].unsqueeze(-1)  # (M*B, 1)

        y_broadcast = y.repeat(M, 1)  # (M*B, 1)
        td_loss = F.smooth_l1_loss(q_pred, y_broadcast)
        td_error = (q_pred - y_broadcast).detach()

        self.optim.zero_grad()
        td_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optim.step()

        with torch.no_grad():
            for p, p_target in zip(
                self.policy_net.parameters(), self.target_net.parameters()
            ):
                p_target.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "td_loss": td_loss.item(),
            "td_error_mean": td_error.mean().item(),
            "q_mean": q_pred.mean().item(),
        }
