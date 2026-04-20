"""RAD-DQN. Subclass of DQNAgent that rotates (obs, next_obs) by a
shared per-row C_N angle and permutes xy_id accordingly, then runs the
vanilla DQN update on the augmented transition.

Only the xy_id column of the discrete action transforms under SO(2).
p_id, z_id, theta_id are invariant (vertical, grasp, and gripper-rotation
deltas don't see the planar rotation).
"""

import math
from typing import Dict, Type

import torch
import torch.nn as nn

from so2_equi_rl.agents.dqn import DQNAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.dqn_rad import DQNRADConfig
from so2_equi_rl.utils import augmentation as aug_mod

# Fixed offset on cfg.seed so the aug RNG is decoupled from DrQ (1337),
# RAD-SAC (2022), and FERM.
_AUG_SEED_OFFSET = 2023


class DQNRADAgent(DQNAgent):
    """DQN with one shared C_N rotation per replay row."""

    def __init__(
        self,
        cfg: DQNRADConfig,
        net_cls: Type[nn.Module],
    ) -> None:
        super().__init__(cfg, net_cls)

        self._rad_group_order = int(cfg.rad_group_order)

        # CPU generator. torch.randint would crash on a cuda generator in this torch pin.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

        # (N, n_xy) lookup on device so the gather in update() stays on GPU.
        self._xy_perm = self._build_xy_perm(self._rad_group_order).to(self.device)

    def _build_xy_perm(self, group_order: int) -> torch.Tensor:
        # For each k in 0..N-1, rotate the 9 xy-grid offsets by k*(2 pi / N)
        # via rotate_action_dxy, then argmin-snap each rotated cell back to
        # its nearest original grid cell. At N in {2, 4} the snap is exact.
        xy_offsets = self._xy_table  # (n_xy, 2), CPU float32
        n_xy = xy_offsets.shape[0]

        # 5-col pseudo-action since rotate_action_dxy expects (B, >=5).
        # Only cols 1-2 are read; other columns are passthrough.
        pseudo = torch.zeros(n_xy, 5, dtype=torch.float32)
        pseudo[:, 1:3] = xy_offsets

        perms = torch.zeros(group_order, n_xy, dtype=torch.long)
        for k in range(group_order):
            theta = torch.full(
                (n_xy,), k * (2.0 * math.pi / group_order), dtype=torch.float32
            )
            rotated = aug_mod.rotate_action_dxy(pseudo, theta)  # (n_xy, 5)
            rotated_dxy = rotated[:, 1:3]  # (n_xy, 2)
            # Squared-L2 argmin matches encode_action's joint xy snap.
            dist = (
                (rotated_dxy.unsqueeze(1) - xy_offsets.unsqueeze(0)).pow(2).sum(dim=-1)
            )
            perms[k] = dist.argmin(dim=1)
        return perms

    def update(self, batch: Transition) -> Dict[str, float]:
        batch = batch.to(self.device, non_blocking=True)

        B = batch.obs.shape[0]
        # Sample one discrete rotation index per row on CPU, then reuse for obs and xy-id.
        k_cpu = torch.randint(0, self._rad_group_order, (B,), generator=self._aug_gen)
        theta_cpu = k_cpu.to(torch.float32) * (2.0 * math.pi / self._rad_group_order)

        # rotate_obs moves theta onto obs.device internally.
        aug_obs = aug_mod.rotate_obs(batch.obs, theta_cpu)
        aug_next_obs = aug_mod.rotate_obs(batch.next_obs, theta_cpu)

        # Permute xy_id via the (N, n_xy) lookup. Other action columns
        # (p, z, theta) stay fixed, they're SO(2)-invariant.
        k_dev = k_cpu.to(self.device)
        xy_ids = batch.action[:, 1].long()
        new_xy_ids = self._xy_perm[k_dev, xy_ids].to(batch.action.dtype)
        aug_action = batch.action.clone()
        aug_action[:, 1] = new_xy_ids

        aug_batch = batch._replace(
            obs=aug_obs,
            next_obs=aug_next_obs,
            action=aug_action,
        )
        return super().update(aug_batch)
