"""RAD-SAC. Subclass of SACAgent that rotates (obs, next_obs, action) by
one shared per-row theta and runs a vanilla SAC update on the rotated
transition. No K/M averaging unlike DrQ.
"""

from typing import Dict, Type

import torch
import torch.nn as nn

from so2_equi_rl.agents.sac import SACAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.sac_rad import SACRADConfig
from so2_equi_rl.utils import augmentation as aug_mod

# Fixed offset on cfg.seed so the aug RNG is decoupled from DrQ (1337) and FERM.
_AUG_SEED_OFFSET = 2022


class SACRADAgent(SACAgent):
    """Twin-Q SAC with one shared SO(2) rotation per transition."""

    def __init__(
        self,
        cfg: SACRADConfig,
        encoder_cls: Type[nn.Module],
        actor_cls: Type[nn.Module],
        critic_cls: Type[nn.Module],
    ) -> None:
        super().__init__(cfg, encoder_cls, actor_cls, critic_cls)

        self.rad_aug_mode = cfg.rad_aug_mode
        self.rad_group_order = int(
            cfg.rad_group_order if cfg.rad_group_order is not None else cfg.group_order
        )

        # CPU generator. sample_so2_angles uses torch.randint which would crash on a cuda generator.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        # Rotate (obs, next_obs, action) by one shared theta per row, hand off to base SAC.
        # state, reward, next_state, done are SO(2)-invariant so they pass through.
        batch = batch.to(self.device, non_blocking=True)

        B = batch.obs.shape[0]
        theta = aug_mod.sample_so2_angles(
            B,
            mode=self.rad_aug_mode,
            group_order=self.rad_group_order,
            generator=self._aug_gen,
        )

        aug_obs = aug_mod.rotate_obs(batch.obs, theta)
        aug_next_obs = aug_mod.rotate_obs(batch.next_obs, theta)
        aug_action = aug_mod.rotate_action_dxy(batch.action, theta)

        aug_batch = batch._replace(
            obs=aug_obs,
            next_obs=aug_next_obs,
            action=aug_action,
        )
        return super().update(aug_batch)
