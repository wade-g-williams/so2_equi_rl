"""RAD-DQN. Random crops (obs, next_obs) and runs the vanilla DQN update
on the augmented transition.

Paper sec E: random crop (142x142 -> 128x128). Action is not augmented
since pixel translation doesn't change world-frame delta actions.
"""

from typing import Dict, Type

import torch
import torch.nn as nn

from so2_equi_rl.agents.dqn import DQNAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.dqn_rad import DQNRADConfig
from so2_equi_rl.utils import augmentation as aug_mod

# Offset on cfg.seed so aug RNG is decoupled from DrQ (1337), RAD-SAC (2022), and FERM.
_AUG_SEED_OFFSET = 2023


class DQNRADAgent(DQNAgent):
    """DQN with random-crop augmentation."""

    def __init__(
        self,
        cfg: DQNRADConfig,
        net_cls: Type[nn.Module],
    ) -> None:
        super().__init__(cfg, net_cls)

        self.rad_pad = int(cfg.rad_pad)

        # CPU generator. random_crop uses torch.randint which needs cpu.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        batch = batch.to(self.device, non_blocking=True)

        # random_crop's generator is on cpu; move back to device after.
        obs_cpu = batch.obs.cpu()
        next_obs_cpu = batch.next_obs.cpu()

        aug_obs = aug_mod.random_crop(
            obs_cpu, pad=self.rad_pad, generator=self._aug_gen
        )
        aug_next_obs = aug_mod.random_crop(
            next_obs_cpu, pad=self.rad_pad, generator=self._aug_gen
        )

        aug_batch = batch._replace(
            obs=aug_obs.to(self.device),
            next_obs=aug_next_obs.to(self.device),
        )
        return super().update(aug_batch)
