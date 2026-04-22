"""RAD-SAC. Subclass of SACAgent that applies a random crop to (obs,
next_obs) and runs a vanilla SAC update on the augmented transition.

Paper §E: RAD uses random crop (142x142 -> 128x128). Action is NOT
augmented, since pixel-space translation doesn't change the world-frame
delta action. Reward, state, next_state, done are invariant.
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
    """Twin-Q SAC with paper-faithful random-crop augmentation."""

    def __init__(
        self,
        cfg: SACRADConfig,
        encoder_cls: Type[nn.Module],
        actor_cls: Type[nn.Module],
        critic_cls: Type[nn.Module],
    ) -> None:
        super().__init__(cfg, encoder_cls, actor_cls, critic_cls)

        self.rad_pad = int(cfg.rad_pad)

        # CPU generator. random_crop uses torch.randint which needs cpu.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        # Random crop obs and next_obs per row, hand off to base SAC.
        # Action stays untouched, pixel shift is SO(2)-agnostic.
        batch = batch.to(self.device, non_blocking=True)

        # random_crop expects CPU tensors for the generator step; move obs
        # back to the device after crop.
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
