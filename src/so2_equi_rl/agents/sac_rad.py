"""RAD-SAC. Subclass of SACAgent that rotates (obs, next_obs, action) by
one shared per-row theta and then runs a vanilla SAC update on the
augmented transition.

Unlike DrQ, RAD does not average the target over K copies or the loss
over M copies. The shared theta across obs, next_obs, and the (dx, dy)
columns of the action is what keeps the transition on the SO(2)-
equivariant manifold.

Aug RNG lives on a dedicated CPU Generator seeded from cfg.seed + a
fixed offset, so it's decoupled from the global torch RNG (network init,
env, buffer sampling) but still reproducible from one int.
"""

from typing import Dict, Type

import torch
import torch.nn as nn

from so2_equi_rl.agents.sac import SACAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.sac_rad import SACRADConfig
from so2_equi_rl.utils import augmentation as aug_mod

# Fixed offset on top of cfg.seed so the RAD aug RNG is decoupled from
# both the global torch RNG and DrQ's aug RNG (offset 1337). So two
# variants sharing cfg.seed don't replay the same theta sequence.
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
        # rad_group_order resolves to cfg.group_order in SACRADConfig.__post_init__,
        # so None should not leak this far. Cast defensively.
        self.rad_group_order = int(
            cfg.rad_group_order if cfg.rad_group_order is not None else cfg.group_order
        )

        # Dedicated CPU Generator. Same rationale as sac_drq.py: theta
        # is built via torch.randint/torch.rand with no device forwarding,
        # so its output is CPU; rotate_obs moves theta to obs.device for
        # us. A cuda generator would crash on torch.randint.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        # Rotate (obs, next_obs, action) by one shared theta per row, then
        # hand off to the base SAC update. State, reward, next_state, and
        # done are SO(2)-invariant so they pass through untouched.
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

        # _replace builds a new Transition with the three rotated fields
        # swapped in; state, reward, next_state, done are the original refs.
        aug_batch = batch._replace(
            obs=aug_obs,
            next_obs=aug_next_obs,
            action=aug_action,
        )
        return super().update(aug_batch)
