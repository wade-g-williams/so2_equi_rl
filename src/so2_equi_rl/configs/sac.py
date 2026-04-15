"""SAC-specific config. Inherits shared training knobs from TrainConfig
and adds the SAC update rule's hyperparameters.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from so2_equi_rl.configs.base import TrainConfig


@dataclass
class SACConfig(TrainConfig):
    # Network capacity + equivariance group. Forwarded to encoder_cls(**)
    # inside SACAgent.__init__. For CNN variants, group_order is ignored
    # by CNNEncoder but kept in the kwargs bundle for signature uniformity.
    obs_channels: int = 2
    action_dim: int = 5
    n_hidden: int = 128
    group_order: int = 8

    # Action decoder bounds. dpos is the per-axis position delta magnitude
    # in meters, drot is the rotation delta magnitude in radians, p_range
    # is the closed gripper interval the tanh output maps into.
    dpos: float = 0.05
    drot: float = math.pi / 8
    p_range: Tuple[float, float] = (0.0, 1.0)

    # SAC hyperparameters.
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    init_alpha: float = 0.1

    # When None, target_entropy defaults to -action_dim inside SACAgent.
    target_entropy: Optional[float] = None

    # Global-norm clip applied per optimizer. None = disabled (matches paper repo).
    grad_clip_norm: Optional[float] = None

    # UTD ratio: gradient updates per env.step batch. 1 = standard SAC.
    n_updates_per_step: int = 1
