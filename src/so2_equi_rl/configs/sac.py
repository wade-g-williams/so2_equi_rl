"""SAC config. Inherits trainer knobs from TrainConfig and adds the SAC
update rule's hyperparameters.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from so2_equi_rl.configs.base import TrainConfig


@dataclass
class SACConfig(TrainConfig):
    # Forwarded to encoder_cls(**) inside SACAgent. group_order is ignored
    # by CNNEncoder but kept in the kwargs bundle for parity with EquiEncoder.
    obs_channels: int = 2
    action_dim: int = 5
    n_hidden: int = 128
    group_order: int = 8

    # Action decoder bounds. dpos in meters, drot in radians, p_range is
    # the closed gripper interval the tanh output maps into.
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

    # None resolves to -action_dim inside SACAgent.
    target_entropy: Optional[float] = None

    # Per-optimizer global-norm clip. None = disabled (matches paper repo).
    grad_clip_norm: Optional[float] = None

    # UTD ratio. 1 = standard SAC.
    n_updates_per_step: int = 1
