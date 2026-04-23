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

    # SAC hyperparameters, per paper Appendix F:
    #   lr = 1e-3 (all three: actor, critic, alpha)
    #   gamma = 0.99
    #   tau = 1e-2 (soft target update)
    #   alpha initialized at 1e-2
    #   target_entropy = -5 (resolves from None to -action_dim inside agent)
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    init_alpha: float = 0.01

    # None resolves to -action_dim inside SACAgent.
    target_entropy: Optional[float] = None

    # Per-optimizer global-norm clip. None = paper default.
    grad_clip_norm: Optional[float] = None

    # UTD ratio. 1 = standard SAC.
    n_updates_per_step: int = 1

    # SO(2) replay-buffer augmentation per Wang et al. Fig 7. Every pushed
    # transition is followed by k random-rotation copies (obs + dxdy only).
    # Paper default = 4. Set to 0 to disable (useful for ablation or when
    # comparing against old unaugmented curves).
    so2_aug_k: int = 4
