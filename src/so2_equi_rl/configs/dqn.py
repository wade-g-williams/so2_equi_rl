"""DQN config. Inherits trainer knobs from TrainConfig and adds the
discrete pxyzr action grid plus the DQN update rule's hyperparameters.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from so2_equi_rl.configs.base import TrainConfig


@dataclass
class DQNConfig(TrainConfig):
    # Trainer-loop overrides matching paper Table 8 / Appendix F. The paper
    # repo's parameters.py argparse defaults differ (bs=64) but training runs
    # reported in the paper used the Table 8 values.
    total_steps: int = 20_000
    warmup_steps: int = 5_000  # ~100 expert episodes at max_steps=50
    batch_size: int = 32
    buffer_capacity: int = 100_000

    # Network bundle. Forwarded to net_cls(**) inside DQNAgent. group_order
    # is ignored by CNNDQNNet but kept in the kwargs bundle for parity.
    obs_channels: int = 2
    group_order: int = 4
    n_hidden: int = 64

    # Action-grid sizes. action_dim=4 columns (p_id, xy_id, z_id, theta_id)
    # stored as float32 in the shared buffer, cast to long inside the agent.
    action_dim: int = 4
    n_p: int = 2
    n_xy: int = 9
    n_z: int = 3
    n_theta: int = 3

    # Action-grid step sizes per paper text (Wang et al. 2022 §6.1):
    #   A_xy = {(x,y) | x,y ∈ {-0.05m, 0m, 0.05m}}
    #   A_z  = {-0.02m, 0m, 0.02m}
    #   A_θ  = {-π/16, 0, π/16}
    # Paper runs dx=dy=dpos and a separate, smaller dz. The paper repo's
    # argparse defaults match here (dpos=0.05); earlier internal runs used
    # dpos=0.02 which diverged from paper text.
    # p_range is the closed gripper interval the discrete p indices map onto.
    dpos: float = 0.05
    dz: float = 0.02
    drot: float = math.pi / 16
    p_range: Tuple[float, float] = (0.0, 1.0)

    # DQN hyperparameters per paper Table 8 / Appendix F.
    gamma: float = 0.95
    tau: float = 0.01  # Polyak soft target update
    lr: float = 1e-4

    # None = disabled (matches paper repo).
    grad_clip_norm: Optional[float] = None

    # Linear epsilon decay over explore_steps env steps.
    init_eps: float = 1.0
    final_eps: float = 0.0
    explore_steps: int = 10_000

    # UTD ratio. 1 = standard DQN.
    n_updates_per_step: int = 1
