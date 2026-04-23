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
    warmup_steps: int = 5_000  # kept for backward compat, superseded below
    # Paper sec 6.1: DQN warmup is 100 episodes.
    warmup_episodes: int = 100
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

    # Action-grid step sizes per paper text (Wang et al. 2022 sec 6.1):
    #   A_xy = {(x,y) | x,y in {-0.05m, 0m, 0.05m}}
    #   A_z  = {-0.02m, 0m, 0.02m}
    #   A_theta  = {-pi/16, 0, pi/16}
    # Paper text says A_xy = 0.05 but the released code binds dx = dy = dz
    # = 0.02 and Fig 6 was produced at that setting. 0.05 also overshoots
    # the 5 cm cube in Object Picking, so 0.02 is the value that works.
    dpos: float = 0.02
    dz: float = 0.02
    drot: float = math.pi / 16
    p_range: Tuple[float, float] = (0.0, 1.0)

    # DQN hyperparameters per paper Table 8 / Appendix F.
    gamma: float = 0.95
    tau: float = 0.01  # Polyak soft target update
    lr: float = 1e-4

    # None = paper default.
    grad_clip_norm: Optional[float] = None

    # Epsilon-greedy schedule. Paper sets explore=0, which gives final_eps=0
    # from step 0 and no explicit eps-greedy noise. Early exploration comes
    # from the untrained Q-net's near-random argmax plus the 100-episode
    # expert warmup. Decaying from init_eps=1.0 instead dilutes the expert
    # signal for the first ~10k updates.
    init_eps: float = 0.0
    final_eps: float = 0.0
    explore_steps: int = 1  # irrelevant when init_eps == final_eps

    # UTD ratio. 1 = standard DQN.
    n_updates_per_step: int = 1
