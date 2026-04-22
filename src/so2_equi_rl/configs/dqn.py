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
    # Paper text sec 6.1 claims A_xy step = 0.05, but every paper repo DQN
    # command in README uses `--dpos=0.02` (binds dx=dy=dz=dpos at
    # create_agent.py:35). Fig 6 curves were produced with dpos=0.02;
    # text appears to have a typo, no code path can produce A_xy != A_z.
    # dpos=0.05 is too coarse for Object Picking (5cm step over a 5cm
    # cube overshoots; 2cm step gives the finer control paper actually uses).
    dpos: float = 0.02
    dz: float = 0.02
    drot: float = math.pi / 16
    p_range: Tuple[float, float] = (0.0, 1.0)

    # DQN hyperparameters per paper Table 8 / Appendix F.
    gamma: float = 0.95
    tau: float = 0.01  # Polyak soft target update
    lr: float = 1e-4

    # None = disabled (matches paper repo).
    grad_clip_norm: Optional[float] = None

    # Epsilon-greedy schedule. Paper repo runs DQN with `--explore=0`
    # (parameters.py:35), which makes LinearSchedule return final_eps=0.0
    # from step 0: no explicit eps-greedy noise. Early exploration comes
    # from the untrained Q-net's near-random argmax plus the 100-episode
    # expert warmup. An earlier init_eps=1.0 decayed 1.0 -> 0.0 uniform
    # random, which diluted the expert-demo signal for the first ~10k updates.
    init_eps: float = 0.0
    final_eps: float = 0.0
    explore_steps: int = 1  # irrelevant when init_eps == final_eps

    # UTD ratio. 1 = standard DQN.
    n_updates_per_step: int = 1
