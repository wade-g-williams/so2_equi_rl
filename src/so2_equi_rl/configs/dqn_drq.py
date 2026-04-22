"""DrQ-DQN config. Paper sec E + Kostrikov et al. 2020: DrQ uses random
pixel shift of +/-4 pixels, K=M=2 averages of target and loss.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.dqn import DQNConfig


@dataclass
class DQNDrQConfig(DQNConfig):
    # Paper sec E: K = M = 2.
    drq_k: int = 2
    drq_m: int = 2

    # Paper sec E: +/-4 pixel shift on 128x128 heightmaps.
    drq_pad: int = 4
