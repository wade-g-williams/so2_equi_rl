"""DrQ-SAC config. Paper sec E + Kostrikov et al. 2020: DrQ uses random
pixel SHIFT of +/-4 pixels with K=M=2 augmentations for target and loss.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACDrQConfig(SACConfig):
    # DrQ multipliers. Paper sec E: K = M = 2.
    drq_k: int = 2
    drq_m: int = 2

    # Paper sec E: +/-4 pixel shift on 128x128 heightmaps.
    drq_pad: int = 4
