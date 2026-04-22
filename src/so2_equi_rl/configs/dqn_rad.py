"""RAD-DQN config. Paper §E: RAD uses random crop (142x142 -> 128x128).
Our heightmaps are rendered at 128x128, so we pad to 142 (pad=7) then
random-crop back to 128.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.dqn import DQNConfig


@dataclass
class DQNRADConfig(DQNConfig):
    # Paper §E: pad by 7 on each side (128 + 14 = 142, crop back to 128).
    rad_pad: int = 7
