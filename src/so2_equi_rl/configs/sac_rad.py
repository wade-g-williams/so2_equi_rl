"""RAD-SAC config. Paper §E: "RAD Crop baselines use random crop for data
augmentation. The random crop crops a 142x142 state image to the size of
128x128." Our heightmaps are already rendered at 128x128, so we pad to
142 (pad=7) then random-crop back to 128, giving the same effective op.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACRADConfig(SACConfig):
    # Paper pads by 7 pixels (128 -> 142 -> crop 128) = ±7 pixel shift.
    # Matches Laskin et al. 2020a and Wang et al. 2022 §E.
    rad_pad: int = 7
