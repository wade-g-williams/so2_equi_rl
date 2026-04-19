"""RAD-SAC config. Adds the two RAD aug knobs on top of SACConfig and
leaves every other SAC hyperparameter untouched so the baseline and the
augmented variant share one set of defaults.

Paper's sac_rad.py does not hardcode aug_type, so rad_aug_mode is a free
config field. Default 'continuous' matches the so2 baseline the paper
highlights for RAD.
"""

from dataclasses import dataclass
from typing import Optional

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACRADConfig(SACConfig):
    # Default to continuous SO(2) to match the paper's 'so2' RAD baseline.
    # Set to 'discrete_cN' to run RAD over the C_N lattice instead.
    rad_aug_mode: str = "continuous"

    # None resolves to cfg.group_order in __post_init__ so one knob drives
    # both the encoder's equivariance group and the RAD aug lattice when the
    # user picks discrete_cN. Unused in continuous mode.
    rad_group_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.rad_group_order is None:
            self.rad_group_order = self.group_order
