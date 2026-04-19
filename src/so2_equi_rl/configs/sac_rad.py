"""RAD-SAC config. Adds the two RAD aug knobs on top of SACConfig.

Paper's sac_rad.py doesn't hardcode aug_type, so rad_aug_mode is a free
field. Default 'continuous' matches the paper's so2 RAD baseline.
"""

from dataclasses import dataclass
from typing import Optional

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACRADConfig(SACConfig):
    # 'continuous' matches the paper's so2 RAD baseline. 'discrete_cN'
    # runs RAD over the C_N lattice instead.
    rad_aug_mode: str = "continuous"

    # None resolves to cfg.group_order in __post_init__. Unused in continuous mode.
    rad_group_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.rad_group_order is None:
            self.rad_group_order = self.group_order
