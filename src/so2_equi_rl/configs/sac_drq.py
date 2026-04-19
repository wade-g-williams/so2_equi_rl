"""DrQ-SAC config. Adds the four DrQ knobs on top of SACConfig and leaves
every other hyperparameter untouched so baseline and DrQ share defaults.
"""

from dataclasses import dataclass
from typing import Optional

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACDrQConfig(SACConfig):
    # DrQ multipliers. Paper defaults (Wang et al.): K = M = 2.
    drq_k: int = 2
    drq_m: int = 2

    # Paper's sac_drq.py hardcodes 'cn' regardless of CLI aug_type, so DrQ
    # is discrete C_N here too. Continuous is still available as an override.
    drq_aug_mode: str = "discrete_cN"

    # None resolves to cfg.group_order in __post_init__ so one knob drives
    # both the encoder's equivariance group and the DrQ aug lattice.
    drq_group_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.drq_group_order is None:
            self.drq_group_order = self.group_order
