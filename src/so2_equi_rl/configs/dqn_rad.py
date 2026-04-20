"""RAD-DQN config. Only adds the two RAD knobs on top of DQNConfig.

Unlike SAC-RAD, continuous SO(2) is disallowed here: the 3x3 xy grid is
only invariant under C_4 rotations, so continuous theta would force
snapping and lose the lossless-augmentation property. Paper's aug
type is 'dqn_c4'.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.dqn import DQNConfig


@dataclass
class DQNRADConfig(DQNConfig):
    # 'discrete_cN' is the only supported mode. Kept as a field so the
    # flag shows up in train_dqn.py's auto-registered CLI.
    rad_aug_mode: str = "discrete_cN"

    # Must be a factor of 4 so k * (2 pi / N) is always a multiple of pi/2.
    # N=4 matches the paper's dqn_c4 baseline.
    rad_group_order: int = 4

    def __post_init__(self) -> None:
        if self.rad_aug_mode != "discrete_cN":
            raise ValueError(
                f"DQN-RAD supports only rad_aug_mode='discrete_cN', got {self.rad_aug_mode!r}"
            )
        if self.rad_group_order not in (2, 4):
            raise ValueError(
                f"rad_group_order must be 2 or 4 (to preserve the 3x3 xy grid), "
                f"got {self.rad_group_order}"
            )
