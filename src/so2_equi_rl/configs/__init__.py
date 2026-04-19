"""Training configs. TrainConfig holds the trainer-agnostic knobs;
SACConfig adds the SAC update rule's hyperparameters.
"""

from so2_equi_rl.configs.base import TrainConfig
from so2_equi_rl.configs.sac import SACConfig
from so2_equi_rl.configs.sac_drq import SACDrQConfig
from so2_equi_rl.configs.sac_rad import SACRADConfig

__all__ = ["TrainConfig", "SACConfig", "SACDrQConfig", "SACRADConfig"]
