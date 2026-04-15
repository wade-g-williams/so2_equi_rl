"""Training configs. TrainConfig holds the trainer-agnostic knobs;
SACConfig adds the SAC update rule's hyperparameters.
"""

from so2_equi_rl.configs.base import TrainConfig
from so2_equi_rl.configs.sac import SACConfig

__all__ = ["TrainConfig", "SACConfig"]
