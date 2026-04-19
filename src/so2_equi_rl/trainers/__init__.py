"""Training loops. Shared skeleton in base.py, SAC and DQN specializations next to it."""

from so2_equi_rl.trainers.base import BaseTrainer
from so2_equi_rl.trainers.dqn import DQNTrainer
from so2_equi_rl.trainers.sac import SACTrainer

__all__ = ["BaseTrainer", "SACTrainer", "DQNTrainer"]
