"""DQNTrainer stub. Not wired up yet. Pins the epsilon-greedy contract so
DQNAgent has a seam to land in. Every hook raises so accidental
instantiation fails loudly.
"""

import torch

from so2_equi_rl.agents.base import ActionPair
from so2_equi_rl.trainers.base import BaseTrainer


class DQNTrainer(BaseTrainer):
    """DQN stub, variant not implemented yet."""

    def _warmup_action(self, state: torch.Tensor, obs: torch.Tensor) -> ActionPair:
        raise NotImplementedError("DQNTrainer is a stub; DQN variant not built yet")

    def _explore(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        global_step: int,
    ) -> ActionPair:
        raise NotImplementedError("DQNTrainer is a stub; DQN variant not built yet")
