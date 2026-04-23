"""SACTrainer. Scripted-expert warmup, stochastic exploration.

All four SAC variants (base, DrQ, RAD, FERM) share this trainer; variant
differences live in agent.update.
"""

import torch

from so2_equi_rl.agents.base import ActionPair
from so2_equi_rl.trainers.base import BaseTrainer


class SACTrainer(BaseTrainer):
    """SAC rollout with scripted-expert warmup and stochastic exploration."""

    def _warmup_action(self, state: torch.Tensor, obs: torch.Tensor) -> ActionPair:
        # Seed buffer with scripted planner actions; encode_action clamps to [-1, 1].
        physical = self.train_env.get_expert_action()
        unscaled = self.agent.encode_action(physical)
        return ActionPair(unscaled=unscaled, physical=physical)

    def _explore(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        global_step: int,
    ) -> ActionPair:
        # SAC's entropy term handles exploration; global_step is here for the DQN signature.
        del global_step
        return self.agent.select_action(state, obs, deterministic=False)
