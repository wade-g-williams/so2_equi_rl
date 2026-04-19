"""SACTrainer. Scripted-expert warmup, always-stochastic exploration. All
four SAC variants (base, DrQ, RAD, FERM) instantiate this unchanged since
their differences live in agent.update.
"""

import torch

from so2_equi_rl.agents.base import ActionPair
from so2_equi_rl.trainers.base import BaseTrainer


class SACTrainer(BaseTrainer):
    """SAC rollout with scripted-expert warmup and stochastic exploration."""

    def _warmup_action(self, state: torch.Tensor, obs: torch.Tensor) -> ActionPair:
        # Wang et al.'s close-loop tasks ship scripted planners; seed the
        # buffer with those instead of random noise. encode_action clamps
        # so the unscaled-in-[-1, 1] invariant holds even at the decoder edges.
        physical = self.train_env.get_expert_action()
        unscaled = self.agent.encode_action(physical)
        return ActionPair(unscaled=unscaled, physical=physical)

    def _explore(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        global_step: int,
    ) -> ActionPair:
        # SAC's entropy term handles exploration on its own. global_step
        # is in the signature for DQN's eventual epsilon decay.
        del global_step
        return self.agent.select_action(state, obs, deterministic=False)
