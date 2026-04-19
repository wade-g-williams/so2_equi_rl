"""DQNTrainer. Scripted-expert warmup snapped to the discrete grid,
epsilon-greedy exploration with linear decay over cfg.explore_steps.
"""

import torch

from so2_equi_rl.agents.base import ActionPair
from so2_equi_rl.agents.dqn import DQNAgent
from so2_equi_rl.configs.dqn import DQNConfig
from so2_equi_rl.trainers.base import BaseTrainer


class DQNTrainer(BaseTrainer):
    """DQN rollout with grid-snapped expert warmup and eps-greedy exploration."""

    cfg: DQNConfig
    agent: DQNAgent

    def _warmup_action(self, state: torch.Tensor, obs: torch.Tensor) -> ActionPair:
        # Scripted planner outputs continuous physical actions. encode_action
        # snaps to the grid cell, decode_action re-emits the realized physical
        # command so env.step, reward, and the buffer all agree on the action.
        physical_raw = self.train_env.get_expert_action()
        indices = self.agent.encode_action(physical_raw)
        physical_on_grid = self.agent.decode_action(indices)
        return ActionPair(
            unscaled=indices.float(),
            physical=physical_on_grid,
        )

    def _explore(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        global_step: int,
    ) -> ActionPair:
        # Linear epsilon decay. Floors at final_eps after explore_steps.
        cfg = self.cfg
        frac = min(global_step / max(cfg.explore_steps, 1), 1.0)
        eps = cfg.init_eps + frac * (cfg.final_eps - cfg.init_eps)
        eps = max(eps, cfg.final_eps)
        return self.agent.select_action(state, obs, eps=eps, deterministic=False)
