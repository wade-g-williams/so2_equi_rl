"""RL agent package. Agent is the abstract contract; concrete agents
(SAC for now, DQN later) inherit from it.
"""

from so2_equi_rl.agents.base import Agent
from so2_equi_rl.agents.sac import SACAgent

__all__ = ["Agent", "SACAgent"]
