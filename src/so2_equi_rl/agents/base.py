"""Abstract base for RL agents. Concrete agents (SAC, DQN, ...) implement
this surface so the trainer stays agnostic: select_action for rollouts,
update for learning, state_dict pair for checkpointing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple

from torch import Tensor


class ActionPair(NamedTuple):
    """Return shape for Agent.select_action.

    unscaled: what the replay buffer stores. SAC: [-1, 1] continuous;
    DQN: int64 grid index.
    physical: what env.step consumes (pxyzr in physical units).
    """

    unscaled: Tensor
    physical: Tensor


class Agent(ABC):
    """Minimal contract the trainer calls against. Networks, optimizers,
    action decoders, and temperature tuning all live inside subclasses.
    """

    @abstractmethod
    def select_action(
        self,
        state: Tensor,
        obs: Tensor,
        deterministic: bool = False,
    ) -> ActionPair:
        """Returns ActionPair(unscaled, physical)."""

    @abstractmethod
    def update(self, batch: Any) -> Dict[str, float]:
        """One gradient step on a minibatch. Returns scalar losses/metrics
        the trainer logs verbatim.
        """

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Full trainable state for checkpointing: nets, optimizers, and
        any tuned temperatures. Lives on the agent so optimizer state
        travels with the weights.
        """

    @abstractmethod
    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Inverse of state_dict."""
