"""Abstract Agent base class. The trainer only calls the methods on
this class, concrete agents (SAC, DQN, ...) implement them.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple

from torch import Tensor


class ActionPair(NamedTuple):
    """Return shape for Agent.select_action.

    unscaled goes into the buffer (SAC: [-1, 1] continuous; DQN: int64
    grid index). physical goes into env.step (pxyzr in physical units).
    """

    unscaled: Tensor
    physical: Tensor


class Agent(ABC):
    """Minimal contract the trainer calls. Networks, optimizers, action
    decoder, and entropy tuning all live in subclasses.
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
        """One gradient step. Returns scalar losses and metrics for logging."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Full trainable state for checkpointing. Optimizer state lives
        with the weights.
        """

    @abstractmethod
    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Inverse of state_dict."""
