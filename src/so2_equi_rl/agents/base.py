"""Abstract Agent base class. Trainer calls this contract, subclasses implement it."""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple

import torch
from torch import Tensor


class ActionPair(NamedTuple):
    """unscaled goes into the buffer, physical goes into env.step."""

    unscaled: Tensor
    physical: Tensor


class Agent(ABC):
    """Minimal contract the trainer calls."""

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
        """Full trainable state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Inverse of state_dict."""

    @staticmethod
    def _unalias_buffers(module: torch.nn.Module) -> None:
        # e2cnn R2Conv basis-expansion buffers share storage across blocks, so
        # load_state_dict's copy_ refuses to write. Swap each non-contiguous
        # buffer for a contiguous copy.
        for submod in module.modules():
            for name in list(submod._buffers):
                buf = submod._buffers[name]
                if buf is not None and not buf.is_contiguous():
                    submod._buffers[name] = buf.contiguous()
