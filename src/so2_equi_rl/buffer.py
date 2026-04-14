"""Fixed-capacity uniform replay buffer

Stores transitions in a fixed-capacity uniform buffer, with a fixed-size
batch of transitions sampled uniformly at random.
"""

import collections
from typing import Optional, Tuple

import numpy as np
import torch

Transition = collections.namedtuple(
    "Transition",
    [
        "state",
        "obs",
        "action",
        "reward",
        "next_state",
        "next_obs",
        "done",
        "step_left",
        "expert",
    ],
)


class ReplayBuffer:
    """Uniform replay buffer with fixed-capacity."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
    ) -> None:
        self.capacity = capacity

        # One column per transition field.
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)

        # float32, not bool -- lets Bellman target compute (1 - done)
        # without dtype cast each gradient step.
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._step_lefts = np.zeros((capacity,), dtype=np.int32)
        self._experts = np.zeros((capacity,), dtype=np.bool_)

        # Ring-buffer pointers.
        self._idx = 0  # write index for next transition
        self._size = 0  # number of transitions in buffer

    def push(
        self,
        states: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        step_lefts: Optional[torch.Tensor] = None,
        experts: Optional[torch.Tensor] = None,
    ) -> None:
        batch = states.shape[0]
        if batch > self.capacity:
            raise ValueError(
                "batch size {} exceeds buffer capacity {}".format(batch, self.capacity)
            )

        # Required fields to numpy.
        states_np = states.detach().cpu().numpy().astype(np.float32)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        rewards_np = rewards.detach().cpu().numpy().astype(np.float32)
        next_states_np = next_states.detach().cpu().numpy().astype(np.float32)
        next_obs_np = next_obs.detach().cpu().numpy().astype(np.float32)
        dones_np = dones.detach().cpu().numpy().astype(np.float32)

        # Defaults for the optional fields.
        if step_lefts is None:
            step_lefts_np = np.zeros((batch,), dtype=np.int32)
        else:
            step_lefts_np = step_lefts.detach().cpu().numpy().astype(np.int32)
        if experts is None:
            experts_np = np.zeros((batch,), dtype=np.bool_)
        else:
            experts_np = experts.detach().cpu().numpy().astype(np.bool_)

        # Modulo wraps the indices when the batch crosses the end of the array.
        idxs = (self._idx + np.arange(batch)) % self.capacity

        self._states[idxs] = states_np
        self._obs[idxs] = obs_np
        self._actions[idxs] = actions_np
        self._rewards[idxs] = rewards_np
        self._next_states[idxs] = next_states_np
        self._next_obs[idxs] = next_obs_np
        self._dones[idxs] = dones_np
        self._step_lefts[idxs] = step_lefts_np
        self._experts[idxs] = experts_np

        # Advance write head; size saturates at capacity.
        self._idx = (self._idx + batch) % self.capacity
        self._size = min(self._size + batch, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        if self._size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")

        # Uniform random indices; replacement is acceptable at usual batch sizes.
        idxs = np.random.randint(0, self._size, size=batch_size)

        return Transition(
            state=torch.from_numpy(self._states[idxs]),
            obs=torch.from_numpy(self._obs[idxs]),
            action=torch.from_numpy(self._actions[idxs]),
            reward=torch.from_numpy(self._rewards[idxs]),
            next_state=torch.from_numpy(self._next_states[idxs]),
            next_obs=torch.from_numpy(self._next_obs[idxs]),
            done=torch.from_numpy(self._dones[idxs]),
            step_left=torch.from_numpy(self._step_lefts[idxs]),
            expert=torch.from_numpy(self._experts[idxs]),
        )

    def __len__(self) -> int:
        return self._size
