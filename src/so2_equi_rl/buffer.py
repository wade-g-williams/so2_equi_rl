"""Standard replay buffer.

Backed by preallocated numpy arrays instead of a Python deque or torch
tensors. Preallocation keeps memory flat over long training runs, and
numpy indexing makes random sampling O(batch_size) with no graph-tracking
overhead.

- Fixed-size ring buffer. Once it's full, each new transition
  overwrites the oldest one.
- Stores (state, obs, action, reward, next_state, next_obs, done)
- Samples a uniform random batch when the agent asks for one.
"""

import collections
from typing import Tuple
import numpy as np
import torch

Transition = collections.namedtuple(
    "Transition",
    ["state", "obs", "action", "reward", "next_state", "next_obs", "done"],
)


# Convert a torch tensor to a float32 numpy array
def _as_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


class ReplayBuffer:
    """Fixed-capacity ring buffer over (state, obs, action, reward, next_state, next_obs, done)."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        seed: int = 0,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        self._idx = 0  # where the next transition gets written
        self._size = 0  # how many transitions are currently stored

    def push(
        self,
        states: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch = states.shape[0]
        if batch > self.capacity:
            raise ValueError(
                "batch size {} exceeds buffer capacity {}".format(batch, self.capacity)
            )
        # rewards/dones must be (B,), not (B,1) — the latter would broadcast
        # into self._rewards[idxs] silently and corrupt the buffer.
        assert rewards.shape == (batch,), "rewards must be shape ({},), got {}".format(
            batch, tuple(rewards.shape)
        )
        assert dones.shape == (batch,), "dones must be shape ({},), got {}".format(
            batch, tuple(dones.shape)
        )

        states_np = _as_np(states)
        obs_np = _as_np(obs)
        actions_np = _as_np(actions)
        rewards_np = _as_np(rewards)
        next_states_np = _as_np(next_states)
        next_obs_np = _as_np(next_obs)
        dones_np = _as_np(dones)

        # Where each new transition lands
        idxs = (self._idx + np.arange(batch)) % self.capacity

        self._states[idxs] = states_np
        self._obs[idxs] = obs_np
        self._actions[idxs] = actions_np
        self._rewards[idxs] = rewards_np
        self._next_states[idxs] = next_states_np
        self._next_obs[idxs] = next_obs_np
        self._dones[idxs] = dones_np

        self._idx = (self._idx + batch) % self.capacity
        self._size = min(self._size + batch, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        if self._size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")

        # Uniform random sample with replacement. If batch_size > _size
        # during warmup, entries repeat.
        idxs = self._rng.integers(0, self._size, size=batch_size)

        # self._states[idxs] copies the data, so the
        # returned tensors don't share memory with the buffer.
        return Transition(
            state=torch.from_numpy(self._states[idxs]),
            obs=torch.from_numpy(self._obs[idxs]),
            action=torch.from_numpy(self._actions[idxs]),
            reward=torch.from_numpy(self._rewards[idxs]),
            next_state=torch.from_numpy(self._next_states[idxs]),
            next_obs=torch.from_numpy(self._next_obs[idxs]),
            done=torch.from_numpy(self._dones[idxs]),
        )

    def __len__(self) -> int:
        return self._size
