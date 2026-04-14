"""Uniform replay buffer for off-policy pixel-based RL.

Public API:
    ReplayBuffer(capacity, state_dim, obs_shape, action_dim, seed=None)
    buf.push(states, obs, actions, rewards, next_states, next_obs, dones)
    buf.sample(batch_size) -> Transition
    len(buf)

Transition field shapes (per-sample):
    state       (state_dim,)    float32   scalar gripper open/close
    obs         (1, H, W)       float32   top-down heightmap
    action      (action_dim,)   float32   (p, x, y, z, r)
    reward      ()              float32
    next_state  (state_dim,)    float32
    next_obs    (1, H, W)       float32
    done        ()              float32   float so Bellman (1-d) is cast-free
"""

import collections
from typing import Optional, Tuple

import numpy as np
import torch

Transition = collections.namedtuple(
    "Transition",
    ["state", "obs", "action", "reward", "next_state", "next_obs", "done"],
)


class ReplayBuffer:
    """Uniform replay buffer with fixed capacity."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity

        # Instance-local RNG so seeding the buffer doesn't touch the
        # global numpy RNG.
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        # Ring-buffer pointers.
        self._idx = 0  # write index for next transition
        self._size = 0  # number of transitions currently stored

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

        states_np = states.detach().cpu().numpy().astype(np.float32)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        rewards_np = rewards.detach().cpu().numpy().astype(np.float32)
        next_states_np = next_states.detach().cpu().numpy().astype(np.float32)
        next_obs_np = next_obs.detach().cpu().numpy().astype(np.float32)
        dones_np = dones.detach().cpu().numpy().astype(np.float32)

        # Modulo wraps indices when the batch crosses the end of the array.
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

        # Uniform with replacement. If batch_size > _size during warmup,
        # entries repeat, which is standard for off-policy replay.
        idxs = self._rng.integers(0, self._size, size=batch_size)

        # self._states[idxs] is fancy indexing, which copies, so the
        # returned tensors are decoupled from buffer storage.
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
