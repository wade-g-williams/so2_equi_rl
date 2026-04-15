"""Fixed-capacity replay buffer on preallocated CPU tensors. Once full,
each new transition overwrites the oldest. Stores (state, obs, action,
reward, next_state, next_obs, done) and uniform-samples batches.
"""

from typing import NamedTuple, Tuple

import torch


class Transition(NamedTuple):
    state: torch.Tensor
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor

    def to(self, device, non_blocking: bool = False) -> "Transition":
        # Move every field to `device` in one pass. Used by the SAC update
        # loop so device placement stays out of the hot path.
        return Transition(
            state=self.state.to(device, non_blocking=non_blocking),
            obs=self.obs.to(device, non_blocking=non_blocking),
            action=self.action.to(device, non_blocking=non_blocking),
            reward=self.reward.to(device, non_blocking=non_blocking),
            next_state=self.next_state.to(device, non_blocking=non_blocking),
            next_obs=self.next_obs.to(device, non_blocking=non_blocking),
            done=self.done.to(device, non_blocking=non_blocking),
        )


# Tolerance on the [-1, 1] action-bound check in push(). A tight bound
# catches float drift from tanh squashing or encode_action rounding.
_ACTION_BOUND_TOL = 1e-5


def _as_cpu_f32(t: torch.Tensor) -> torch.Tensor:
    # Detach from graph, then coerce to CPU float32. .to() is a no-op when
    # device and dtype already match; .detach() always returns a new view.
    return t.detach().to(dtype=torch.float32, device="cpu")


class ReplayBuffer:
    """Fixed-capacity replay buffer over (state, obs, action, reward, next_state, next_obs, done)."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity

        # CPU float32 storage. Fancy-index sampling copies into a new
        # tensor so returned batches never alias the replay storage.
        self._state = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self._action = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self._reward = torch.zeros((capacity,), dtype=torch.float32)
        self._next_state = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self._done = torch.zeros((capacity,), dtype=torch.float32)

        # Torch Generator keeps sampling reproducible without importing numpy.
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

        self._idx = 0  # where the next transition gets written
        self._size = 0  # how many transitions are currently stored

        self._state_dim = state_dim
        self._obs_shape = tuple(obs_shape)
        self._action_dim = action_dim

    def push(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        batch = state.shape[0]
        if batch > self.capacity:
            raise ValueError(
                f"batch size {batch} exceeds buffer capacity {self.capacity}"
            )

        # Check every field's shape. reward/done as (B, 1) would broadcast
        # into self._reward[idxs] silently and corrupt the buffer; the
        # other fields would fail later with a less obvious error.
        schema = (
            ("state", state, (batch, self._state_dim)),
            ("obs", obs, (batch, *self._obs_shape)),
            ("action", action, (batch, self._action_dim)),
            ("reward", reward, (batch,)),
            ("next_state", next_state, (batch, self._state_dim)),
            ("next_obs", next_obs, (batch, *self._obs_shape)),
            ("done", done, (batch,)),
        )
        for name, tensor, expected in schema:
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must be shape {expected}, got {tuple(tensor.shape)}"
                )

        state_c = _as_cpu_f32(state)
        obs_c = _as_cpu_f32(obs)
        action_c = _as_cpu_f32(action)
        reward_c = _as_cpu_f32(reward)
        next_state_c = _as_cpu_f32(next_state)
        next_obs_c = _as_cpu_f32(next_obs)
        done_c = _as_cpu_f32(done)

        # Invariant: stored actions are unscaled in [-1, 1]. Catches
        # physical-unit actions leaking in from the scripted planner
        # without going through agent.encode_action first.
        max_abs = action_c.abs().max().item()
        if max_abs > 1.0 + _ACTION_BOUND_TOL:
            raise ValueError(
                "ReplayBuffer stores unscaled actions in [-1, 1]; got "
                f"max|a|={max_abs:.4f}. Did you forget to encode_action() before push?"
            )

        # Wrap-around write indices into the replay storage.
        idxs = (torch.arange(batch) + self._idx) % self.capacity

        self._state[idxs] = state_c
        self._obs[idxs] = obs_c
        self._action[idxs] = action_c
        self._reward[idxs] = reward_c
        self._next_state[idxs] = next_state_c
        self._next_obs[idxs] = next_obs_c
        self._done[idxs] = done_c

        self._idx = (self._idx + batch) % self.capacity
        self._size = min(self._size + batch, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        if self._size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")

        # Uniform with replacement. If batch_size > _size during warmup,
        # entries repeat. Fancy-indexing copies, so returned tensors don't
        # share storage with the buffer.
        idxs = torch.randint(0, self._size, (batch_size,), generator=self._gen)

        return Transition(
            state=self._state[idxs],
            obs=self._obs[idxs],
            action=self._action[idxs],
            reward=self._reward[idxs],
            next_state=self._next_state[idxs],
            next_obs=self._next_obs[idxs],
            done=self._done[idxs],
        )

    def __len__(self) -> int:
        return self._size

    def state_dict(self) -> dict:
        # Snapshot for checkpointing. Tensors are cloned so later push() calls
        # can't mutate the saved copy before torch.save hits disk.
        return {
            "state": self._state.clone(),
            "obs": self._obs.clone(),
            "action": self._action.clone(),
            "reward": self._reward.clone(),
            "next_state": self._next_state.clone(),
            "next_obs": self._next_obs.clone(),
            "done": self._done.clone(),
            "idx": self._idx,
            "size": self._size,
            "gen_state": self._gen.get_state(),  # torch.ByteTensor; restore via set_state
            "schema": {
                "capacity": self.capacity,
                "state_dim": self._state_dim,
                "obs_shape": self._obs_shape,
                "action_dim": self._action_dim,
            },
        }

    def load_state_dict(self, d: dict) -> None:
        # Hard schema check: resuming into a differently-shaped buffer is
        # almost always a CLI mistake, not an intentional resize.
        schema = d["schema"]
        expected = {
            "capacity": self.capacity,
            "state_dim": self._state_dim,
            "obs_shape": self._obs_shape,
            "action_dim": self._action_dim,
        }
        if schema != expected:
            raise ValueError(
                f"ReplayBuffer schema mismatch on load: expected {expected}, got {schema}"
            )

        # copy_ into preallocated storage so the replay tensors stay in place.
        self._state.copy_(d["state"])
        self._obs.copy_(d["obs"])
        self._action.copy_(d["action"])
        self._reward.copy_(d["reward"])
        self._next_state.copy_(d["next_state"])
        self._next_obs.copy_(d["next_obs"])
        self._done.copy_(d["done"])

        self._idx = int(d["idx"])
        self._size = int(d["size"])
        self._gen.set_state(d["gen_state"])
