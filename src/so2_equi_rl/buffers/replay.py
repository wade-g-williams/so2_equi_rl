"""Fixed-capacity replay buffer over preallocated CPU tensors. Once full,
new transitions overwrite the oldest. obs and next_obs are quantized to
uint8 on push and dequantized on sample (~4x memory savings, 13 GB -> 3.25
GB at default capacity). sample() returns float32 so agent code doesn't care.

SO(2) augmentation: with so2_aug_k>0, every pushed transition is followed
by k rotated copies (random theta per copy). Paper Fig 7 uses k=4. Action
must be 5-dim [p, dx, dy, dz, dr]; rotation applies to obs, next_obs, and
(dx, dy) only. Paper: Wang et al. 2022, Sec 6.2 + utils/torch_utils.py.
"""

from typing import NamedTuple, Tuple

import torch

from so2_equi_rl.buffers.so2_aug import augment_transition_so2, sample_thetas


class Transition(NamedTuple):
    state: torch.Tensor
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor

    def to(self, device, non_blocking: bool = False) -> "Transition":
        # Move every field in one pass so device placement stays out of the SAC update hot path.
        return Transition(
            state=self.state.to(device, non_blocking=non_blocking),
            obs=self.obs.to(device, non_blocking=non_blocking),
            action=self.action.to(device, non_blocking=non_blocking),
            reward=self.reward.to(device, non_blocking=non_blocking),
            next_state=self.next_state.to(device, non_blocking=non_blocking),
            next_obs=self.next_obs.to(device, non_blocking=non_blocking),
            done=self.done.to(device, non_blocking=non_blocking),
        )


# Catches float drift from tanh squashing or encode_action rounding.
_ACTION_BOUND_TOL = 1e-5


def _as_cpu_f32(t: torch.Tensor) -> torch.Tensor:
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
        obs_clip: float = 0.32,
        obs_scale: float = 0.4,
        enforce_unscaled_action_range: bool = True,
        so2_aug_k: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if so2_aug_k < 0:
            raise ValueError(f"so2_aug_k must be >= 0, got {so2_aug_k}")
        # so2_aug assumes the 5-dim SAC action layout [p, dx, dy, dz, dr].
        # DQN stores integer grid indices (action_dim=4) and never needs aug
        # per the paper (Fig 6 uses no aug buffer).
        if so2_aug_k > 0 and action_dim != 5:
            raise ValueError(
                f"so2_aug_k>0 requires action_dim=5 (SAC layout), got {action_dim}"
            )
        self.capacity = capacity
        self._obs_clip = float(obs_clip)
        self._obs_scale = float(obs_scale)
        self._so2_aug_k = int(so2_aug_k)
        # SAC stores tanh-squashed actions in [-1, 1]; DQN stores integer
        # grid indices cast to float32. Flip off for the latter.
        self._enforce_unscaled_action_range = bool(enforce_unscaled_action_range)

        # CPU float32 everywhere except obs/next_obs (quantized uint8 in [0, 255],
        # dequantized only at sample()). Fancy-index sampling copies, so returned
        # batches don't alias the storage.
        self._state = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self._action = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self._reward = torch.zeros((capacity,), dtype=torch.float32)
        self._next_state = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self._done = torch.zeros((capacity,), dtype=torch.float32)

        # Torch Generator keeps sampling reproducible without pulling in numpy.
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

        self._idx = 0
        self._size = 0

        self._state_dim = state_dim
        self._obs_shape = tuple(obs_shape)
        self._action_dim = action_dim

    def _quantize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # Clip to the heightmap ceiling (BulletArm workspace bound) and rescale so obs_scale -> 255.
        scaled = obs.clamp(min=0.0, max=self._obs_clip) * (255.0 / self._obs_scale)
        return scaled.round().clamp(0, 255).to(torch.uint8)

    def _dequantize_obs(self, obs_u8: torch.Tensor) -> torch.Tensor:
        # Max quantization error is obs_scale/255.
        return obs_u8.to(torch.float32) * (self._obs_scale / 255.0)

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

        # Shape check every field. (B, 1) reward/done would broadcast into
        # self._reward[idxs] silently and corrupt the buffer.
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
        action_c = _as_cpu_f32(action)
        reward_c = _as_cpu_f32(reward)
        next_state_c = _as_cpu_f32(next_state)
        done_c = _as_cpu_f32(done)

        # Invariant: stored actions are unscaled in [-1, 1]. Catches
        # physical-unit actions leaking in from the planner without going
        # through agent.encode_action first. DQN turns this off so it can
        # store integer grid indices as float32.
        if self._enforce_unscaled_action_range:
            max_abs = action_c.abs().max().item()
            if max_abs > 1.0 + _ACTION_BOUND_TOL:
                raise ValueError(
                    "ReplayBuffer stores unscaled actions in [-1, 1]; got "
                    f"max|a|={max_abs:.4f}. Did you forget to encode_action() before push?"
                )

        # Keep float obs/next_obs in scope for aug before we quantize.
        obs_f32 = _as_cpu_f32(obs)
        next_obs_f32 = _as_cpu_f32(next_obs)
        obs_c = self._quantize_obs(obs_f32)
        next_obs_c = self._quantize_obs(next_obs_f32)

        # Write the original B transitions first.
        self._write_chunk(
            state_c, obs_c, action_c, reward_c, next_state_c, next_obs_c, done_c
        )

        # Paper's SO(2) buffer aug: k random rotations per transition,
        # pushed alongside the original. Done on the float obs so bilinear
        # interp doesn't interact with the uint8 quantization noise floor.
        # Scalars (state, reward, next_state, done) are rotation-invariant.
        for _ in range(self._so2_aug_k):
            thetas = sample_thetas(batch, self._gen)
            s_aug, o_aug, a_aug, r_aug, ns_aug, no_aug, d_aug = augment_transition_so2(
                state_c,
                obs_f32,
                action_c,
                reward_c,
                next_state_c,
                next_obs_f32,
                done_c,
                thetas,
            )
            self._write_chunk(
                s_aug,
                self._quantize_obs(o_aug),
                a_aug,
                r_aug,
                ns_aug,
                self._quantize_obs(no_aug),
                d_aug,
            )

    def _write_chunk(
        self,
        state_c: torch.Tensor,
        obs_c: torch.Tensor,
        action_c: torch.Tensor,
        reward_c: torch.Tensor,
        next_state_c: torch.Tensor,
        next_obs_c: torch.Tensor,
        done_c: torch.Tensor,
    ) -> None:
        # Wrap-around write a batch of already-cast tensors. Factored out
        # because the aug path writes k+1 chunks per push().
        batch = state_c.shape[0]
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

        # Uniform with replacement. During warmup batch_size > _size will repeat entries.
        idxs = torch.randint(0, self._size, (batch_size,), generator=self._gen)

        return Transition(
            state=self._state[idxs],
            obs=self._dequantize_obs(self._obs[idxs]),
            action=self._action[idxs],
            reward=self._reward[idxs],
            next_state=self._next_state[idxs],
            next_obs=self._dequantize_obs(self._next_obs[idxs]),
            done=self._done[idxs],
        )

    def __len__(self) -> int:
        return self._size

    def state_dict(self) -> dict:
        # Tensors cloned so a later push() can't mutate the saved copy
        # before torch.save hits disk. Schema includes quantization params
        # so a resume catches a mismatched clip/scale (silent corruption).
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
            "gen_state": self._gen.get_state(),
            "schema": {
                "capacity": self.capacity,
                "state_dim": self._state_dim,
                "obs_shape": self._obs_shape,
                "action_dim": self._action_dim,
                "obs_clip": self._obs_clip,
                "obs_scale": self._obs_scale,
            },
        }

    def load_state_dict(self, d: dict) -> None:
        # Hard schema check: resuming into a differently-shaped buffer is almost always a CLI mistake.
        schema = d["schema"]
        expected = {
            "capacity": self.capacity,
            "state_dim": self._state_dim,
            "obs_shape": self._obs_shape,
            "action_dim": self._action_dim,
            "obs_clip": self._obs_clip,
            "obs_scale": self._obs_scale,
        }
        if schema != expected:
            raise ValueError(
                f"ReplayBuffer schema mismatch on load: expected {expected}, got {schema}"
            )

        # copy_ into preallocated storage to keep the replay tensors in place.
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
