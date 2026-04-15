"""Wrapper around the helping_hands_rl_envs PyBullet task library.

Hides the library's two runners (in-process and worker-subprocess)
behind one batched interface with fixed tensor shapes. Restricted to
close-loop tasks, so obs is always a top-down heightmap.

No action decoding lives here: the agent hands in a 5-D tensor and the
wrapper passes it straight through. All [-1, 1] -> physical scaling
belongs in the agent.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

# The helping_hands_rl_envs __file__ patch lives in so2_equi_rl.envs.__init__
# so it fires on any entry into the envs package, not just this module.
from helping_hands_rl_envs import env_factory


class EnvStep(NamedTuple):
    """Return shape for EnvWrapper.step. Positional order matches the
    4-tuple unpacking trainers expect: state, obs, reward, done.
    """

    state: torch.Tensor
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


# The paper's workspace
_DEFAULT_WORKSPACE = np.asarray(
    [
        [0.25, 0.65],  # x: table's long axis
        [-0.20, 0.20],  # y: across the table (symmetric around the robot base)
        [0.00, 1.00],  # z: vertical, positive = up
    ],
    dtype=np.float32,
)


# Step sizes for the library's scripted expert (5 cm, 22.5 deg).
# The RL agent has its own dpos/drot in agents/sac.py.
_EXPERT_DPOS = 0.05
_EXPERT_DROT = float(np.pi / 8)

# Locked env_config knobs.
_ACTION_SEQUENCE = "pxyzr"  # 5-D continuous action: (p, dx, dy, dz, dtheta)
_FAST_MODE = True  # skip intermediate motion waypoints
_RANDOM_ORIENTATION = False  # objects spawn axis-aligned
_ROBOT = "kuka"  # Kuka LBR iiwa arm
_PHYSICS_MODE = "fast"  # lower-fidelity physics, much faster


# Close-loop tasks the wrapper supports. Locking to this set keeps the
# observation a plain top-down heightmap (no in-hand camera).
_CLOSE_LOOP_ENVS = {
    "close_loop_block_picking",
    "close_loop_block_reaching",
    "close_loop_block_stacking",
    "close_loop_block_pulling",
    "close_loop_house_building_1",
    "close_loop_block_picking_corner",
    "close_loop_drawer_opening",
    "close_loop_household_picking",
}


class EnvWrapper:
    """Batched wrapper. Returns (B, ...) tensors; B=1 in-process, B=N workers."""

    def __init__(
        self,
        env_name: str,
        num_processes: int = 0,
        seed: int = 0,
        obs_size: int = 128,
        max_steps: int = 50,
        num_objects: int = 1,
        render: bool = False,
        workspace: Optional[np.ndarray] = None,
        planner_config: Optional[dict] = None,
    ) -> None:
        if env_name not in _CLOSE_LOOP_ENVS:
            raise ValueError(
                f"env_name={env_name!r} is not a supported close-loop task. "
                f"Valid options: {sorted(_CLOSE_LOOP_ENVS)}"
            )

        if workspace is None:
            # Copy so callers can't mutate the module-level default.
            workspace = _DEFAULT_WORKSPACE.copy()

        # Settings the task library's createEnvs() expects.
        env_config = {
            "workspace": workspace,
            "max_steps": max_steps,  # episode length cap
            "obs_size": obs_size,  # heightmap side length (H == W)
            "action_sequence": _ACTION_SEQUENCE,
            "num_objects": num_objects,  # most tasks use 1; stacking uses 2+
            "render": render,  # True pops a GUI window; False for training
            "fast_mode": _FAST_MODE,
            "random_orientation": _RANDOM_ORIENTATION,
            "robot": _ROBOT,
            "physics_mode": _PHYSICS_MODE,
            "seed": seed,
        }

        # Settings for the library's scripted expert (used to generate demos).
        if planner_config is None:
            planner_config = {
                "random_orientation": False,
                "dpos": _EXPERT_DPOS,
                "drot": _EXPERT_DROT,
            }

        # createEnvs gives us a SingleRunner when num_processes=0, or a
        # MultiRunner that spawns num_processes worker subprocesses otherwise.
        self._runner = env_factory.createEnvs(
            num_processes, "pybullet", env_name, env_config, planner_config
        )

        self.env_name = env_name
        self.num_processes = num_processes
        self.action_dim = len(_ACTION_SEQUENCE)  # one char per dim: "pxyzr" -> 5
        self.obs_size = obs_size
        self.state_dim = 1  # scalar gripper open/close
        self.batch_size = num_processes if num_processes > 0 else 1

        self._is_single = num_processes == 0

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # The library also returns an in-hand camera obs, which close-loop
        # tasks don't use. Throw it away.
        states, _in_hand, obs = self._runner.reset()
        return self._to_batched_obs(states, obs)

    def step(self, actions: torch.Tensor) -> EnvStep:
        if not torch.is_tensor(actions):
            raise TypeError(
                f"actions must be a torch tensor, got {type(actions).__name__}"
            )
        expected_shape = (self.batch_size, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"expected action shape {expected_shape}, got {tuple(actions.shape)}"
            )

        actions_np = actions.detach().cpu().numpy()
        if actions_np.dtype != np.float32:
            actions_np = actions_np.astype(np.float32)

        # SingleRunner wants a flat (5,) action; MultiRunner wants (N, 5).
        if self._is_single:
            actions_np = actions_np[0]

        (states, _in_hand, obs), rewards, dones = self._runner.step(actions_np)

        states_t, obs_t = self._to_batched_obs(states, obs)

        # SingleRunner returns Python scalars for reward/done; MultiRunner
        # returns (N,) arrays. Force everything to (B,) arrays.
        if self._is_single:
            rewards_np = np.asarray([rewards], dtype=np.float32)
            dones_np = np.asarray([float(dones)], dtype=np.float32)
        else:
            rewards_np = np.asarray(rewards, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.float32)

        return EnvStep(
            state=states_t,
            obs=obs_t,
            reward=torch.from_numpy(rewards_np),
            done=torch.from_numpy(dones_np),
        )

    def get_expert_action(self) -> torch.Tensor:
        # Asks the library's scripted planner what it would do next.
        # Used to fill the replay buffer with demos before SAC training.
        actions = self._runner.getNextAction()
        actions = np.asarray(actions, dtype=np.float32)
        if self._is_single:
            actions = actions[None]
        return torch.from_numpy(actions)

    def close(self) -> None:
        self._runner.close()

    def _to_batched_obs(
        self, states: np.ndarray, obs: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Force both runners' output shapes into the same (B, ...) layout.
        # Single gives (1,)/(1,H,W); Multi gives (N,1)/(N,1,H,W).
        states = np.asarray(states, dtype=np.float32).reshape(
            self.batch_size, self.state_dim
        )
        obs = np.asarray(obs, dtype=np.float32).reshape(
            self.batch_size, 1, self.obs_size, self.obs_size
        )
        return torch.from_numpy(states), torch.from_numpy(obs)
