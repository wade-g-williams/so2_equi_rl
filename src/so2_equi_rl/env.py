"""Adapter between helping_hands_rl_envs and the rest of the codebase.

Wraps the library's Runner (SingleRunner for num_processes=0, MultiRunner
for num_processes>0) behind a uniform interface.
"""

from typing import Optional, Tuple

import numpy as np
import torch

# helping_hands_rl_envs is the PyBullet-based manipulation-task library
# pinned as a git submodule at ./helping_hands_rl_envs/.
from helping_hands_rl_envs import env_factory

# The paper's training workspace: a 0.40m cube centered at (x=0.45, y=0),
# with z ranging from the table surface to 1 m above. Shape (3, 2),
# each row = (low, high) bounds in meters. Matches the config the paper
# repo builds from its --workspace_size=0.4 CLI flag.
_DEFAULT_WORKSPACE = np.asarray(
    [
        [0.25, 0.65],  # x: table's long axis
        [-0.20, 0.20],  # y: across the table (symmetric around the robot base)
        [0.00, 1.00],  # z: vertical, positive = up
    ],
    dtype=np.float32,
)


# The close-loop task family
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
    """Wrapper around a helping_hands_rl_envs Runner."""

    def __init__(
        self,
        env_name: str,
        num_processes: int = 0,
        seed: int = 0,
        obs_size: int = 128,
        max_steps: int = 50,
        action_sequence: str = "pxyzr",
        num_objects: int = 1,
        render: bool = False,
        workspace: Optional[np.ndarray] = None,
        planner_config: Optional[dict] = None,
    ) -> None:
        if env_name not in _CLOSE_LOOP_ENVS:
            raise ValueError(
                "env_name={!r} is not a supported close-loop task. "
                "Valid options: {}".format(env_name, sorted(_CLOSE_LOOP_ENVS))
            )

        if workspace is None:
            workspace = _DEFAULT_WORKSPACE

        # env_config is the dict the library's createEnvs() uses. (same as paper)
        env_config = {
            "workspace": workspace,
            "max_steps": max_steps,  # episode length cap
            "obs_size": obs_size,  # heightmap side length (H == W)
            "action_sequence": action_sequence,  # "pxyzr" -> 5-D continuous action
            "num_objects": num_objects,  # most tasks use 1; stacking uses 2+
            "render": render,  # True pops a GUI; False for training
            "fast_mode": True,  # skip intermediate waypoints
            "random_orientation": False,  # objects spawn axis-aligned
            "robot": "kuka",  # Kuka LBR iiwa
            "physics_mode": "fast",  # lower-fidelity physics, much faster
            "seed": seed,
        }

        # planner_config parameterizes the library's scripted expert demonstrator.
        # dpos/drot are the physical step sizes the planner uses when deciding its own
        # action magnitudes (5 cm per position step, 22.5 degrees per rotation step).
        if planner_config is None:
            planner_config = {
                "random_orientation": False,
                "dpos": 0.05,
                "drot": float(np.pi / 8),
            }

        # createEnvs returns a SingleRunner (num_processes=0) or a
        # MultiRunner (num_processes>0) that spawns that many worker subprocesses.
        self._runner = env_factory.createEnvs(
            num_processes, "pybullet", env_name, env_config, planner_config
        )

        # Public metadata. Downstream modules (networks, buffer, train
        # loop) read these to size tensors without having to step the
        # env first to learn its shapes.
        self.env_name = env_name
        self.num_processes = num_processes
        self.action_dim = len(action_sequence)  # "pxyzr" -> 5
        self.obs_size = obs_size
        self.state_dim = 1  # scalar gripper open/close
        self.batch_size = num_processes if num_processes > 0 else 1

        self._is_single = num_processes == 0

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Discard hand_obs -- unused for close-loop tasks.
        states, _in_hand, obs = self._runner.reset()
        return self._to_batched_obs(states, obs)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(actions):
            raise TypeError(
                "actions must be a torch tensor, got {}".format(type(actions).__name__)
            )
        expected_shape = (self.batch_size, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                "expected action shape {}, got {}".format(
                    expected_shape, tuple(actions.shape)
                )
            )

        actions_np = actions.detach().cpu().numpy().astype(np.float32)

        # SingleRunner.step wants (5,); MultiRunner wants (N, 5).
        if self._is_single:
            actions_np = actions_np[0]

        # Both runners can return a 4-tuple (obs, rewards, dones, metadata)
        # if the underlying env emits metadata. Close-loop tasks in fast_mode
        # don't, but assert to fail fast with a clear message if that changes.
        result = self._runner.step(actions_np)
        assert len(result) == 3, (
            "unexpected step() return shape (len={}); "
            "metadata path not supported".format(len(result))
        )
        (states, _in_hand, obs), rewards, dones = result

        states_t, obs_t = self._to_batched_obs(states, obs)

        # Single gives Python scalars, multi gives (N,) arrays.
        if self._is_single:
            rewards_np = np.asarray([rewards], dtype=np.float32)
            dones_np = np.asarray([float(dones)], dtype=np.float32)
        else:
            rewards_np = np.asarray(rewards, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.float32)

        return (
            states_t,
            obs_t,
            torch.from_numpy(rewards_np),
            torch.from_numpy(dones_np),
        )

    def get_expert_action(self) -> torch.Tensor:
        # Library's scripted planner -- used to bootstrap the replay
        # buffer with demo episodes before SAC training starts.
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
        # Reshape to contract shape directly -- robust to both runners'
        # inner shapes (Single: (1,)/(1,H,W); Multi: (N,1)/(N,1,H,W)).
        states = np.asarray(states, dtype=np.float32).reshape(
            self.batch_size, self.state_dim
        )
        obs = np.asarray(obs, dtype=np.float32).reshape(
            self.batch_size, 1, self.obs_size, self.obs_size
        )
        return torch.from_numpy(states), torch.from_numpy(obs)
