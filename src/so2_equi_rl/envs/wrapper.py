"""Wrapper around helping_hands_rl_envs. Batched (B, ...) tensor interface.

Restricted to close-loop tasks so obs is a top-down heightmap. Agent owns
action decoding; [-1, 1] to physical scaling lives in the agent, not here.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch

# helping_hands_rl_envs ships an empty __init__ (random_object.py reads
# hhe.__file__). Patch it here so the import succeeds.
import helping_hands_rl_envs as _hhe

if _hhe.__file__ is None:
    # __path__ is the namespace root; real package is one deeper.
    _hhe.__file__ = os.path.join(
        list(_hhe.__path__)[0], "helping_hands_rl_envs", "__init__.py"
    )

from helping_hands_rl_envs import env_factory  # noqa: E402, must follow the patch

# Re-export EnvStep so existing import paths keep working.
from so2_equi_rl.envs import EnvStep  # noqa: E402, F401

# Paper sec C workspace, 0.4m x 0.4m x 0.24m.
_DEFAULT_WORKSPACE = np.asarray(
    [
        [0.25, 0.65],  # x: table's long axis, 0.4m
        [-0.20, 0.20],  # y: across the table, 0.4m (symmetric around robot base)
        [0.01, 0.25],  # z: vertical, 0.24m (paper spec)
    ],
    dtype=np.float32,
)


# Scripted-expert fallback step sizes (5 cm, 22.5 deg). Must match agent's action grid.
_EXPERT_DPOS = 0.05
_EXPERT_DROT = float(np.pi / 8)

# Locked env_config knobs.
_ACTION_SEQUENCE = "pxyzr"  # 5-D continuous action: (p, dx, dy, dz, dtheta)
_FAST_MODE = True  # skip intermediate motion waypoints
_RANDOM_ORIENTATION = False  # objects spawn axis-aligned
_ROBOT = "kuka"  # Kuka LBR iiwa arm
_PHYSICS_MODE = "fast"  # lower-fidelity physics, much faster


# Close-loop only so obs stays a plain top-down heightmap.
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
        dpos: Optional[float] = None,
        drot: Optional[float] = None,
    ) -> None:
        if env_name not in _CLOSE_LOOP_ENVS:
            raise ValueError(
                f"env_name={env_name!r} is not a supported close-loop task. "
                f"Valid options: {sorted(_CLOSE_LOOP_ENVS)}"
            )

        if workspace is None:
            # Copy so callers can't mutate the module default.
            workspace = _DEFAULT_WORKSPACE.copy()

        env_config = {
            "workspace": workspace,
            "max_steps": max_steps,
            "obs_size": obs_size,  # heightmap side length (H == W)
            "action_sequence": _ACTION_SEQUENCE,
            "num_objects": num_objects,  # most tasks use 1; stacking uses 2+
            "render": render,  # True pops a GUI window
            "fast_mode": _FAST_MODE,
            "random_orientation": _RANDOM_ORIENTATION,
            "robot": _ROBOT,
            "physics_mode": _PHYSICS_MODE,
            "seed": seed,
        }

        # Scripted expert config; dpos/drot must match the agent's action grid.
        if planner_config is None:
            planner_config = {
                "random_orientation": False,
                "dpos": float(dpos) if dpos is not None else _EXPERT_DPOS,
                "drot": float(drot) if drot is not None else _EXPERT_DROT,
            }

        # createEnvs returns SingleRunner for num_processes=0, else MultiRunner.
        self._runner = env_factory.createEnvs(
            num_processes, "pybullet", env_name, env_config, planner_config
        )

        self.env_name = env_name
        self.num_processes = num_processes
        self.action_dim = len(_ACTION_SEQUENCE)  # one char per dim, "pxyzr" is 5
        self.obs_size = obs_size
        self.state_dim = 1  # scalar gripper open/close
        self.batch_size = num_processes if num_processes > 0 else 1

        self._is_single = num_processes == 0

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # _in_hand is an in-hand camera obs; close-loop tasks don't use it.
        states, _in_hand, obs = self._runner.reset()
        return self._to_batched_obs(states, obs)

    def step(self, actions: torch.Tensor) -> EnvStep:
        actions_np = actions.detach().cpu().numpy()
        if actions_np.dtype != np.float32:
            actions_np = actions_np.astype(np.float32)

        # SingleRunner wants flat (5,), MultiRunner wants (N, 5).
        if self._is_single:
            actions_np = actions_np[0]

        # SingleRunner.step() drops auto_reset (runner.py:419), MultiRunner honors it.
        # Emulate MultiRunner's semantics here: reset on done when single.
        (states, _in_hand, obs), rewards, dones = self._runner.step(
            actions_np, auto_reset=True
        )
        if self._is_single and bool(dones):
            states, _in_hand, obs = self._runner.reset()

        states_t, obs_t = self._to_batched_obs(states, obs)

        # Force (B,). Single returns scalars, Multi returns (N,) arrays.
        if self._is_single:
            rewards_np = np.asarray([rewards], dtype=np.float32)
            dones_np = np.asarray([float(dones)], dtype=np.float32)
        else:
            rewards_np = np.asarray(rewards, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.float32)

        # BulletArm rewards are sparse {0, 1}; success == reward 1.
        success_np = (rewards_np > 0.5).astype(np.float32)
        return EnvStep(
            state=states_t,
            obs=obs_t,
            reward=torch.from_numpy(rewards_np),
            done=torch.from_numpy(dones_np),
            success=torch.from_numpy(success_np),
        )

    def get_expert_action(self) -> torch.Tensor:
        # Scripted planner's next action for warmup demos.
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
        # Force both runners into a common (B, ...) layout.
        states = np.asarray(states, dtype=np.float32).reshape(
            self.batch_size, self.state_dim
        )
        obs = np.asarray(obs, dtype=np.float32).reshape(
            self.batch_size, 1, self.obs_size, self.obs_size
        )
        return torch.from_numpy(states), torch.from_numpy(obs)
