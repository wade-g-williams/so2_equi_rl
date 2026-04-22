"""Wrapper around the helping_hands_rl_envs PyBullet task library.

Hides the in-process and worker-subprocess runners behind one batched
interface with fixed tensor shapes. Restricted to close-loop tasks so
obs is always a top-down heightmap.

No action decoding here. The agent hands in a 5-D tensor and the wrapper
passes it straight through; [-1, 1] to physical-unit scaling lives in the agent.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch

# helping_hands_rl_envs ships an empty __init__, so __file__ is None and
# random_object.py crashes on os.path.dirname(hhe.__file__). Patch it here
# so ms3-only workflows that skip wrapper.py don't need hhe installed.
import helping_hands_rl_envs as _hhe

if _hhe.__file__ is None:
    # __path__ points at the namespace-package root; the real package with
    # simulators/urdf assets lives one level deeper.
    _hhe.__file__ = os.path.join(
        list(_hhe.__path__)[0], "helping_hands_rl_envs", "__init__.py"
    )

from helping_hands_rl_envs import env_factory  # noqa: E402, must follow the patch

# Re-export so existing `from so2_equi_rl.envs.wrapper import EnvStep`
# callers keep working.
from so2_equi_rl.envs import EnvStep  # noqa: E402, F401

# Paper's workspace.
_DEFAULT_WORKSPACE = np.asarray(
    [
        [0.25, 0.65],  # x: table's long axis
        [-0.20, 0.20],  # y: across the table (symmetric around the robot base)
        [0.00, 1.00],  # z: vertical, positive = up
    ],
    dtype=np.float32,
)


# Fallback step sizes for the library's scripted expert (5 cm, 22.5 deg).
# EnvWrapper takes dpos/drot ctor args so the planner's step size stays
# aligned with the agent's action grid. A mismatch silently corrupts warmup
# data (env executes a 5 cm expert move, buffer stores the snapped grid
# index, Q-learning trains on transitions where action and next_obs disagree).
_EXPERT_DPOS = 0.05
_EXPERT_DROT = float(np.pi / 8)

# Locked env_config knobs.
_ACTION_SEQUENCE = "pxyzr"  # 5-D continuous action: (p, dx, dy, dz, dtheta)
_FAST_MODE = True  # skip intermediate motion waypoints
_RANDOM_ORIENTATION = False  # objects spawn axis-aligned
_ROBOT = "kuka"  # Kuka LBR iiwa arm
_PHYSICS_MODE = "fast"  # lower-fidelity physics, much faster


# Locking to close-loop tasks keeps the obs a plain top-down heightmap
# (no in-hand camera).
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

        # Scripted expert config (used to generate demos). The planner's
        # dpos/drot must match the agent's action grid / scaling so the
        # transitions stored in the buffer use the same action units the
        # env actually executed.
        if planner_config is None:
            planner_config = {
                "random_orientation": False,
                "dpos": float(dpos) if dpos is not None else _EXPERT_DPOS,
                "drot": float(drot) if drot is not None else _EXPERT_DROT,
            }

        # createEnvs returns a SingleRunner when num_processes=0, else a
        # MultiRunner spawning num_processes workers.
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
        # Library also returns an in-hand camera obs; close-loop tasks don't use it.
        states, _in_hand, obs = self._runner.reset()
        return self._to_batched_obs(states, obs)

    def step(self, actions: torch.Tensor) -> EnvStep:
        actions_np = actions.detach().cpu().numpy()
        if actions_np.dtype != np.float32:
            actions_np = actions_np.astype(np.float32)

        # SingleRunner wants flat (5,), MultiRunner wants (N, 5).
        if self._is_single:
            actions_np = actions_np[0]

        # auto_reset=True so when an episode ends, the runner immediately
        # resets the env and returns the post-reset obs. Without this the
        # env sits in a terminal state and every subsequent step returns
        # done=True after one tick. See close_loop_env.py step(). The
        # symptom is eval/length_mean == 10.8 (one 50-step episode + four
        # single-tick ghost episodes averaged to 10.8). BulletArm's
        # MultiRunner supports this via the 'step_auto_reset' IPC cmd;
        # SingleRunner was patched to match.
        (states, _in_hand, obs), rewards, dones = self._runner.step(
            actions_np, auto_reset=True
        )

        states_t, obs_t = self._to_batched_obs(states, obs)

        # Force everything to (B,). Single returns scalars, Multi returns (N,) arrays.
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
        # Asks the scripted planner what it would do next. Used to fill the buffer with demos.
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
        # Force both runners' shapes into a common (B, ...) layout.
        # Single gives (1,) and (1, H, W); Multi gives (N, 1) and (N, 1, H, W).
        states = np.asarray(states, dtype=np.float32).reshape(
            self.batch_size, self.state_dim
        )
        obs = np.asarray(obs, dtype=np.float32).reshape(
            self.batch_size, 1, self.obs_size, self.obs_size
        )
        return torch.from_numpy(states), torch.from_numpy(obs)
