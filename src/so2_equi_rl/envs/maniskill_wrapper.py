"""ManiSkill 3 wrapper matching EnvWrapper's public API so trainers stay
backend-agnostic. Top-down narrow-FOV camera approximates an orthographic
view so the rendered depth behaves like a heightmap (<1% edge distortion
at the defaults in TrainConfig, small enough for the equivariant encoder
to absorb). Physical 5-D action is [p, dx, dy, dz, dtheta] in BulletArm's
pxyzr layout; _physical_to_ms3 below handles the remap into pd_ee_delta_pose.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from so2_equi_rl.envs import EnvStep

# Supported ManiSkill tasks for the paper-reproduction extension. PickCube
# and PullCube are direct analogues of BulletArm's block_picking and
# block_pulling. StackCube stands in for drawer_opening because
# ManiSkill's OpenCabinetDrawer-v1 is a Fetch-based mobile-manipulation
# task, not a tabletop one.
_SUPPORTED_MS3_TASKS = {"PickCube-v1", "PullCube-v1", "StackCube-v1"}


# ManiSkill's default pd_ee_delta_pose bounds. We rescale the agent's
# physical action into the [-1, 1] cube these bounds describe.
_MS3_POS_BOUND = 0.1  # meters, matches pd_ee_delta_pose default
_MS3_ROT_BOUND = 0.1  # radians, matches pd_ee_delta_pose default


class ManiSkillWrapper:
    """GPU-vectorized ManiSkill 3 wrapper. (B, ...) tensor contract."""

    def __init__(
        self,
        cfg,
        seed: int,
        num_envs: int = 1,
        expert_fn: Optional[Callable[["ManiSkillWrapper"], torch.Tensor]] = None,
    ) -> None:
        # Lazy-imported so environments without ManiSkill still import the package.
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401, registers ManiSkill envs
        from mani_skill.utils import sapien_utils
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        if cfg.env_name not in _SUPPORTED_MS3_TASKS:
            raise NotImplementedError(
                f"env_name={cfg.env_name!r} is not a supported ManiSkill task. "
                f"Available: {sorted(_SUPPORTED_MS3_TASKS)}"
            )

        self.env_name = cfg.env_name
        self.num_envs = num_envs
        self.batch_size = num_envs
        self.action_dim = 5
        self.state_dim = 1
        self.obs_size = cfg.obs_size
        self.depth_max = float(cfg.ms3_depth_max)
        self.camera_height = float(cfg.ms3_camera_height)
        self.seed = seed

        self._device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._expert_fn = (
            expert_fn  # set later via set_expert; allows task-specific wiring
        )

        # Build a top-down camera pose. look_at returns a sapien Pose; we
        # flatten into the [x, y, z, qw, qx, qy, qz] list sensor_configs
        # expects to serialize.
        pose = sapien_utils.look_at(
            eye=[0.0, 0.0, float(cfg.ms3_camera_height)],
            target=[0.0, 0.0, 0.0],
        )
        pose_list = pose.p.flatten().tolist() + pose.q.flatten().tolist()

        # gym.make builds the task, sensor_configs override the default
        # camera named base_camera with our top-down rig.
        base_env = gym.make(
            cfg.env_name,
            num_envs=num_envs,
            obs_mode="rgbd",
            reward_mode=cfg.ms3_reward_mode,
            control_mode=cfg.ms3_control_mode,
            sim_backend=cfg.ms3_sim_backend,
            sensor_configs=dict(
                base_camera=dict(
                    pose=pose_list,
                    width=self.obs_size,
                    height=self.obs_size,
                    fov=float(np.deg2rad(cfg.ms3_camera_fov)),
                    near=0.01,
                    far=float(cfg.ms3_camera_height) + 0.5,
                ),
            ),
        )

        # Gymnasium's default TimeLimit returns truncated=True forever
        # after the first timeout, which would collapse critic targets
        # to reward-only. The ManiSkill wrapper auto-resets instead.
        self._env = ManiSkillVectorEnv(
            base_env,
            num_envs=num_envs,
            auto_reset=True,
            ignore_terminations=False,
        )
        self._last_obs: Optional[Dict[str, Any]] = None

    @property
    def unwrapped(self):
        # Access for scripted experts that need privileged poses (e.g.
        # cube position) which rgbd obs_mode doesn't include in `extra`.
        return self._env.unwrapped

    # ------------------------------------------------------------------
    # Public API, mirrors envs/wrapper.py:EnvWrapper
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _info = self._env.reset(seed=self.seed)
        self._last_obs = obs_dict
        return self._extract(obs_dict)

    def step(self, actions: torch.Tensor) -> EnvStep:
        ms3_action = self._physical_to_ms3(actions.to(self._device).float())
        obs_dict, rewards, terminated, truncated, _info = self._env.step(ms3_action)
        self._last_obs = obs_dict

        state, obs = self._extract(obs_dict)
        done = (terminated | truncated).float().reshape(self.batch_size).cpu()
        reward = rewards.float().reshape(self.batch_size).cpu()
        return EnvStep(state=state, obs=obs, reward=reward, done=done)

    def get_expert_action(self) -> torch.Tensor:
        # Experts are registered per task via set_expert; fail loud if warmup would be a no-op.
        if self._expert_fn is None:
            raise RuntimeError(
                f"No scripted expert registered for {self.env_name!r}. "
                "Call wrapper.set_expert(fn) before training."
            )
        return self._expert_fn(self).to(dtype=torch.float32, device="cpu")

    def set_expert(self, fn: Callable[["ManiSkillWrapper"], torch.Tensor]) -> None:
        # fn(self) returns physical 5-D actions (B, 5) matching SAC's pxyzr layout.
        self._expert_fn = fn

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract(self, obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        # ManiSkill's base_camera depth is int16 millimeters from the
        # lens. The equivariant encoder wants a heightmap above the
        # table to match BulletArm's close-loop obs.
        depth_raw = obs_dict["sensor_data"]["base_camera"]["depth"]
        depth_m = depth_raw.float().permute(0, 3, 1, 2) * 0.001  # mm -> m
        height = self.camera_height - depth_m  # depth-from-lens -> height
        # depth_max caps far-field background noise.
        height = torch.clamp(height, 0.0, self.depth_max)
        obs = height.cpu().reshape(self.batch_size, 1, self.obs_size, self.obs_size)

        # Gripper state. Panda's gripper occupies the last qpos slots;
        # threshold the last finger (≈0.04 fully open, ≈0.0 fully closed).
        qpos = obs_dict["agent"]["qpos"]
        gripper_pos = qpos[:, -1]
        state = (
            (gripper_pos < 0.02).float().cpu().reshape(self.batch_size, self.state_dim)
        )
        return state, obs

    def _physical_to_ms3(self, physical: torch.Tensor) -> torch.Tensor:
        # Input layout (from SACAgent.decode_action): [p, dx, dy, dz, dtheta].
        # Output layout (pd_ee_delta_pose): [dx, dy, dz, drx, dry, drz, gripper].
        # Normalize physical deltas by the controller's configured bounds,
        # clip so the controller doesn't see values outside its cube, and
        # remap gripper conventions (BulletArm 0=open, 1=closed →
        # ManiSkill +1=open, -1=closed).
        p = physical[:, 0:1]
        dxyz = physical[:, 1:4]
        dtheta = physical[:, 4:5]

        dxyz_norm = torch.clamp(dxyz / _MS3_POS_BOUND, -1.0, 1.0)
        drz_norm = torch.clamp(dtheta / _MS3_ROT_BOUND, -1.0, 1.0)
        gripper_ms = 1.0 - 2.0 * p  # 0 -> +1, 1 -> -1

        out = torch.zeros(self.batch_size, 7, device=physical.device)
        out[:, 0:3] = dxyz_norm
        # drx, dry already zero; locked to keep SO(2) symmetry.
        out[:, 5:6] = drz_norm
        out[:, 6:7] = gripper_ms
        return out
