"""ManiSkill 3 wrapper. Mirrors EnvWrapper's public API for backend-agnostic trainers.

Top-down narrow-FOV camera approximates an orthographic view (<1% edge distortion).
Physical action is pxyzr; _physical_to_ms3 remaps it into pd_ee_delta_pose.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import sapien
from so2_equi_rl.envs import EnvStep

# Supported MS3 tasks. PickCube/PullCube mirror BulletArm's block_picking/block_pulling.
# StackCube stands in for drawer_opening (MS3's drawer task is mobile-manipulation).
_SUPPORTED_MS3_TASKS = {"PickCube-v1", "PullCube-v1", "StackCube-v1", "EquiPickCube-v1"}

_equi_tasks_registered = False


# pd_ee_delta_pose default bounds. Physical actions rescale into [-1, 1].
_MS3_POS_BOUND = 0.1  # meters
_MS3_ROT_BOUND = 0.1  # radians

# Panda arm + hand links to hide from the sensor camera; fingers stay visible.
# An unhidden arm breaks the SO(2) symmetry the equi encoder assumes.
_PANDA_HIDDEN_LINKS = (
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
)


def _register_equi_tasks() -> None:
    # Register EquiPickCube-v1, EquiPullCube-v1, EquiStackCube-v1. Each subclass
    # drops visibility on _PANDA_HIDDEN_LINKS in _load_scene. Physics components
    # ignore the attribute so collisions and grasping still work.
    global _equi_tasks_registered
    if _equi_tasks_registered:
        return

    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.envs.tasks.tabletop.pull_cube import PullCubeEnv
    from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
    from mani_skill.utils.registration import register_env

    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    class _EquiTaskMixin:
        # Overhead camera at 1 m, 60 deg FOV, 128x128. Replaces every default
        # sensor camera so obs["sensor_data"] only has "overhead_camera".
        @property
        def _default_sensor_configs(self):

            pose = sapien_utils.look_at(eye=[0.0, 0.0, 1.0], target=[0.0, 0.0, 0.0])
            return [
                CameraConfig(
                    uid="overhead_camera",
                    pose=pose,
                    width=128,
                    height=128,
                    fov=float(np.pi / 3),
                    near=0.01,
                    far=2.0,
                )
            ]

        def _load_scene(self, options):
            super()._load_scene(options)
            robot = self.agent.robot
            links_map = getattr(robot, "links_map", None)
            if links_map is None:
                return
            for link_name in _PANDA_HIDDEN_LINKS:
                link = links_map.get(link_name)
                if link is None:
                    continue
                for obj in link._objs:
                    for comp in obj.entity.components:
                        if hasattr(comp, "visibility"):
                            comp.visibility = 0.0

    @register_env("EquiPickCube-v1", max_episode_steps=50)
    class EquiPickCubeEnv(_EquiTaskMixin, PickCubeEnv):
        pass

    @register_env("EquiPullCube-v1", max_episode_steps=50)
    class EquiPullCubeEnv(_EquiTaskMixin, PullCubeEnv):
        pass

    @register_env("EquiStackCube-v1", max_episode_steps=50)
    class EquiStackCubeEnv(_EquiTaskMixin, StackCubeEnv):
        pass

    _equi_tasks_registered = True


class ManiSkillWrapper:
    """GPU-vectorized ManiSkill 3 wrapper. (B, ...) tensor contract."""

    def __init__(
        self,
        cfg,
        seed: int,
        num_envs: int = 1,
        expert_fn: Optional[Callable[["ManiSkillWrapper"], torch.Tensor]] = None,
    ) -> None:
        # Lazy import so non-ManiSkill envs can still import the package.
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401, registers ManiSkill envs
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        if cfg.env_name not in _SUPPORTED_MS3_TASKS:
            raise NotImplementedError(
                f"env_name={cfg.env_name!r} is not a supported ManiSkill task. "
                f"Available: {sorted(_SUPPORTED_MS3_TASKS)}"
            )

        # Register arm-hidden subclasses so visibility drops in _load_scene
        # before the batched render system locks.
        _register_equi_tasks()

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
        # Set later via set_expert for task-specific wiring.
        self._expert_fn = expert_fn

        if self.env_name == "EquiPickCube-v1":
            # Camera + arm hiding handled by the EquiPickCubeEnv subclass in
            # _register_equi_tasks. Camera is pinned at z=1 m so override cfg
            # here to keep _extract's height = camera_height - depth_m aligned.
            self._camera_key = "overhead_camera"
            self.camera_height = 1.0
            base_env = gym.make(
                cfg.env_name,
                num_envs=num_envs,
                obs_mode="rgbd",  # rgbd returns int16 mm matching _extract
                reward_mode=cfg.ms3_reward_mode,
                control_mode=cfg.ms3_control_mode,
                sim_backend=cfg.ms3_sim_backend,
            )
        else:
            # Override default base_camera with our top-down rig.
            camera_height = self.camera_height  # match _extract's placement
            workspace_x = 0.0
            workspace_y = 0.0

            R = np.array(
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0],
                ]
            )
            look_down_quat = Rotation.from_matrix(R).as_quat()
            self._camera_key = "base_camera"
            base_env = gym.make(
                cfg.env_name,
                num_envs=num_envs,
                obs_mode="rgbd",
                reward_mode=cfg.ms3_reward_mode,
                control_mode=cfg.ms3_control_mode,
                sim_backend=cfg.ms3_sim_backend,
                sensor_configs=dict(
                    base_camera=dict(
                        uid="overhead_camera",
                        pose=sapien.Pose(
                            p=[workspace_x, workspace_y, camera_height],
                            q=look_down_quat,
                        ),
                        width=128,
                        height=128,
                        fov=np.pi / 3,  # 60 deg FOV, covers ~1.15m at 1m height
                        near=0.01,
                        far=2.0,
                    ),
                ),
            )

        # Gymnasium TimeLimit returns truncated=True forever after first timeout
        # and collapses critic targets to reward-only; MS wrapper auto-resets instead.
        self._env = ManiSkillVectorEnv(
            base_env,
            num_envs=num_envs,
            auto_reset=True,
            ignore_terminations=False,
        )
        self._last_obs: Optional[Dict[str, Any]] = None
        self._arm_seg_ids: list = []
        self._cache_arm_seg_ids()

    @property
    def unwrapped(self):
        # Scripted experts need privileged poses not exposed in rgbd extra.
        return self._env.unwrapped

    # Public API, mirrors envs/wrapper.py:EnvWrapper.

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _info = self._env.reset(seed=self.seed)
        self._last_obs = obs_dict
        return self._extract(obs_dict)

    def step(self, actions: torch.Tensor) -> EnvStep:
        ms3_action = self._physical_to_ms3(actions.to(self._device).float())
        obs_dict, rewards, terminated, truncated, info = self._env.step(ms3_action)
        self._last_obs = obs_dict

        state, obs = self._extract(obs_dict)
        done = (terminated | truncated).float().reshape(self.batch_size).cpu()
        reward = rewards.float().reshape(self.batch_size).cpu()

        # MS3 dense rewards are positive throughout, so reward>0 != success.
        # info['success'] is authoritative.
        success_raw = info.get("success")
        if success_raw is None:
            success = torch.zeros(self.batch_size, dtype=torch.float32)
        else:
            success = success_raw.float().reshape(self.batch_size).cpu()

        return EnvStep(state=state, obs=obs, reward=reward, done=done, success=success)

    def get_expert_action(self) -> torch.Tensor:
        # Fail loud if no expert was registered (otherwise warmup no-ops).
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

    # Internals.

    def _extract(self, obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_raw = obs_dict["sensor_data"]["overhead_camera"]["depth"]
        depth_m = depth_raw.float() * 0.001  # (B, H, W, 1) mm -> m
        depth_m = depth_m.squeeze(-1)  # (B, H, W)

        # Gripper z from privileged TCP pose; fall back to 0 if missing.
        extra = obs_dict.get("extra", {})
        if "tcp_pose" in extra:
            gripper_z = extra["tcp_pose"][:, 2:3]  # (B, 1)
        else:
            gripper_z = torch.zeros(depth_m.shape[0], 1, device=depth_m.device)

        depth_rel = depth_m - gripper_z.unsqueeze(-1)  # (B, H, W)
        depth_rel = torch.nan_to_num(depth_rel, nan=0.0, posinf=0.0, neginf=0.0)
        depth_rel = depth_rel.clamp(-self.depth_max, self.depth_max)
        obs = (
            depth_rel.unsqueeze(1)
            .cpu()
            .reshape(self.batch_size, 1, self.obs_size, self.obs_size)
        )

        # Binary holding indicator: sum of finger qpos < 0.04 m means grasping.
        qpos = obs_dict["agent"]["qpos"]
        finger_width = qpos[:, -2] + qpos[:, -1]
        state = (
            (finger_width < 0.04).float().cpu().reshape(self.batch_size, self.state_dim)
        )
        return state, obs

    def _physical_to_ms3(self, physical: torch.Tensor) -> torch.Tensor:
        # pxyzr -> pd_ee_delta_pose [dx, dy, dz, drx, dry, drz, gripper].
        # Gripper: BulletArm 0=open/1=closed -> MS +1=open/-1=closed.
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

    # Links to exclude from the depth heightmap; fingers stay visible.
    _ARM_LINKS_TO_MASK = [
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]

    def _cache_arm_seg_ids(self) -> None:
        # Per-env actor seg IDs. Read-only so it's safe after render finalization
        # (unlike set_visibility).
        try:
            robot = self._env.unwrapped.agent.robot
            links_map = {link.name: link for link in robot.links}
            self._arm_seg_ids = [set() for _ in range(self.batch_size)]
            for link_name in self._ARM_LINKS_TO_MASK:
                if link_name not in links_map:
                    continue
                for i, obj in enumerate(links_map[link_name]._objs):
                    if i < self.batch_size:
                        self._arm_seg_ids[i].add(int(obj.entity.per_scene_id))
        except Exception as exc:
            print(
                f"[ManiSkillWrapper] arm seg-id caching failed ({exc}); arm will be visible in depth"
            )
            self._arm_seg_ids = []
