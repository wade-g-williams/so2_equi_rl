"""ManiSkill 3 wrapper. Matches EnvWrapper's public API so trainers stay
backend-agnostic. Top-down narrow-FOV camera approximates an orthographic
view (<1% edge distortion at TrainConfig defaults, small enough for the
equi encoder to absorb). Physical action is pxyzr like BulletArm;
_physical_to_ms3 remaps into pd_ee_delta_pose.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import sapien
import so2_equi_rl.envs.equi_pick_cube  # noqa: F401 — triggers @register_env("EquiPickCube-v1")
from so2_equi_rl.envs import EnvStep

# Supported ManiSkill tasks for the paper-reproduction extension. PickCube
# and PullCube are direct analogues of BulletArm's block_picking and
# block_pulling. StackCube stands in for drawer_opening because
# ManiSkill's OpenCabinetDrawer-v1 is a Fetch-based mobile-manipulation
# task, not a tabletop one.
_SUPPORTED_MS3_TASKS = {"PickCube-v1", "PullCube-v1", "StackCube-v1", "EquiPickCube-v1"}

# User-facing env name -> internal arm-hidden variant. The variant is
# registered the first time ManiSkillWrapper is constructed and shares
# task logic with the base class via a mixin that hides arm links inside
# _load_scene. Has to happen there (not after gym.make) because MS3's
# batched render system locks the scene once it's built.
_TASK_TO_EQUI = {
    "PickCube-v1": "EquiPickCube-v1",
    "PullCube-v1": "EquiPullCube-v1",
    "StackCube-v1": "EquiStackCube-v1",
}

_equi_tasks_registered = False


# ManiSkill's default pd_ee_delta_pose bounds. We rescale the agent's
# physical action into the [-1, 1] cube these bounds describe.
_MS3_POS_BOUND = 0.1  # meters, matches pd_ee_delta_pose default
_MS3_ROT_BOUND = 0.1  # radians, matches pd_ee_delta_pose default

# Panda arm and hand links to hide from the sensor camera. Fingers stay
# visible so the heightmap still shows the grasp. Without this, the arm
# enters the workspace as a direction-dependent feature and breaks the
# SO(2) symmetry the Equi encoder relies on.
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
    # Register EquiPickCube-v1, EquiPullCube-v1, EquiStackCube-v1. Each
    # subclasses the matching base task and, inside _load_scene, walks the
    # Panda articulation and drops visibility to 0 on every RenderBody
    # belonging to a link in _PANDA_HIDDEN_LINKS. Physics components ignore
    # the attribute, so collisions and grasping still work.
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
        # Overhead fixed camera at 1 m, 60 deg FOV, 128x128. Matches Joey's
        # eq_sac maniskill branch. Replaces every default sensor camera with
        # this single top-down rig, so obs["sensor_data"] only carries
        # "overhead_camera".
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
        # Lazy-imported so environments without ManiSkill still import the package.
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401, registers ManiSkill envs
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        if cfg.env_name not in _SUPPORTED_MS3_TASKS:
            raise NotImplementedError(
                f"env_name={cfg.env_name!r} is not a supported ManiSkill task. "
                f"Available: {sorted(_SUPPORTED_MS3_TASKS)}"
            )

        # Register arm-hidden subclasses on first use and remap env_name to
        # the Equi variant so gym.make builds the scene with arm visibility
        # dropped during _load_scene, before the batched render system locks.
        _register_equi_tasks()
        # equi_env_name = _TASK_TO_EQUI[cfg.env_name]

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

        if self.env_name == "EquiPickCube-v1":
            # Camera and arm-hiding are handled by the EquiPickCubeEnv subclass
            # via _default_sensor_configs / _load_scene, so no sensor_configs override.
            # Camera is hardcoded at z=1.0 in equi_pick_cube.py, so override the
            # cfg value here so _extract's height = camera_height - depth_m is correct.
            self._camera_key = "overhead_camera"
            self.camera_height = 1.0
            base_env = gym.make(
                cfg.env_name,
                num_envs=num_envs,
                obs_mode="rgbd",  # "depth" returns float32 m; "rgbd" returns int16 mm matching _extract
                reward_mode=cfg.ms3_reward_mode,
                control_mode=cfg.ms3_control_mode,
                sim_backend=cfg.ms3_sim_backend,
            )
            # base_env = gym.make(
            #     "EquiPickCube-v1",
            #     obs_mode="depth",
            #     control_mode="pd_ee_delta_pose",
            #     reward_mode="normalized_dense",
            #     num_envs=5,
            #     sensor_configs=dict(width=128, height=128),
            # )
        else:
            # gym.make builds the task, sensor_configs override the default
            # camera named base_camera with our top-down rig.
            camera_height = (
                self.camera_height
            )  # keep placement consistent with _extract
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
                        fov=np.pi / 3,  # 60 degrees FOV — covers ~1.15m at 1m height
                        near=0.01,
                        far=2.0,
                        # pose=pose_list,
                        # width=self.obs_size,
                        # height=self.obs_size,
                        # fov=float(np.deg2rad(cfg.ms3_camera_fov)),
                        # near=0.01,
                        # far=float(cfg.ms3_camera_height) + 0.5,
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
        self._arm_seg_ids: list = []
        self._cache_arm_seg_ids()

    @property
    def unwrapped(self):
        # Access for scripted experts that need privileged poses (e.g.
        # cube position) which rgbd obs_mode doesn't include in `extra`.
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

        # Task-internal success predicate. MS3's dense/normalized_dense rewards
        # are positive throughout the episode, so reward>0 does not imply
        # success. info['success'] is authoritative.
        success_raw = info.get("success")
        if success_raw is None:
            success = torch.zeros(self.batch_size, dtype=torch.float32)
        else:
            success = success_raw.float().reshape(self.batch_size).cpu()

        return EnvStep(state=state, obs=obs, reward=reward, done=done, success=success)

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

        # Binary holding indicator. Sum of both finger qpos; < 0.04 m means
        # the gripper has closed on something (Joey's threshold).
        qpos = obs_dict["agent"]["qpos"]
        finger_width = qpos[:, -2] + qpos[:, -1]
        state = (
            (finger_width < 0.04).float().cpu().reshape(self.batch_size, self.state_dim)
        )
        return state, obs

    def _physical_to_ms3(self, physical: torch.Tensor) -> torch.Tensor:
        # Input layout: pxyzr (from SACAgent.decode_action).
        # Output layout: pd_ee_delta_pose = [dx, dy, dz, drx, dry, drz, gripper].
        # Normalize deltas by controller bounds, clip to the cube, remap
        # gripper (BulletArm 0=open, 1=closed -> MS +1=open, -1=closed).
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

    # Links to exclude from the depth heightmap (upper arm; fingers are kept
    # so the gripper remains visible as a task-relevant cue).
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
        # Read per-env actor segmentation IDs from the robot links. This
        # only reads scene state so it is safe to call after the batched
        # render system is finalised (unlike set_visibility which is not).
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
