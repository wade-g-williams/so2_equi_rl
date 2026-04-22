"""
Custom ManiSkill PickCube environment with overhead depth camera
and arm links hidden from sensor rendering.

Matches the observation setup from:
  Wang, Walters & Platt, "SO(2)-Equivariant Reinforcement Learning", ICLR 2022

  - Fixed overhead camera looking straight down at the workspace center
  - Robot arm links hidden from sensor cameras (only fingers visible)
  - Depth observation relative to gripper height (handled in obs wrapper)

Usage:
  import equi_pick_cube  # registers the task
  env = gym.make("EquiPickCube-v1", obs_mode="depth", control_mode="pd_ee_delta_pose", ...)
"""

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env


@register_env("EquiPickCube-v1", max_episode_steps=100)
class EquiPickCubeEnv(PickCubeEnv):
    """
    PickCube with an overhead depth camera and arm links hidden.

    Changes from base PickCube:
      1. Replaces default sensor cameras with a single fixed overhead camera
         centered above the workspace, looking straight down.
      2. Hides robot arm links (but not gripper fingers) from sensor rendering
         so the depth image shows only the workspace, objects, and fingers.
    """

    # Panda arm link names to hide (everything except fingers and hand)
    PANDA_ARM_LINKS = [
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

    @property
    def _default_sensor_configs(self):
        """
        Override sensor cameras single to a overhead camera looking straight down.
        """
        camera_height = 1.0
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
        return [
            CameraConfig(
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
            )
        ]

    def _load_scene(self, options):
        """Load the scene, then hide arm links from sensor cameras."""
        super()._load_scene(options)
        self._hide_arm_links()

    def _hide_arm_links(self):
        """Hide robot arm links from sensor rendering."""
        robot = self.agent.robot
        links_map = robot.links_map

        for link_name in self.PANDA_ARM_LINKS:
            if link_name in links_map:
                link = links_map[link_name]
                for obj in link._objs:
                    entity = obj.entity
                    for comp in entity.components:
                        if hasattr(comp, "visibility"):
                            comp.visibility = 0.0

    @property
    def _default_human_render_camera_configs(self):
        """Keep the default human render camera for GUI/video viewing."""
        # Use parent's default + add our overhead view for reference
        parent_configs = super()._default_human_render_camera_configs
        return parent_configs
