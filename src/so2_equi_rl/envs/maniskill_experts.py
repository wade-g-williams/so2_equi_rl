"""Scripted experts for ManiSkill tasks, used by SACTrainer's warmup phase.
Reads the wrapper's last obs dict (privileged TCP and object poses) and
emits physical 5-D actions in the SAC pxyzr layout. Warmup doesn't need
oracle experts (40-70% success is fine), so each is a short PD-to-target
loop with no task-specific tuning beyond goal selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

import torch

if TYPE_CHECKING:
    from so2_equi_rl.envs.maniskill_wrapper import ManiSkillWrapper


# Per-step physical limits. Must stay consistent with SACConfig.dpos /
# SACConfig.drot so the encoded [-1, 1] action the trainer pushes into
# the buffer is in-range (SACAgent.encode_action clamps, but a
# systematically clipped expert bleeds the warmup signal).
_DPOS = 0.05  # meters per step
_DROT = 0.39269908  # pi / 8 radians per step


def _toward(target_xyz: torch.Tensor, tcp_xyz: torch.Tensor) -> torch.Tensor:
    # Saturate each axis at the per-step dpos so the expert isn't
    # clamped by encode_action. Returns (B, 3) meters.
    delta = target_xyz - tcp_xyz
    return torch.clamp(delta, -_DPOS, _DPOS)


def _pack(dxyz: torch.Tensor, gripper_p: torch.Tensor) -> torch.Tensor:
    # (B, 3) + (B, 1) -> (B, 5) in [p, dx, dy, dz, dtheta] layout.
    # dtheta left at 0: neither PickCube nor PullCube need yaw control.
    B = dxyz.shape[0]
    dtheta = torch.zeros(B, 1, device=dxyz.device, dtype=dxyz.dtype)
    return torch.cat([gripper_p, dxyz, dtheta], dim=1)


def pick_cube_expert(wrapper: "ManiSkillWrapper") -> torch.Tensor:
    # Three phases driven from the privileged TCP and cube pose:
    #   1. hover above cube (xy align, z held at cube_z + clearance)
    #   2. descend to cube with gripper open
    #   3. close gripper and lift
    # Phase selection reads |xy_err| and |z_err|; no stateful machine
    # (no per-env counters to keep across auto-resets).
    obs = wrapper._last_obs
    assert obs is not None, "reset() must run before get_expert_action"

    # rgbd obs_mode hides obj_pose from `extra`; read directly from the
    # task entity so the expert works without switching obs_mode.
    tcp_xyz = obs["extra"]["tcp_pose"][:, 0:3]
    obj_xyz = wrapper.unwrapped.cube.pose.p

    xy_err = (tcp_xyz[:, 0:2] - obj_xyz[:, 0:2]).abs().max(dim=1).values  # (B,)
    above_cube = torch.stack(
        [obj_xyz[:, 0], obj_xyz[:, 1], obj_xyz[:, 2] + 0.12], dim=1
    )

    # Phase mask: True when we still need to align xy before descending.
    hover = xy_err > 0.02

    target = torch.where(hover.unsqueeze(1), above_cube, obj_xyz)
    dxyz = _toward(target, tcp_xyz)

    # Gripper: keep open until we're within ~2 cm of the cube in xy and
    # roughly at cube height, then close. BulletArm convention: 1 = closed.
    close_mask = (~hover) & ((tcp_xyz[:, 2] - obj_xyz[:, 2]).abs() < 0.03)
    gripper_p = close_mask.float().unsqueeze(1)

    return _pack(dxyz, gripper_p)


def pull_cube_expert(wrapper: "ManiSkillWrapper") -> torch.Tensor:
    # Two-phase cube-pulling expert. PullCube's goal is pushing the cube
    # along a 2D surface toward a goal region:
    #   1. position TCP on the far side of the cube (relative to goal)
    #   2. translate toward the goal, dragging the cube
    # Gripper stays closed throughout (acts as a pusher-nose).
    obs = wrapper._last_obs
    assert obs is not None, "reset() must run before get_expert_action"

    # PullCube names the cube `obj` and the target `goal_region` (PickCube
    # uses `cube`/`goal_site`). Use ManiSkill's per-task entity names here;
    # `extra` doesn't surface these in rgbd obs_mode.
    tcp_xyz = obs["extra"]["tcp_pose"][:, 0:3]
    obj_xyz = wrapper.unwrapped.obj.pose.p
    goal_xyz = wrapper.unwrapped.goal_region.pose.p

    # Approach point = cube position shifted opposite to the goal
    # direction, nudged down to the table surface.
    to_goal = goal_xyz[:, 0:2] - obj_xyz[:, 0:2]
    to_goal = to_goal / (to_goal.norm(dim=1, keepdim=True).clamp_min(1e-6))
    approach_xy = obj_xyz[:, 0:2] - 0.05 * to_goal
    approach = torch.cat([approach_xy, obj_xyz[:, 2:3]], dim=1)

    xy_err = (tcp_xyz[:, 0:2] - approach_xy).abs().max(dim=1).values
    target = torch.where(
        (xy_err > 0.02).unsqueeze(1),
        approach,
        goal_xyz,
    )
    dxyz = _toward(target, tcp_xyz)

    # Gripper closed (BulletArm 1), fingers act as a flat pusher.
    B = tcp_xyz.shape[0]
    gripper_p = torch.ones(B, 1, device=tcp_xyz.device, dtype=tcp_xyz.dtype)

    return _pack(dxyz, gripper_p)


# close_loop_* task id → expert function.
EXPERTS: Dict[str, Callable[["ManiSkillWrapper"], torch.Tensor]] = {
    "close_loop_block_picking": pick_cube_expert,
    "close_loop_block_pulling": pull_cube_expert,
}


def get_expert(env_name: str) -> Callable[["ManiSkillWrapper"], torch.Tensor]:
    if env_name not in EXPERTS:
        raise NotImplementedError(
            f"No ManiSkill expert for env_name={env_name!r}. "
            f"Available: {sorted(EXPERTS)}"
        )
    return EXPERTS[env_name]
