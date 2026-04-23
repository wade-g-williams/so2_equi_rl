"""Scripted experts for ManiSkill tasks used by SAC warmup.

Reads the wrapper's last obs (privileged TCP + object poses) and emits 5-D
physical actions in pxyzr layout. Each expert is a short PD-to-target loop;
warmup tolerates 40-70% success so rough heuristics are fine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

import torch

if TYPE_CHECKING:
    from so2_equi_rl.envs.maniskill_wrapper import ManiSkillWrapper


# Per-step physical limits; must match SACConfig.dpos / SACConfig.drot.
# Systematic clipping from a mismatch bleeds warmup signal.
_DPOS = 0.05  # meters per step
_DROT = 0.39269908  # pi / 8 radians per step


def _toward(target_xyz: torch.Tensor, tcp_xyz: torch.Tensor) -> torch.Tensor:
    # Saturate each axis at per-step dpos so encode_action doesn't clip.
    delta = target_xyz - tcp_xyz
    return torch.clamp(delta, -_DPOS, _DPOS)


def _pack(dxyz: torch.Tensor, gripper_p: torch.Tensor) -> torch.Tensor:
    # (B, 3) + (B, 1) -> (B, 5) in [p, dx, dy, dz, dtheta]. dtheta=0 for PickCube/PullCube.
    B = dxyz.shape[0]
    dtheta = torch.zeros(B, 1, device=dxyz.device, dtype=dxyz.dtype)
    return torch.cat([gripper_p, dxyz, dtheta], dim=1)


def pick_cube_expert(wrapper: "ManiSkillWrapper") -> torch.Tensor:
    # Three phases from |xy_err| and |z_err|: hover, descend open, close+lift.
    # Stateless across auto-resets.
    obs = wrapper._last_obs
    assert obs is not None, "reset() must run before get_expert_action"

    # rgbd obs_mode hides obj_pose from extra, so read it off the task entity.
    tcp_xyz = obs["extra"]["tcp_pose"][:, 0:3]
    obj_xyz = wrapper.unwrapped.cube.pose.p

    xy_err = (tcp_xyz[:, 0:2] - obj_xyz[:, 0:2]).abs().max(dim=1).values  # (B,)
    above_cube = torch.stack(
        [obj_xyz[:, 0], obj_xyz[:, 1], obj_xyz[:, 2] + 0.12], dim=1
    )

    # Misaligned -> hover, else descend.
    hover = xy_err > 0.02

    target = torch.where(hover.unsqueeze(1), above_cube, obj_xyz)
    dxyz = _toward(target, tcp_xyz)

    # Close once xy is within ~2 cm and z reaches cube height (BulletArm 1 = closed).
    close_mask = (~hover) & ((tcp_xyz[:, 2] - obj_xyz[:, 2]).abs() < 0.03)
    gripper_p = close_mask.float().unsqueeze(1)

    return _pack(dxyz, gripper_p)


def pull_cube_expert(wrapper: "ManiSkillWrapper") -> torch.Tensor:
    # Two phases: position TCP on far side of cube, then translate toward goal.
    # Gripper stays closed.
    obs = wrapper._last_obs
    assert obs is not None, "reset() must run before get_expert_action"

    # PullCube names the cube 'obj' and the target 'goal_region'.
    tcp_xyz = obs["extra"]["tcp_pose"][:, 0:3]
    obj_xyz = wrapper.unwrapped.obj.pose.p
    goal_xyz = wrapper.unwrapped.goal_region.pose.p

    # Approach point on far side of cube from goal.
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

    # Fingers act as a flat pusher (BulletArm 1 = closed).
    B = tcp_xyz.shape[0]
    gripper_p = torch.ones(B, 1, device=tcp_xyz.device, dtype=tcp_xyz.dtype)

    return _pack(dxyz, gripper_p)


def stack_cube_expert(wrapper: "ManiSkillWrapper") -> torch.Tensor:
    # Two branches selected by whether cubeA is carried (xy-tight + lifted).
    # Pick branch: hover, descend, close. Place branch: hover over cubeB, descend,
    # release. Success requires ~is_cubeA_grasped so the gripper must open at the end.
    obs = wrapper._last_obs
    assert obs is not None, "reset() must run before get_expert_action"

    tcp_xyz = obs["extra"]["tcp_pose"][:, 0:3]
    cubeA_xyz = wrapper.unwrapped.cubeA.pose.p
    cubeB_xyz = wrapper.unwrapped.cubeB.pose.p

    # "Carried" = xy-tight and lifted above resting z (~0.02; >0.05 = lifted).
    tcp_to_A_xy = (tcp_xyz[:, 0:2] - cubeA_xyz[:, 0:2]).norm(dim=1)
    carried = (tcp_to_A_xy < 0.025) & (cubeA_xyz[:, 2] > 0.05)

    # Pick branch.
    above_A = torch.stack(
        [cubeA_xyz[:, 0], cubeA_xyz[:, 1], cubeA_xyz[:, 2] + 0.12], dim=1
    )
    xy_err_A = (tcp_xyz[:, 0:2] - cubeA_xyz[:, 0:2]).abs().max(dim=1).values
    hover_A = xy_err_A > 0.02
    pick_target = torch.where(hover_A.unsqueeze(1), above_A, cubeA_xyz)
    close_now = (~hover_A) & ((tcp_xyz[:, 2] - cubeA_xyz[:, 2]).abs() < 0.03)
    pick_gripper = close_now.float().unsqueeze(1)  # 1 = close

    # Place branch. Stack z = cubeB.z + (half_A + half_B) = +0.04.
    stack_z = cubeB_xyz[:, 2] + 0.04
    above_B = torch.stack([cubeB_xyz[:, 0], cubeB_xyz[:, 1], stack_z + 0.10], dim=1)
    stack_pos = torch.stack([cubeB_xyz[:, 0], cubeB_xyz[:, 1], stack_z], dim=1)
    xy_err_B = (tcp_xyz[:, 0:2] - cubeB_xyz[:, 0:2]).abs().max(dim=1).values
    hover_B = xy_err_B > 0.02
    place_target = torch.where(hover_B.unsqueeze(1), above_B, stack_pos)
    # Open at the stack pose; success requires release.
    at_stack = (~hover_B) & ((tcp_xyz[:, 2] - stack_z).abs() < 0.02)
    place_gripper = (~at_stack).float().unsqueeze(1)  # 1 = closed, 0 = open at stack

    target = torch.where(carried.unsqueeze(1), place_target, pick_target)
    gripper_p = torch.where(carried.unsqueeze(1), place_gripper, pick_gripper)
    dxyz = _toward(target, tcp_xyz)

    return _pack(dxyz, gripper_p)


# MS3 task id -> scripted expert. Keys match _SUPPORTED_MS3_TASKS.
EXPERTS: Dict[str, Callable[["ManiSkillWrapper"], torch.Tensor]] = {
    "PickCube-v1": pick_cube_expert,
    "PullCube-v1": pull_cube_expert,
    "StackCube-v1": stack_cube_expert,
    "EquiPickCube-v1": pick_cube_expert,
}


def get_expert(env_name: str) -> Callable[["ManiSkillWrapper"], torch.Tensor]:
    if env_name not in EXPERTS:
        raise NotImplementedError(
            f"No ManiSkill expert for env_name={env_name!r}. "
            f"Available: {sorted(EXPERTS)}"
        )
    return EXPERTS[env_name]
