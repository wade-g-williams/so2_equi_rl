"""SO(2) rotation helpers for obs and action batches. Used by RAD/DrQ-style
augmentation to rotate obs, next_obs, and the planar part of the action by
one shared per-row angle. Runs as a data transform on the replay-sampled
batch, so there's no gradient to carry and no hidden state between calls.

rotate_obs goes through bilinear grid_sample with padding_mode='border'.
Border padding matches scipy.ndimage.rotate(mode='nearest') on the
zero-background heightmap. Bilinear replaces scipy's default cubic because
the heightmap is smooth enough that interpolation order stops mattering,
and grid_sample is cheap on GPU. rotate_action_dxy rotates the (dx, dy)
columns of the unscaled action and passes the rest through.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# Valid mode strings for sample_so2_angles / random_so2_augment.
_VALID_MODES = ("continuous", "discrete_cN")


def _broadcast_theta(theta: torch.Tensor, batch: int) -> torch.Tensor:
    # Accept a scalar, a (1,), or a (B,) tensor so callers can pass a single
    # angle without pre-expanding. Returned shape is always (B,).
    if theta.ndim == 0 or theta.shape == (1,):
        theta = theta.expand(batch)
    if theta.shape != (batch,):
        raise ValueError(
            f"theta must be shape (B,), got {tuple(theta.shape)} for B={batch}"
        )
    return theta


def sample_so2_angles(
    batch: int,
    mode: str,
    group_order: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample `batch` rotation angles in radians, shape (B,) float32.

    Output device follows generator.device.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    if mode == "continuous":
        # Uniform on [0, 2pi). Paper default aug_type='so2'. group_order is
        # unused in this branch, so we skip validating it here.
        return torch.rand(batch, generator=generator) * (2.0 * math.pi)

    # Discrete C_N lattice: k ~ Uniform{0, ..., N-1}, theta = 2 pi k / N.
    if group_order < 1:
        raise ValueError(f"group_order must be >= 1, got {group_order}")
    k = torch.randint(0, group_order, (batch,), generator=generator)
    return k.to(torch.float32) * (2.0 * math.pi / group_order)


def rotate_obs(obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate a (B, C, H, W) obs batch CCW by per-row theta via grid_sample."""
    if obs.ndim != 4:
        raise ValueError(f"obs must be 4D (B, C, H, W), got shape {tuple(obs.shape)}")
    if obs.shape[-1] != obs.shape[-2]:
        raise ValueError(
            f"obs must be spatially square, got HxW = {obs.shape[-2]}x{obs.shape[-1]}"
        )

    B = obs.shape[0]
    theta = _broadcast_theta(theta, B).to(device=obs.device, dtype=obs.dtype)

    # grid_sample pulls values from the input (inverse map), AND its y axis
    # points down. Those two flips cancel, so the affine here uses +theta and
    # the image content ends up rotated +theta CCW from the caller's point of
    # view (matching torch.rot90(k=1) at pi/2). DO NOT FLIP THIS SIGN.
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    affine = torch.zeros(B, 2, 3, device=obs.device, dtype=obs.dtype)
    affine[:, 0, 0] = cos
    affine[:, 0, 1] = -sin
    affine[:, 1, 0] = sin
    affine[:, 1, 1] = cos

    # Border padding matches scipy rotate(mode='nearest') on the zero-background
    # heightmap. Zero padding would bleed the outside into the rotated image
    # and break paper faithfulness.
    grid = F.affine_grid(affine, obs.shape, align_corners=False)
    return F.grid_sample(
        obs,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def rotate_action_dxy(action: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate the (dx, dy) columns of an unscaled action batch by per-row theta."""
    if action.ndim != 2:
        raise ValueError(f"action must be 2D (B, A), got shape {tuple(action.shape)}")
    if action.shape[1] < 5:
        raise ValueError(
            f"action must have >= 5 columns (p, dx, dy, dz, dtheta), got {action.shape[1]}"
        )

    B = action.shape[0]
    theta = _broadcast_theta(theta, B).to(device=action.device, dtype=action.dtype)

    # Action column layout from sac.py::decode_action:
    # 0 = p (gripper), 1 = dx, 2 = dy, 3 = dz, 4 = dtheta.
    # Only columns 1-2 transform as irrep(1); the rest are invariant.
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    dx = action[:, 1]
    dy = action[:, 2]
    rotated_dx = cos * dx - sin * dy
    rotated_dy = sin * dx + cos * dy

    # Rebuild the action out-of-place so the op is autograd-safe and so it
    # handles action tensors with more than five columns without a special case.
    cols = [action[:, 0], rotated_dx, rotated_dy] + [
        action[:, i] for i in range(3, action.shape[1])
    ]
    return torch.stack(cols, dim=1)


def random_so2_augment(
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    action: torch.Tensor,
    *,
    mode: str = "continuous",
    group_order: int = 8,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one shared per-row SO(2) rotation to obs, next_obs, and action.

    Returns the rotated (obs, next_obs, action) tuple. The scalar state
    channels are SO(2)-invariant, so this function only takes the tensors
    that transform under rotation and leaves state/reward/done to the caller.
    """
    B = obs.shape[0]
    theta = sample_so2_angles(
        B, mode=mode, group_order=group_order, generator=generator
    )

    # Shared theta per row across obs / next_obs / action is the equivariance
    # contract: break it and the augmentation stops being a symmetry.
    rotated_obs = rotate_obs(obs, theta)
    rotated_next_obs = rotate_obs(next_obs, theta)
    rotated_action = rotate_action_dxy(action, theta)
    return rotated_obs, rotated_next_obs, rotated_action
