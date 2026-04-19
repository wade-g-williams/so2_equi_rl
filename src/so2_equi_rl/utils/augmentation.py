"""SO(2) rotation helpers for obs and action batches. Used by RAD and
DrQ to rotate obs, next_obs, and the planar (dx, dy) part of the action
by per-row angles. Runs on replay-sampled batches so there's no gradient
to carry and no hidden state between calls.

rotate_obs uses bilinear grid_sample with padding_mode='border', which
matches scipy.ndimage.rotate(mode='nearest') on the zero-background
heightmap.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

_VALID_MODES = ("continuous", "discrete_cN")


def _broadcast_theta(theta: torch.Tensor, batch: int) -> torch.Tensor:
    # Accept a scalar, (1,), or (B,) so callers don't have to pre-expand a single angle.
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
        # Uniform on [0, 2pi). group_order is unused here.
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

    # Border padding matches scipy rotate(mode='nearest'). Zero padding
    # would bleed the outside in and diverge from the paper.
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

    # Action layout from sac.py::decode_action: 0 = p, 1 = dx, 2 = dy,
    # 3 = dz, 4 = dtheta. Only cols 1-2 transform as irrep(1).
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    dx = action[:, 1]
    dy = action[:, 2]
    rotated_dx = cos * dx - sin * dy
    rotated_dy = sin * dx + cos * dy

    # Rebuild out-of-place so the op is autograd-safe and handles
    # action tensors with > 5 columns.
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

    Returns the rotated tuple. State, reward, and done are SO(2)-invariant
    and stay with the caller.
    """
    B = obs.shape[0]
    theta = sample_so2_angles(
        B, mode=mode, group_order=group_order, generator=generator
    )

    # Shared theta per row across all three is the equivariance contract.
    rotated_obs = rotate_obs(obs, theta)
    rotated_next_obs = rotate_obs(next_obs, theta)
    rotated_action = rotate_action_dxy(action, theta)
    return rotated_obs, rotated_next_obs, rotated_action
