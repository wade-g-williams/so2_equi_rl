"""Image-space augmentation helpers for replay-sampled batches.

random_shift is the DrQ primitive (pad + random HxW crop, default pad=4).
random_crop is the RAD/CURL/FERM primitive, same mechanics at pad=7
(128+14=142 effective, cropped back to 128). Neither rotates the action
since pixel translation doesn't change world-frame deltas.

rotate_obs, rotate_action_dxy, sample_so2_angles, random_so2_augment stay
here for backward-compat with tests and older equivariance experiments;
the paper-faithful SO(2) aug now lives in buffers/so2_aug.py.
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


def random_shift(
    obs: torch.Tensor,
    *,
    pad: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Paper-style DrQ random shift: zero-pad by `pad` pixels on each side,
    then take a random HxW crop from the padded (H+2*pad)x(W+2*pad) tensor.

    Wang et al. 2022 sec E: "Shift baselines use random shift of +/-4 pixels."
    Matches Kostrikov et al. 2020 (DrQ) ShiftsAug on 128x128 heightmaps.

    obs: (B, C, H, W), different per-row shifts
    returns: (B, C, H, W)
    """
    if obs.ndim != 4:
        raise ValueError(f"obs must be 4D (B, C, H, W), got {tuple(obs.shape)}")
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    if pad == 0:
        return obs.clone()

    B, C, H, W = obs.shape
    # Pad with zeros (the heightmap's out-of-workspace area is already 0).
    padded = F.pad(obs, (pad, pad, pad, pad), mode="constant", value=0.0)

    # Sample per-row offsets in [0, 2*pad] so the crop stays inside the padded tensor.
    # cpu generator seeded by caller; offsets are integer pixels.
    dx = torch.randint(0, 2 * pad + 1, (B,), generator=generator)
    dy = torch.randint(0, 2 * pad + 1, (B,), generator=generator)

    out = torch.empty_like(obs)
    for i in range(B):
        out[i] = padded[i, :, dy[i] : dy[i] + H, dx[i] : dx[i] + W]
    return out


def random_crop(
    obs: torch.Tensor,
    *,
    pad: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Paper-style RAD/CURL/FERM random crop: zero-pad then take a HxW crop.

    Paper used 142x142 raw obs cropped to 128x128; we already render at
    128x128, so pad=7 gives (128+14)=142 effective -> 128 crop, matching.

    obs: (B, C, H, W)
    returns: (B, C, H, W)
    """
    # Mechanically identical to random_shift (pad + random-crop-to-HxW) but
    # exposed under the semantic paper uses in sec E ("random crop" for RAD,
    # CURL, FERM; "random shift" for DrQ). Same underlying op, different
    # pad sizes (paper uses pad=7 for crop, pad=4 for shift).
    return random_shift(obs, pad=pad, generator=generator)


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
