"""SO(2) replay-buffer aug (Wang et al. 2022 Fig 7).

Every transition is stored K+1 times: once as-is plus K random-theta rotated copies.
Rotation acts on (obs, next_obs, action[xy]); p, dz, dtheta are invariant.

rotate_xy uses the SAME theta as rotate_obs so stored (obs, action) pairs stay
aligned under the equi actor's irrep(1) output.

CPU + torch only. Thetas uniform in [-pi, pi). xy clipped to [-1, 1] after
rotation to preserve the buffer's unscaled-action invariant.
"""

import math

import torch
import torch.nn.functional as F


def sample_thetas(n: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform in [-pi, pi). Shape (n,), float32, CPU."""
    u = torch.rand(n, generator=generator, dtype=torch.float32)
    return (u - 0.5) * (2.0 * math.pi)


def rotate_obs(
    obs: torch.Tensor,
    thetas: torch.Tensor,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Rotate each (C, H, W) image by its own theta with bilinear interp."""
    if obs.dim() != 4:
        raise ValueError(f"rotate_obs expects (B, C, H, W), got {tuple(obs.shape)}")
    if thetas.shape[0] != obs.shape[0]:
        raise ValueError(
            f"thetas shape {tuple(thetas.shape)} must match batch dim {obs.shape[0]}"
        )

    cos_t = thetas.cos()
    sin_t = thetas.sin()

    # y-axis-down + grid_sample's inverse-transform expectation cancel, so
    # +theta in the affine matrix gives +theta CCW (matches torch.rot90(k=1)).
    theta_mat = torch.zeros(obs.shape[0], 2, 3, dtype=obs.dtype, device=obs.device)
    theta_mat[:, 0, 0] = cos_t
    theta_mat[:, 0, 1] = -sin_t
    theta_mat[:, 1, 0] = sin_t
    theta_mat[:, 1, 1] = cos_t

    grid = F.affine_grid(theta_mat, obs.shape, align_corners=False)
    return F.grid_sample(
        obs, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False
    )


def rotate_xy(xy: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Rotate (dx, dy) pairs by each item's theta. xy is (B, 2) float32."""
    if xy.shape[-1] != 2:
        raise ValueError(f"rotate_xy expects last dim=2, got {tuple(xy.shape)}")

    cos_t = thetas.cos()
    sin_t = thetas.sin()
    x_new = cos_t * xy[:, 0] - sin_t * xy[:, 1]
    y_new = sin_t * xy[:, 0] + cos_t * xy[:, 1]
    return torch.stack([x_new, y_new], dim=-1)


def augment_transition_so2(
    state: torch.Tensor,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    thetas: torch.Tensor,
) -> tuple:
    """SO(2) rotation on a batch of transitions. Assumes action layout [p, dx, dy, dz, dr]."""
    obs_rot = rotate_obs(obs, thetas)
    next_obs_rot = rotate_obs(next_obs, thetas)

    # Rotate xy by same theta as obs to match the equi actor's irrep(1) output.
    xy_rot = rotate_xy(action[:, 1:3], thetas).clamp(-1.0, 1.0)
    action_rot = action.clone()
    action_rot[:, 1:3] = xy_rot

    # Fresh clones so callers can't alias augmented views back into buffer storage.
    return (
        state.clone(),
        obs_rot,
        action_rot,
        reward.clone(),
        next_state.clone(),
        next_obs_rot,
        done.clone(),
    )
