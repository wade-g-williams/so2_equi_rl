"""SO(2) data-augmentation helpers for the replay buffer.

Wang et al. 2022 Fig 7 training pipeline: every env transition is stored K+1
times, once as-is, then K rotated copies with a fresh random angle per copy.
The rotation applies to (obs, next_obs, action[xy]) as a rigid world rotation
about the image center. Position deltas (dx, dy) rotate by R(theta);
primitive/gripper (p), depth delta (z), and yaw delta (r) are invariant
under world rotation (r is a rotation *delta*, not an absolute frame).

Implementation notes
- obs is a depth heightmap (C, H, W), float32. Rotation uses bilinear
  grid_sample which is differentiable but we don't need gradients here;
  we just want to stay on CPU + torch to avoid a scipy dependency.
- Rotations are sampled uniformly in [-pi, pi].
- xy are clipped to [-1, 1] after rotation (paper does the same; the buffer's
  unscaled-action invariant requires it anyway).
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
    """Rotate each (C, H, W) image in a batch by its own theta.

    obs:     (B, C, H, W) float32
    thetas:  (B,) float32, radians
    returns: (B, C, H, W) float32

    Uses an inverse affine grid (grid_sample expects inverse) + bilinear
    interp. Padding mode matches scipy.ndimage.affine_transform 'zeros'
    default, which is what the paper uses for out-of-frame pixels.
    """
    if obs.dim() != 4:
        raise ValueError(f"rotate_obs expects (B, C, H, W), got {tuple(obs.shape)}")
    if thetas.shape[0] != obs.shape[0]:
        raise ValueError(
            f"thetas shape {tuple(thetas.shape)} must match batch dim {obs.shape[0]}"
        )

    cos_t = thetas.cos()
    sin_t = thetas.sin()

    # grid_sample expects the inverse transform (output -> input). For a
    # rotation by +theta of the image, the inverse rotation is by -theta.
    theta_mat = torch.zeros(obs.shape[0], 2, 3, dtype=obs.dtype, device=obs.device)
    theta_mat[:, 0, 0] = cos_t
    theta_mat[:, 0, 1] = sin_t
    theta_mat[:, 1, 0] = -sin_t
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
    """Apply SO(2) rotation to a single batch of transitions.

    Inputs are (B, ...) tensors; thetas is (B,). Returns the same shapes
    with obs + next_obs + action[xy] rotated. Scalars (state, reward,
    next_state, done) are invariant under world rotation.

    Assumes action_dim == 5 with layout [p, dx, dy, dz, dr]. Enforced by
    caller (ReplayBuffer.push sees action_dim at construction time).
    """
    obs_rot = rotate_obs(obs, thetas)
    next_obs_rot = rotate_obs(next_obs, thetas)

    xy_rot = rotate_xy(action[:, 1:3], thetas).clamp(-1.0, 1.0)
    action_rot = action.clone()
    action_rot[:, 1:3] = xy_rot

    # state, reward, next_state, done: invariant. Return fresh refs so the
    # caller can't accidentally alias augmented views into the buffer
    # storage alongside the original.
    return (
        state.clone(),
        obs_rot,
        action_rot,
        reward.clone(),
        next_state.clone(),
        next_obs_rot,
        done.clone(),
    )
