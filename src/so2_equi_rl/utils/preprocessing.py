"""Tile the scalar gripper state onto the heightmap so the encoder sees
one (C+S, H, W) tensor.
"""

import torch


def tile_state(obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    # obs (B, C, H, W) + state (B, S) -> (B, C+S, H, W). Each state scalar
    # broadcasts to a full (H, W) plane stacked onto the heightmap.
    batch, _, height, width = obs.shape
    state_dim = state.shape[1]
    state_plane = state.view(batch, state_dim, 1, 1).expand(
        batch, state_dim, height, width
    )
    return torch.cat([obs, state_plane], dim=1)
