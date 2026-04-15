"""Observation preprocessing: tile the scalar gripper state onto the
heightmap so the encoder sees one 2-channel tensor instead of an (obs, state)
pair.
"""

import torch


def tile_state(obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    # (B,C,H,W) + (B,S) -> (B,C+S,H,W): broadcast each state scalar to a
    # full (H, W) plane and stack onto the heightmap.
    batch, _, height, width = obs.shape
    state_dim = state.shape[1]
    state_plane = state.view(batch, state_dim, 1, 1).expand(
        batch, state_dim, height, width
    )
    return torch.cat([obs, state_plane], dim=1)
