"""C_N-equivariant image encoder.

Matches Wang et al. (2022) EquivariantEncoder128. Seven-block e2cnn stack,
regular-representation features throughout, 128x128 input -> 1x1 output.

Paper notation -> code:
    N       -> group_order    (default 8, matches paper main results)
    n_out   -> n_hidden       (default 128)

Public API:
    tile_state(obs, state) -> (B, obs_channels, H, W)    # state_dim=1 only
    EquiEncoder(obs_channels=2, n_hidden=128, group_order=8)
    encoder(tiled_obs) -> GeometricTensor of shape (B, n_hidden * N, 1, 1)
    encoder.output_type   # e2cnn FieldType, for chaining into actor/critic
    encoder.output_dim    # int, for sizing downstream heads
"""

import torch

from e2cnn import gspaces
from e2cnn import nn as enn


def tile_state(obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    # Broadcast the scalar gripper flag to a full channel plane so it
    # rides the conv stack alongside the heightmap.
    batch, _, height, width = obs.shape
    state_plane = state.view(batch, 1, 1, 1).expand(batch, 1, height, width)
    return torch.cat([obs, state_plane], dim=1)


class EquiEncoder(torch.nn.Module):
    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 128,
        group_order: int = 8,
    ) -> None:
        super().__init__()

        # Spatial progression (channel mults are per-field; each field has
        # N values from the regular representation):
        #
        #             conv         spatial    mult
        #   input                  128x128    obs_channels (trivial repr)
        #   block 1   3x3 + pool   64x64      n_hidden // 8
        #   block 2   3x3 + pool   32x32      n_hidden // 4
        #   block 3   3x3 + pool   16x16      n_hidden // 2
        #   block 4   3x3 + pool    8x8       n_hidden
        #   block 5   3x3           8x8       n_hidden * 2    <- bottleneck
        #   block 6   3x3 valid + pool  3x3   n_hidden        <- valid: 8->6
        #   block 7   3x3 valid     1x1       n_hidden
        #   output                 (B, n_hidden * N, 1, 1)

        self._gspace = gspaces.Rot2dOnR2(N=group_order)

        # Both input channels are scalar, so they use trivial reprs.
        self.input_type = enn.FieldType(
            self._gspace, [self._gspace.trivial_repr] * obs_channels
        )

        mults = (
            n_hidden // 8,
            n_hidden // 4,
            n_hidden // 2,
            n_hidden,
            n_hidden * 2,
            n_hidden,
            n_hidden,
        )
        regular_types = [
            enn.FieldType(self._gspace, m * [self._gspace.regular_repr]) for m in mults
        ]

        self.conv = enn.SequentialModule(
            # block 1
            enn.R2Conv(self.input_type, regular_types[0], kernel_size=3, padding=1),
            enn.ReLU(regular_types[0], inplace=True),
            enn.PointwiseMaxPool(regular_types[0], kernel_size=2, stride=2),
            # block 2
            enn.R2Conv(regular_types[0], regular_types[1], kernel_size=3, padding=1),
            enn.ReLU(regular_types[1], inplace=True),
            enn.PointwiseMaxPool(regular_types[1], kernel_size=2, stride=2),
            # block 3
            enn.R2Conv(regular_types[1], regular_types[2], kernel_size=3, padding=1),
            enn.ReLU(regular_types[2], inplace=True),
            enn.PointwiseMaxPool(regular_types[2], kernel_size=2, stride=2),
            # block 4
            enn.R2Conv(regular_types[2], regular_types[3], kernel_size=3, padding=1),
            enn.ReLU(regular_types[3], inplace=True),
            enn.PointwiseMaxPool(regular_types[3], kernel_size=2, stride=2),
            # block 5 (bottleneck, no pool)
            enn.R2Conv(regular_types[3], regular_types[4], kernel_size=3, padding=1),
            enn.ReLU(regular_types[4], inplace=True),
            # block 6 (valid conv 8->6, then pool 6->3)
            enn.R2Conv(regular_types[4], regular_types[5], kernel_size=3, padding=0),
            enn.ReLU(regular_types[5], inplace=True),
            enn.PointwiseMaxPool(regular_types[5], kernel_size=2, stride=2),
            # block 7 (valid conv 3->1)
            enn.R2Conv(regular_types[5], regular_types[6], kernel_size=3, padding=0),
            enn.ReLU(regular_types[6], inplace=True),
        )

        self.output_type = regular_types[6]
        self.output_dim = n_hidden * group_order

    def forward(self, obs: torch.Tensor) -> enn.GeometricTensor:
        x = enn.GeometricTensor(obs, self.input_type)
        return self.conv(x)
