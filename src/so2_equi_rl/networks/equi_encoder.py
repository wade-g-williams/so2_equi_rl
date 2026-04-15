"""Rotation-equivariant image encoder matching Wang et al. (2022). C_N group
via e2cnn R2Conv layers; default C_8 matches the paper.
"""

import torch
from e2cnn import gspaces, nn as enn


def tile_state(obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    # Broadcast the scalar gripper flag to a full (H, W) plane and stack
    # it onto the heightmap: (B,1,H,W) + (B,1) -> (B,2,H,W).
    batch, _, height, width = obs.shape
    state_plane = state.view(batch, 1, 1, 1).expand(batch, 1, height, width)
    return torch.cat([obs, state_plane], dim=1)


class EquiEncoder(torch.nn.Module):
    """C_N-equivariant conv encoder. 7 R2Conv blocks, regular_repr hidden,
    maps (B, obs_channels, 128, 128) -> GeometricTensor with n_hidden*N channels
    at 1x1 spatial. Rotating the input by 360/N deg cyclically permutes the
    feature map.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 128,
        group_order: int = 8,
    ) -> None:
        super().__init__()

        # Block 1 uses n_hidden // 8 channels, so n_hidden has to be
        # divisible by 8 or the first block silently truncates.
        if n_hidden % 8 != 0:
            raise ValueError(
                "n_hidden must be divisible by 8 (block 1 uses n_hidden // 8 channels); "
                "got n_hidden={}".format(n_hidden)
            )

        self.gspace = gspaces.Rot2dOnR2(N=group_order)

        # Both input channels are scalars that don't change under rotation
        self.input_type = enn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * obs_channels
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
            enn.FieldType(self.gspace, m * [self.gspace.regular_repr]) for m in mults
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

        self.output_type = regular_types[-1]
        self.output_dim = n_hidden * group_order

    def forward(self, obs: torch.Tensor) -> enn.GeometricTensor:
        x = enn.GeometricTensor(obs, self.input_type)
        return self.conv(x)
