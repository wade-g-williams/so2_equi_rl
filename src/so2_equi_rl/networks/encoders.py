"""Rotation-equivariant image encoder matching Wang et al. (2022). C_N group
via e2cnn R2Conv layers; default C_8 matches the paper.
"""

import torch
from e2cnn import gspaces, nn as enn

# The encoder's block stack reduces spatial 128 -> 1 exactly. Other
# sizes would produce >1x1 output and silently break the downstream
# .view(B, -1) flatten in actor/critic heads.
EXPECTED_OBS_SIZE = 128


def irrep1_multiplicity(group_order: int) -> int:
    # irrep(1) is 2D for N >= 3 (one copy covers (dx, dy)); for N = 2 it's the 1D sign rep, so need two.
    return 2 if group_order == 2 else 1


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
                f"got n_hidden={n_hidden}"
            )

        # Stored so downstream heads can read them without re-receiving
        # the same kwargs on the caller side.
        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.group_order = group_order

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
        # Flat channel count after the final 1x1 spatial collapse: last
        # multiplicity * group_order. Exposed so downstream heads / tests
        # can size flatten layers without reaching into FieldType internals.
        self.output_dim = mults[-1] * group_order

    def forward(self, obs: torch.Tensor) -> enn.GeometricTensor:
        # The conv stack is hand-tuned for 128x128 input; other sizes
        # produce non-1x1 output and break downstream flatten logic.
        if obs.shape[-2:] != (EXPECTED_OBS_SIZE, EXPECTED_OBS_SIZE):
            raise ValueError(
                f"EquiEncoder expects (B, C, {EXPECTED_OBS_SIZE}, {EXPECTED_OBS_SIZE}) "
                f"input; got spatial {tuple(obs.shape[-2:])}"
            )
        x = enn.GeometricTensor(obs, self.input_type)
        return self.conv(x)


class CNNEncoder(torch.nn.Module):
    """Plain-CNN image encoder baseline. Stub surface so heads can inject
    it and the create_agent factory can resolve it; body lands later.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 128,
        group_order: int = 1,
    ) -> None:
        super().__init__()
        # Mirrors EquiEncoder's surface so heads can read the same attrs
        # without branching on encoder type.
        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.group_order = group_order
        # Matches EquiEncoder's n_hidden * group_order so shared heads
        # can size themselves off self.encoder.output_dim either way.
        self.output_dim = n_hidden * group_order

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
