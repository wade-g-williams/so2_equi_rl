"""Image encoders. Equivariant C_N stack from Wang et al. (2022) and a
plain CNN baseline. Default C_8 matches the paper.
"""

import torch
from e2cnn import gspaces, nn as enn

# The block stack reduces 128 spatial down to 1 exactly. Other sizes
# silently break the .view(B, -1) flatten in the actor and critic heads.
EXPECTED_OBS_SIZE = 128


def irrep1_multiplicity(group_order: int) -> int:
    # irrep(1) is 2D for N >= 3 (one copy covers (dx, dy)). For N = 2 it's
    # the 1D sign rep, so we need two copies.
    return 2 if group_order == 2 else 1


class EquiEncoder(torch.nn.Module):
    """C_N-equivariant conv encoder. Seven R2Conv blocks with regular_repr
    hidden fields. Takes (B, obs_channels, 128, 128), returns a
    GeometricTensor with n_hidden * N channels at 1x1 spatial. Rotating
    the input by (360 / N) degrees cyclically permutes the feature map.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 128,
        group_order: int = 8,
    ) -> None:
        super().__init__()

        # Block 1 uses n_hidden // 8 channels, so n_hidden must divide 8.
        if n_hidden % 8 != 0:
            raise ValueError(
                "n_hidden must be divisible by 8 (block 1 uses n_hidden // 8 channels); "
                f"got n_hidden={n_hidden}"
            )

        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.group_order = group_order

        self.gspace = gspaces.Rot2dOnR2(N=group_order)

        # Both input channels are scalars (don't change under rotation).
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
        # Flat channel count after the final 1x1 collapse. Exposed so heads
        # and tests don't have to reach into FieldType internals.
        self.output_dim = mults[-1] * group_order

    def forward(self, obs: torch.Tensor) -> enn.GeometricTensor:
        if obs.shape[-2:] != (EXPECTED_OBS_SIZE, EXPECTED_OBS_SIZE):
            raise ValueError(
                f"EquiEncoder expects (B, C, {EXPECTED_OBS_SIZE}, {EXPECTED_OBS_SIZE}) "
                f"input; got spatial {tuple(obs.shape[-2:])}"
            )
        x = enn.GeometricTensor(obs, self.input_type)
        return self.conv(x)


class CNNEncoder(torch.nn.Module):
    """Plain-CNN baseline. Channel schedule (n_hidden // 8, // 4, // 2, 1x,
    2x, 2x, 1x) lines up with EquiEncoder so the CNN SAC run is comparable.
    Takes (B, obs_channels, 128, 128), returns (B, n_hidden, 1, 1).
    group_order is accepted for kwarg parity with EquiEncoder and ignored.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 128,
        group_order: int = 1,
    ) -> None:
        super().__init__()

        if n_hidden % 8 != 0:
            raise ValueError(
                "n_hidden must be divisible by 8 (block 1 uses n_hidden // 8 channels); "
                f"got n_hidden={n_hidden}"
            )

        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.group_order = group_order

        # Channel schedule. Block 6 is 2*n_hidden (not 1*n_hidden) to match the reference repo.
        c = (
            n_hidden // 8,
            n_hidden // 4,
            n_hidden // 2,
            n_hidden,
            n_hidden * 2,
            n_hidden * 2,
            n_hidden,
        )

        self.conv = torch.nn.Sequential(
            # block 1: 128 -> 64
            torch.nn.Conv2d(obs_channels, c[0], kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2: 64 -> 32
            torch.nn.Conv2d(c[0], c[1], kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # block 3: 32 -> 16
            torch.nn.Conv2d(c[1], c[2], kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # block 4: 16 -> 8
            torch.nn.Conv2d(c[2], c[3], kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # block 5: 8 -> 8 (bottleneck, no pool)
            torch.nn.Conv2d(c[3], c[4], kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # block 6: valid conv 8 -> 6, then pool 6 -> 3
            torch.nn.Conv2d(c[4], c[5], kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # block 7: valid conv 3 -> 1
            torch.nn.Conv2d(c[5], c[6], kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
        )

        self.output_dim = n_hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2:] != (EXPECTED_OBS_SIZE, EXPECTED_OBS_SIZE):
            raise ValueError(
                f"CNNEncoder expects (B, C, {EXPECTED_OBS_SIZE}, {EXPECTED_OBS_SIZE}) "
                f"input; got spatial {tuple(obs.shape[-2:])}"
            )
        return self.conv(obs)
