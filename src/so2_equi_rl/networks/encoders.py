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
    """Plain-CNN image encoder baseline. Mirrors the paper's sac_net.py
    schedule of 7 conv blocks with channel widths (n_hidden // 8, // 4,
    // 2, 1x, 2x, 2x, 1x) so the CNN SAC run lines up with EquiEncoder.
    Takes (B, obs_channels, 128, 128) and returns (B, n_hidden, 1, 1).
    group_order is accepted for kwarg uniformity with EquiEncoder and
    ignored.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 128,
        group_order: int = 1,
    ) -> None:
        super().__init__()

        # Block 1 uses n_hidden // 8 channels; enforce divisibility so we
        # never silently truncate. Matches EquiEncoder's constraint.
        if n_hidden % 8 != 0:
            raise ValueError(
                "n_hidden must be divisible by 8 (block 1 uses n_hidden // 8 channels); "
                f"got n_hidden={n_hidden}"
            )

        # Exposed so heads can size themselves without re-receiving kwargs.
        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.group_order = group_order

        # Channel schedule matches paper's SACEncoder under n_hidden=128.
        # Block 6 is 2*n_hidden (not 1*n_hidden) to match reference repo.
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

        # Flat feature count after the 1x1 spatial collapse equals the
        # final block's channel count. group_order is accepted for kwarg
        # uniformity with EquiEncoder and ignored here (SACAgent forwards
        # cfg.group_order to every encoder_cls blindly).
        self.output_dim = n_hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Conv stack is hand-tuned for 128x128; other sizes produce a
        # non-1x1 output and break downstream flatten logic in heads.
        if obs.shape[-2:] != (EXPECTED_OBS_SIZE, EXPECTED_OBS_SIZE):
            raise ValueError(
                f"CNNEncoder expects (B, C, {EXPECTED_OBS_SIZE}, {EXPECTED_OBS_SIZE}) "
                f"input; got spatial {tuple(obs.shape[-2:])}"
            )
        return self.conv(obs)
