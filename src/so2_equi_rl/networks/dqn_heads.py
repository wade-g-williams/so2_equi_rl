"""DQN Q-networks for the 4-axis discrete action grid.

Both nets emit Q for every (xy_id, z_id, theta_id, p_id) cell. The
equivariant variant (paper Fig 12a) puts the rotation-equivariant
dxy axis on the spatial 3x3 grid and the invariant (dz, dtheta, p)
axes on the channel dim with trivial reps. The CNN baseline (paper
Fig 13a) flattens the conv stack and emits 162 logits via FC. Both
return the same (B, n_xy, n_z, n_theta, n_p) shape so DQNAgent stays
net-agnostic.
"""

import torch
import torch.nn as nn
from e2cnn import gspaces, nn as enn

from so2_equi_rl.networks.encoders import EXPECTED_OBS_SIZE


class EquiDQNNet(nn.Module):
    """C_N-equivariant Q-net. Seven R2Conv blocks reduce 128 spatial down
    to 3 (the dxy grid). Output channels = n_p * n_z * n_theta = n_inv,
    all trivial reps so the channel axis is rotation-invariant. A 90/N deg
    rotation of the input cyclically permutes the spatial 3x3 grid.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 64,
        group_order: int = 4,
        n_p: int = 2,
        n_xy: int = 9,
        n_z: int = 3,
        n_theta: int = 3,
    ) -> None:
        super().__init__()

        if n_hidden % 4 != 0:
            raise ValueError(
                "n_hidden must be divisible by 4 (block 1 uses n_hidden // 4 channels); "
                f"got n_hidden={n_hidden}"
            )

        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.group_order = group_order
        self.n_p = n_p
        self.n_xy = n_xy
        self.n_z = n_z
        self.n_theta = n_theta
        self.n_inv = n_p * n_z * n_theta

        self.gspace = gspaces.Rot2dOnR2(N=group_order)

        # Heightmap + tiled gripper-state are scalars under rotation.
        self.input_type = enn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * obs_channels
        )

        # Channel mults for blocks 1-6. Hidden fields use regular_repr so
        # all rotated copies live together inside one channel block.
        mults = (
            n_hidden // 4,
            n_hidden // 2,
            n_hidden,
            n_hidden * 2,
            n_hidden * 4,
            n_hidden * 4,
        )
        regular_types = [
            enn.FieldType(self.gspace, m * [self.gspace.regular_repr]) for m in mults
        ]
        # Final layer projects to n_inv trivial fields. Q-values are
        # invariant under joint rotation when (xy_id) is fixed.
        self.output_type = enn.FieldType(
            self.gspace, self.n_inv * [self.gspace.trivial_repr]
        )

        self.conv = enn.SequentialModule(
            # block 1: 128 -> 64
            enn.R2Conv(self.input_type, regular_types[0], kernel_size=3, padding=1),
            enn.ReLU(regular_types[0], inplace=True),
            enn.PointwiseMaxPool(regular_types[0], kernel_size=2, stride=2),
            # block 2: 64 -> 32
            enn.R2Conv(regular_types[0], regular_types[1], kernel_size=3, padding=1),
            enn.ReLU(regular_types[1], inplace=True),
            enn.PointwiseMaxPool(regular_types[1], kernel_size=2, stride=2),
            # block 3: 32 -> 16
            enn.R2Conv(regular_types[1], regular_types[2], kernel_size=3, padding=1),
            enn.ReLU(regular_types[2], inplace=True),
            enn.PointwiseMaxPool(regular_types[2], kernel_size=2, stride=2),
            # block 4: 16 -> 8
            enn.R2Conv(regular_types[2], regular_types[3], kernel_size=3, padding=1),
            enn.ReLU(regular_types[3], inplace=True),
            enn.PointwiseMaxPool(regular_types[3], kernel_size=2, stride=2),
            # block 5: 8 -> 8 (bottleneck, no pool)
            enn.R2Conv(regular_types[3], regular_types[4], kernel_size=3, padding=1),
            enn.ReLU(regular_types[4], inplace=True),
            # block 6: valid 8 -> 6, then pool 6 -> 3
            enn.R2Conv(regular_types[4], regular_types[5], kernel_size=3, padding=0),
            enn.ReLU(regular_types[5], inplace=True),
            enn.PointwiseMaxPool(regular_types[5], kernel_size=2, stride=2),
            # block 7: 1x1 projection to n_inv trivial channels at 3x3 spatial
            enn.R2Conv(regular_types[5], self.output_type, kernel_size=1, padding=0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2:] != (EXPECTED_OBS_SIZE, EXPECTED_OBS_SIZE):
            raise ValueError(
                f"EquiDQNNet expects (B, C, {EXPECTED_OBS_SIZE}, {EXPECTED_OBS_SIZE}) "
                f"input; got spatial {tuple(obs.shape[-2:])}"
            )
        B = obs.shape[0]
        x = enn.GeometricTensor(obs, self.input_type)
        out = self.conv(x).tensor  # (B, n_inv, 3, 3)

        # Spatial (3, 3) -> n_xy=9 in row-major: row = dx_idx, col = dy_idx.
        # Channel n_inv -> (n_z, n_theta, n_p) in row-major so the agent
        # gathers q[..., z_id, theta_id, p_id] consistently.
        out = out.permute(0, 2, 3, 1).contiguous()  # (B, 3, 3, n_inv)
        return out.view(B, self.n_xy, self.n_z, self.n_theta, self.n_p)


class CNNDQNNet(nn.Module):
    """Plain-CNN DQN baseline. Channel schedule (n_hidden//2, n_hidden,
    2x, 4x, 8x, 8x, 2x) is parameter-matched to the equivariant net at
    n_hidden=64. Final FC emits all 162 logits, reshaped to match the
    equi net's (B, n_xy, n_z, n_theta, n_p) layout.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        n_hidden: int = 64,
        group_order: int = 1,  # ignored, kept for kwarg parity
        n_p: int = 2,
        n_xy: int = 9,
        n_z: int = 3,
        n_theta: int = 3,
    ) -> None:
        super().__init__()
        del group_order  # unused

        if n_hidden % 2 != 0:
            raise ValueError(
                "n_hidden must be divisible by 2 (block 1 uses n_hidden // 2 channels); "
                f"got n_hidden={n_hidden}"
            )

        self.obs_channels = obs_channels
        self.n_hidden = n_hidden
        self.n_p = n_p
        self.n_xy = n_xy
        self.n_z = n_z
        self.n_theta = n_theta
        self.n_inv = n_p * n_z * n_theta

        c = (
            n_hidden // 2,
            n_hidden,
            n_hidden * 2,
            n_hidden * 4,
            n_hidden * 8,
            n_hidden * 8,
            n_hidden * 2,
        )

        self.conv = nn.Sequential(
            # block 1: 128 -> 64
            nn.Conv2d(obs_channels, c[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2: 64 -> 32
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 3: 32 -> 16
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 4: 16 -> 8
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 5: 8 -> 8 (bottleneck, no pool)
            nn.Conv2d(c[3], c[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # block 6: valid 8 -> 6, then pool 6 -> 3
            nn.Conv2d(c[4], c[5], kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 7: 1x1 projection at 3x3 spatial
            nn.Conv2d(c[5], c[6], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Flatten (c[6], 3, 3) -> n_xy * n_inv. Single FC matches the paper's
        # "FC at end" baseline.
        self.fc = nn.Linear(c[6] * 3 * 3, self.n_xy * self.n_inv)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2:] != (EXPECTED_OBS_SIZE, EXPECTED_OBS_SIZE):
            raise ValueError(
                f"CNNDQNNet expects (B, C, {EXPECTED_OBS_SIZE}, {EXPECTED_OBS_SIZE}) "
                f"input; got spatial {tuple(obs.shape[-2:])}"
            )
        B = obs.shape[0]
        feat = self.conv(obs).view(B, -1)
        flat = self.fc(feat)
        # Same row-major channel order as the equi net: (n_xy, n_z, n_theta, n_p).
        return flat.view(B, self.n_xy, self.n_z, self.n_theta, self.n_p)
