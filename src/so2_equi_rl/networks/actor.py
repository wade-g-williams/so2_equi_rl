"""SAC policy networks for the 5-D pxyzr action. EquiActor (equivariant),
CNNActor (plain-CNN baseline), and SACGaussianPolicyBase (shared sampler).
"""

from typing import Tuple
import torch
from torch.distributions import Normal
from e2cnn import nn as enn
from so2_equi_rl.networks.equi_encoder import EquiEncoder, irrep1_multiplicity

# Bounds we clamp log_std to before sampling, plus a small epsilon
# that keeps the tanh log-prob correction numerically stable.
LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPSILON = 1e-6


class SACGaussianPolicyBase(torch.nn.Module):
    """Shared tanh-squashed reparameterized Gaussian sampler. Subclasses implement forward()."""

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reparameterized Gaussian sample, tanh-squashed to [-1, 1].
        # rsample() keeps gradients flowing through the sample.
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Correct the log-prob for the tanh squash (change of variables),
        # then sum over action dims so the output is (B, 1).
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # tanh(mean) is the deterministic action we use at eval time.
        mean_tanh = torch.tanh(mean)

        return y_t, log_prob, mean_tanh


class EquiActor(SACGaussianPolicyBase):
    """C_N-equivariant SAC policy. (dx, dy) go through irrep(1) so they rotate
    with the input; (p, dz, dtheta) means and all 5 log_stds are trivial_repr
    and stay fixed. One encoder feeds both halves without breaking symmetry.
    """

    def __init__(
        self,
        obs_channels: int = 2,
        action_dim: int = 5,
        n_hidden: int = 128,
        group_order: int = 8,
    ) -> None:
        super().__init__()

        if action_dim != 5:
            raise ValueError(
                "EquiActor hardcodes the pxyzr action layout (action_dim=5); "
                "got action_dim={}".format(action_dim)
            )

        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.group_order = group_order

        self.n_rho1 = irrep1_multiplicity(group_order)

        # The actor keeps its own encoder. We don't share with the critic.
        self.encoder = EquiEncoder(
            obs_channels=obs_channels,
            n_hidden=n_hidden,
            group_order=group_order,
        )
        gspace = self.encoder.gspace

        # Final 1x1 equivariant head. One irrep(1) field (2 scalars) for the
        # (dx, dy) mean, plus 8 trivial fields for the invariants:
        #   3 trivial -> (p, dz, dtheta) means
        #   5 trivial -> log_stds for all 5 actions
        # 9 fields total, producing 10 scalar outputs.
        n_trivial = 8
        head_out_type = enn.FieldType(
            gspace,
            self.n_rho1 * [gspace.irrep(1)] + n_trivial * [gspace.trivial_repr],
        )

        self.head = enn.R2Conv(
            self.encoder.output_type,
            head_out_type,
            kernel_size=1,
            padding=0,
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder, then 1x1 equivariant head, then flatten to (B, 10).
        encoded = self.encoder(obs)
        head_out = self.head(encoded)
        flat = head_out.tensor.view(obs.shape[0], -1)

        # Head layout: 2 irrep(1) scalars, then 3 trivials for the invariant
        # means (p, dz, dtheta), then 5 trivials for all log_stds.
        dxy = flat[:, 0:2]
        p = flat[:, 2:3]
        dz_dtheta = flat[:, 3:5]
        log_std = flat[:, 5:10]

        # Reassemble into the pxyzr order: (p, dx, dy, dz, dtheta).
        mean = torch.cat([p, dxy, dz_dtheta], dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std


class CNNActor(SACGaussianPolicyBase):
    """Plain-CNN SAC policy baseline for comparison against EquiActor. WIP."""

    def __init__(
        self,
        obs_channels: int = 2,
        action_dim: int = 5,
        n_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.n_hidden = n_hidden

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("CNNActor body not yet implemented")
