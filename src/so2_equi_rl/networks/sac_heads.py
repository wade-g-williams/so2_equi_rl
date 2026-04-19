"""SAC actor and critic heads for the 5-D pxyzr action.

Equi variants: (dx, dy) transform as irrep(1) under the C_N action on the
heightmap and rotate with the obs; (p, dz, dtheta) are trivial. Critics
are C_N-invariant under the joint (obs, action) action.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from e2cnn import nn as enn

from so2_equi_rl.networks.encoders import CNNEncoder, EquiEncoder, irrep1_multiplicity

LOG_SIG_MIN = -20.0
LOG_SIG_MAX = 2.0


class SACGaussianPolicyBase(torch.nn.Module):
    """Shared tanh-squashed reparameterized Gaussian sampler. Subclasses implement forward()."""

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # rsample so gradients flow through the sample.
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Tanh change-of-variables on log-prob, in the stable form
        # 2 * (log 2 - x - softplus(-2x)) to avoid precision loss near |tanh(x)| ~ 1.
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - 2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # tanh(mean) is the deterministic eval-time action.
        mean_tanh = torch.tanh(mean)

        return y_t, log_prob, mean_tanh


class EquiActor(SACGaussianPolicyBase):
    """C_N-equivariant SAC policy. (dx, dy) means go through irrep(1);
    (p, dz, dtheta) means and all 5 log_stds are trivial. Caller owns
    encoder lifetime so actor and critic can't share weights.
    """

    def __init__(
        self,
        encoder: EquiEncoder,
        action_dim: int = 5,
    ) -> None:
        super().__init__()

        if action_dim != 5:
            raise ValueError(
                "EquiActor hardcodes the pxyzr action layout (action_dim=5); "
                f"got action_dim={action_dim}"
            )

        self.encoder = encoder
        self.action_dim = action_dim

        n_rho1 = irrep1_multiplicity(encoder.group_order)
        gspace = encoder.gspace

        # Final 1x1 head. n_rho1 irrep(1) fields cover (dx, dy); 8 trivial
        # fields cover 3 invariant means (p, dz, dtheta) and all 5 log_stds.
        head_out_type = enn.FieldType(
            gspace,
            n_rho1 * [gspace.irrep(1)] + 8 * [gspace.trivial_repr],
        )

        self.head = enn.R2Conv(
            encoder.output_type,
            head_out_type,
            kernel_size=1,
            padding=0,
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        head_out = self.head(encoded)
        flat = head_out.tensor.view(obs.shape[0], -1)

        # Layout: 2 irrep(1) scalars, 3 trivial means (p, dz, dtheta), 5 log_stds.
        dxy = flat[:, 0:2]
        p = flat[:, 2:3]
        dz_dtheta = flat[:, 3:5]
        log_std = flat[:, 5:10]

        # Reassemble into pxyzr order.
        mean = torch.cat([p, dxy, dz_dtheta], dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std


class EquiCritic(torch.nn.Module):
    """C_N-invariant twin-Q critic. One shared EquiEncoder drives both Q
    heads with independent proj and mix layers on top.
    """

    def __init__(
        self,
        encoder: EquiEncoder,
        action_dim: int = 5,
    ) -> None:
        super().__init__()

        if action_dim != 5:
            raise ValueError(
                "EquiCritic hardcodes the pxyzr action layout (action_dim=5); "
                f"got action_dim={action_dim}"
            )

        self.encoder = encoder
        self.action_dim = action_dim

        n_rho1 = irrep1_multiplicity(encoder.group_order)
        gspace = encoder.gspace

        # Action field type. Mirrors the actor head's layout, read the
        # other way since action is an INPUT to the critic.
        self.action_type = enn.FieldType(
            gspace,
            n_rho1 * [gspace.irrep(1)] + 3 * [gspace.trivial_repr],
        )

        # Concat FieldType: summing rep lists is how e2cnn builds it.
        self.merged_type = enn.FieldType(
            gspace,
            list(self.action_type.representations)
            + list(self.action_type.representations),
        )

        self.q_out_type = enn.FieldType(gspace, [gspace.trivial_repr])

        # Two independent projections from encoder output to action shape.
        self.proj_1 = enn.R2Conv(
            encoder.output_type, self.action_type, kernel_size=1, padding=0
        )
        self.proj_2 = enn.R2Conv(
            encoder.output_type, self.action_type, kernel_size=1, padding=0
        )

        # Two independent mixers from the merged feature to a single trivial scalar.
        self.mix_1 = enn.R2Conv(
            self.merged_type, self.q_out_type, kernel_size=1, padding=0
        )
        self.mix_2 = enn.R2Conv(
            self.merged_type, self.q_out_type, kernel_size=1, padding=0
        )

    def _wrap_action(self, action: torch.Tensor) -> enn.GeometricTensor:
        # Reorder pxyzr -> (dx, dy, p, dz, dtheta) so irrep(1) sits at the
        # front of the channel axis where action_type expects them.
        p = action[:, 0:1]
        dxy = action[:, 1:3]
        dz_dtheta = action[:, 3:5]
        reshuffled = torch.cat([dxy, p, dz_dtheta], dim=1)

        # 1x1 spatial to line up with the encoder's 1x1 output.
        spatial = reshuffled.view(action.shape[0], 5, 1, 1)
        return enn.GeometricTensor(spatial, self.action_type)

    def _twin_head(
        self,
        proj: enn.R2Conv,
        mix: enn.R2Conv,
        enc_out: enn.GeometricTensor,
        action_gt: enn.GeometricTensor,
        batch: int,
    ) -> torch.Tensor:
        projected = proj(enc_out)
        merged_tensor = torch.cat([projected.tensor, action_gt.tensor], dim=1)
        merged_gt = enn.GeometricTensor(merged_tensor, self.merged_type)
        return mix(merged_gt).tensor.view(batch, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out = self.encoder(obs)
        action_gt = self._wrap_action(action)
        batch = obs.shape[0]
        q1 = self._twin_head(self.proj_1, self.mix_1, enc_out, action_gt, batch)
        q2 = self._twin_head(self.proj_2, self.mix_2, enc_out, action_gt, batch)
        return q1, q2


class CNNActor(SACGaussianPolicyBase):
    """Plain-CNN SAC policy. Two parallel Linear heads off the flattened
    encoder output (no hidden MLP), matching the paper's SACGaussianPolicy.

    detach_encoder is for FERM-SAC where the encoder is shared with the
    critic and trains from TD + InfoNCE only. Default False leaves
    vanilla, DrQ, and RAD untouched.
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        action_dim: int = 5,
        detach_encoder: bool = False,
    ) -> None:
        super().__init__()

        if action_dim != 5:
            raise ValueError(
                "CNNActor hardcodes the pxyzr action layout (action_dim=5); "
                f"got action_dim={action_dim}"
            )

        self.encoder = encoder
        self.action_dim = action_dim
        self.detach_encoder = detach_encoder

        self.mean_linear = torch.nn.Linear(encoder.output_dim, action_dim)
        self.log_std_linear = torch.nn.Linear(encoder.output_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        # FERM: stop actor-loss gradient at the encoder boundary.
        if self.detach_encoder:
            encoded = encoded.detach()
        flat = encoded.view(obs.shape[0], -1)

        mean = self.mean_linear(flat)
        log_std = self.log_std_linear(flat)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std


class CNNCritic(torch.nn.Module):
    """Plain-CNN twin-Q critic. One shared CNNEncoder drives both heads
    with independent single-hidden-layer MLPs taking (encoder_flat, action).
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        action_dim: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if action_dim != 5:
            raise ValueError(
                "CNNCritic hardcodes the pxyzr action layout (action_dim=5); "
                f"got action_dim={action_dim}"
            )

        self.encoder = encoder
        self.action_dim = action_dim

        # Two independent MLPs decorrelate the twin Qs without duplicating conv weights.
        mlp_in = encoder.output_dim + action_dim
        self.q1_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.q2_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        detach_encoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # detach_encoder is a per-call flag for FERM's actor step (stops
        # actor-loss gradient on the Q-path from reaching the shared encoder).
        encoded = self.encoder(obs)
        if detach_encoder:
            encoded = encoded.detach()
        flat = encoded.view(obs.shape[0], -1)
        merged = torch.cat([flat, action], dim=1)
        q1 = self.q1_mlp(merged)
        q2 = self.q2_mlp(merged)
        return q1, q2
