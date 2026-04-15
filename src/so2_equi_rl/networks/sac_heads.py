"""SAC actor + critic heads for the 5-D pxyzr action.

EquiActor shares the tanh-squashed reparameterized Gaussian sampler in
SACGaussianPolicyBase. EquiCritic is the C_N-invariant twin-Q critic.

Action-split contract on the equivariant variants:
- (dx, dy) transform as irrep(1) under the C_N group action on the
  heightmap, so they rotate with the obs.
- (p, dz, dtheta) transform trivially and pass through invariant.
- Critics are C_N-invariant under the joint (obs, action) group action.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from e2cnn import nn as enn

from so2_equi_rl.networks.encoders import EquiEncoder, irrep1_multiplicity

# Bounds we clamp log_std to before sampling.
LOG_SIG_MIN = -20.0
LOG_SIG_MAX = 2.0


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

        # Correct the log-prob for the tanh squash (change of variables).
        # Stable form of log(1 - tanh(x)^2): 2 * (log 2 - x - softplus(-2x)).
        # Avoids precision loss near |tanh(x)| ~ 1.
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - 2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # tanh(mean) is the deterministic action we use at eval time.
        mean_tanh = torch.tanh(mean)

        return y_t, log_prob, mean_tanh


class EquiActor(SACGaussianPolicyBase):
    """C_N-equivariant SAC policy. (dx, dy) go through irrep(1) so they rotate
    with the input; (p, dz, dtheta) means and all 5 log_stds are trivial_repr
    and stay fixed.

    Takes an already-constructed `EquiEncoder` as a dependency. The caller
    (typically `SACAgent`) owns encoder lifetime so the critic cannot accidentally
    share weights with the actor.
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

        # Final 1x1 equivariant head. n_rho1 irrep(1) fields cover (dx, dy);
        # 8 trivial fields cover 3 invariant means (p, dz, dtheta) plus
        # all 5 log_stds. 10 scalars total in both N>=3 and N=2 cases.
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

        # Reassemble into pxyzr order: (p, dx, dy, dz, dtheta).
        mean = torch.cat([p, dxy, dz_dtheta], dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std


class EquiCritic(torch.nn.Module):
    """C_N-invariant twin-Q critic. Rotating obs by g and the (dx, dy)
    components of action by the same g leaves q1 and q2 unchanged.
    (p, dz, dtheta) are scalars under 2D rotation and pass through.

    Takes a single `EquiEncoder` shared between the two Q heads (matches
    Wang et al.'s paper repo). The twin Qs stay decorrelated via independent
    projection + mix layers on top of the shared backbone.
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

        # Action field type: 1 irrep(1) for (dx, dy) + 3 trivials for
        # (p, dz, dtheta). Mirrors the actor head's output layout, read
        # the other way: here the action is an INPUT to the critic.
        self.action_type = enn.FieldType(
            gspace,
            n_rho1 * [gspace.irrep(1)] + 3 * [gspace.trivial_repr],
        )

        # After concat we see (projected, action) fields side-by-side.
        # Summing representation lists is how e2cnn builds a concat FieldType.
        self.merged_type = enn.FieldType(
            gspace,
            list(self.action_type.representations)
            + list(self.action_type.representations),
        )

        # One trivial scalar out per Q-head (invariant Q-value).
        self.q_out_type = enn.FieldType(gspace, [gspace.trivial_repr])

        # Two independent projection layers from encoder output to action shape.
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
        # Reorder from pxyzr into (dx, dy, p, dz, dtheta) so the irrep(1)
        # components sit at the front of the channel axis where the
        # action_type expects them.
        p = action[:, 0:1]
        dxy = action[:, 1:3]
        dz_dtheta = action[:, 3:5]
        reshuffled = torch.cat([dxy, p, dz_dtheta], dim=1)  # (B, 5)

        # 1x1 spatial so it lines up with the encoder's 1x1 output.
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
        # Project shared encoded features to action-shape, concat with the
        # wrapped action, rewrap under merged FieldType, mix to one scalar.
        projected = proj(enc_out)
        merged_tensor = torch.cat([projected.tensor, action_gt.tensor], dim=1)
        merged_gt = enn.GeometricTensor(merged_tensor, self.merged_type)
        return mix(merged_gt).tensor.view(batch, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # One shared encoder forward drives both twin Q heads.
        enc_out = self.encoder(obs)
        action_gt = self._wrap_action(action)
        batch = obs.shape[0]
        q1 = self._twin_head(self.proj_1, self.mix_1, enc_out, action_gt, batch)
        q2 = self._twin_head(self.proj_2, self.mix_2, enc_out, action_gt, batch)
        return q1, q2
