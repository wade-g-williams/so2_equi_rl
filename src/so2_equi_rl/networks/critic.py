"""Twin-Q critics for the 5-D pxyzr SAC agent. EquiCritic (C_N-invariant
under the joint obs/action group action) and CNNCritic (plain-CNN baseline
stub). Unlike the actors, critics do not share a base class: they have no
common sampling logic, just a forward(obs, action) -> (q1, q2) signature.
"""

from typing import Tuple
import torch
from e2cnn import nn as enn
from so2_equi_rl.networks.equi_encoder import EquiEncoder, irrep1_multiplicity


class EquiCritic(torch.nn.Module):
    """C_N-invariant twin-Q critic. Rotating obs by g and the (dx, dy)
    components of action by the same g leaves q1 and q2 unchanged.
    (p, dz, dtheta) are scalars under 2D rotation and pass through.

    Two independent encoders + two independent 1x1 mix heads give SAC its
    min(q1, q2) target without forcing Q1 and Q2 to share a backbone.
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
                "EquiCritic hardcodes the pxyzr action layout (action_dim=5); "
                "got action_dim={}".format(action_dim)
            )
        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.group_order = group_order

        self.n_rho1 = irrep1_multiplicity(group_order)

        # Two independent encoders so Q1 and Q2 cannot collapse to the same
        # function during training. Standard SAC twin-Q trick.
        self.encoder_1 = EquiEncoder(
            obs_channels=obs_channels, n_hidden=n_hidden, group_order=group_order
        )
        self.encoder_2 = EquiEncoder(
            obs_channels=obs_channels, n_hidden=n_hidden, group_order=group_order
        )
        gspace = self.encoder_1.gspace

        # Action field type: 1 irrep(1) for (dx, dy) + 3 trivials for
        # (p, dz, dtheta). Same layout as the actor's head output but read
        # the other way: here the action is an INPUT to the critic.
        self.action_type = enn.FieldType(
            gspace,
            self.n_rho1 * [gspace.irrep(1)] + 3 * [gspace.trivial_repr],
        )

        # Projection target: same shape as the action so the subsequent
        # concat produces a symmetric (enc-projected, action) tensor.
        self.mid_type = self.action_type

        # After concat we see mid_type + action_type fields side-by-side.
        # Summing the representation lists is how e2cnn builds a concatenated
        # FieldType: 2 irrep(1) copies + 6 trivials = 10 scalars.
        self.merged_type = enn.FieldType(
            gspace,
            list(self.mid_type.representations)
            + list(self.action_type.representations),
        )

        # One trivial scalar out per Q-head (invariant Q-value).
        self.q_out_type = enn.FieldType(gspace, [gspace.trivial_repr])

        # Two independent projection layers (encoder -> mid shape).
        self.proj_1 = enn.R2Conv(
            self.encoder_1.output_type, self.mid_type, kernel_size=1, padding=0
        )
        self.proj_2 = enn.R2Conv(
            self.encoder_2.output_type, self.mid_type, kernel_size=1, padding=0
        )

        # Two independent mixers (merged -> single trivial scalar).
        self.mix_1 = enn.R2Conv(
            self.merged_type, self.q_out_type, kernel_size=1, padding=0
        )
        self.mix_2 = enn.R2Conv(
            self.merged_type, self.q_out_type, kernel_size=1, padding=0
        )

    def _wrap_action(self, action: torch.Tensor) -> enn.GeometricTensor:
        # Reshuffle pxyzr -> (dx, dy, p, dz, dtheta) so the irrep(1)
        # components sit at the front of the channel axis where the
        # action_type expects them.
        p = action[:, 0:1]
        dxy = action[:, 1:3]
        dz_dtheta = action[:, 3:5]
        reshuffled = torch.cat([dxy, p, dz_dtheta], dim=1)  # (B, 5)

        # 1x1 spatial so it lines up with the encoder's 1x1 output.
        spatial = reshuffled.view(action.shape[0], 5, 1, 1)
        return enn.GeometricTensor(spatial, self.action_type)

    def _twin_forward(
        self,
        encoder: EquiEncoder,
        proj: enn.R2Conv,
        mix: enn.R2Conv,
        obs: torch.Tensor,
        action_gt: enn.GeometricTensor,
    ) -> torch.Tensor:
        # Encode, project to action-shape, concat with the wrapped action,
        # rewrap under the merged FieldType, mix to a single scalar.
        enc_out = encoder(obs)
        projected = proj(enc_out)
        merged_tensor = torch.cat([projected.tensor, action_gt.tensor], dim=1)
        merged_gt = enn.GeometricTensor(merged_tensor, self.merged_type)
        q = mix(merged_gt).tensor.view(obs.shape[0], 1)
        return q

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Single action wrap is shared across both twins, since the action
        # GeometricTensor is read-only downstream.
        action_gt = self._wrap_action(action)
        q1 = self._twin_forward(self.encoder_1, self.proj_1, self.mix_1, obs, action_gt)
        q2 = self._twin_forward(self.encoder_2, self.proj_2, self.mix_2, obs, action_gt)
        return q1, q2


class CNNCritic(torch.nn.Module):
    """Plain-CNN twin-Q baseline for comparison against EquiCritic. WIP."""

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
