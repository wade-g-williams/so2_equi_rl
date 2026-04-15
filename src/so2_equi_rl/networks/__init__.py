"""Neural nets for the SO(2)-equivariant agent: image encoder, SAC actor,
and twin-Q critic. Built on e2cnn over C_N.
"""

from so2_equi_rl.networks.sac_heads import (
    CNNActor,
    CNNCritic,
    EquiActor,
    EquiCritic,
    SACGaussianPolicyBase,
)
from so2_equi_rl.networks.encoders import CNNEncoder, EquiEncoder

__all__ = [
    "EquiEncoder",
    "CNNEncoder",
    "SACGaussianPolicyBase",
    "EquiActor",
    "EquiCritic",
    "CNNActor",
    "CNNCritic",
]
