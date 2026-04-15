"""Neural nets for the SO(2)-equivariant agent: image encoder, SAC actors,
and twin-Q critics. Everything equivariant is built on e2cnn over C_N.
"""

from so2_equi_rl.networks.actor import CNNActor, EquiActor, SACGaussianPolicyBase
from so2_equi_rl.networks.critic import CNNCritic, EquiCritic
from so2_equi_rl.networks.equi_encoder import EquiEncoder, tile_state

__all__ = [
    "EquiEncoder",
    "tile_state",
    "SACGaussianPolicyBase",
    "EquiActor",
    "CNNActor",
    "EquiCritic",
    "CNNCritic",
]
