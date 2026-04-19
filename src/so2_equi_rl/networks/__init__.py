"""Nets for the SO(2)-equivariant agent. Image encoder, SAC actor, twin-Q critic, built on e2cnn over C_N."""

from so2_equi_rl.networks.sac_heads import (
    CNNActor,
    CNNCritic,
    EquiActor,
    EquiCritic,
    SACGaussianPolicyBase,
)
from so2_equi_rl.networks.dqn_heads import CNNDQNNet, EquiDQNNet
from so2_equi_rl.networks.encoders import CNNEncoder, EquiEncoder

__all__ = [
    "EquiEncoder",
    "CNNEncoder",
    "SACGaussianPolicyBase",
    "EquiActor",
    "EquiCritic",
    "CNNActor",
    "CNNCritic",
    "EquiDQNNet",
    "CNNDQNNet",
]
