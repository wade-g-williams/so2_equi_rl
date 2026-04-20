"""Equivariant-network import shim. Prefers escnn (torch >= 2.0, needed by
the ms3 env), falls back to e2cnn for the paper-replication env (torch 1.7).
escnn renamed gspaces.Rot2dOnR2 to lowercase rot2dOnR2; alias it back so
encoders/heads don't care which library resolved.
"""

try:
    from escnn import gspaces, nn as enn  # noqa: F401

    if not hasattr(gspaces, "Rot2dOnR2"):
        gspaces.Rot2dOnR2 = gspaces.rot2dOnR2
except ImportError:
    from e2cnn import gspaces, nn as enn  # noqa: F401

__all__ = ["gspaces", "enn"]
