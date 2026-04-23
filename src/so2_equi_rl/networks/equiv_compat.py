"""Import shim. Prefers escnn (torch >= 2.0 for ms3), falls back to e2cnn (torch 1.7).
escnn renamed gspaces.Rot2dOnR2 to lowercase rot2dOnR2; alias it back.
"""

try:
    from escnn import gspaces, nn as enn  # noqa: F401

    if not hasattr(gspaces, "Rot2dOnR2"):
        gspaces.Rot2dOnR2 = gspaces.rot2dOnR2
except ImportError:
    from e2cnn import gspaces, nn as enn  # noqa: F401

__all__ = ["gspaces", "enn"]
