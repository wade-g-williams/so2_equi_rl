"""Thin CLI shim. Real orchestrator logic lives in so2_equi_rl.launch so
tests can import it without sys.path contortions (see tests/conftest.py).
"""

# ruff: noqa: E402  (so2_equi_rl imports must come after the sys.path fix below)

import os
import sys

# `python -m scripts.launch_matrix` puts the repo root on sys.path, which
# makes helping_hands_rl_envs resolve to the namespace dir instead of the
# editable install. Drop it so the real package wins (see tests/conftest.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]

from so2_equi_rl.launch import main

if __name__ == "__main__":
    sys.exit(main())
