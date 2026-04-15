"""Global seeding: torch (CPU + CUDA), numpy, Python random. Called once
at the top of the trainer's run() so every run with the same seed replays
the same trajectory. EnvWrapper is NOT seeded here -- its underlying
PyBullet runner is seeded at construction via env_config["seed"], so the
trainer just needs to pass the same seed into EnvWrapper(...).
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # Cover every process-level RNG the training loop touches.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Safe to call even without CUDA compiled in; becomes a no-op.
    torch.cuda.manual_seed_all(seed)
