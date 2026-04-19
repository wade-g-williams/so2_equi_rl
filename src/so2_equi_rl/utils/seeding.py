"""Global seeding for torch (cpu+cuda), numpy, and python random. Called
once at the top of trainer.run(). EnvWrapper is NOT seeded here, its
PyBullet runner is seeded at construction via env_config["seed"].
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Safe to call without CUDA, becomes a no-op.
    torch.cuda.manual_seed_all(seed)
