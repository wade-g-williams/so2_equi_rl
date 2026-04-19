"""Trainer-loop knobs shared by every agent variant. Agent-specific
hyperparameters live on subclasses so the trainer stays update-rule-agnostic.

Every field defaults so dataclass inheritance stays safe (a non-default
after a default is a TypeError).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # Must be one of the close-loop tasks in envs/wrapper.py:_CLOSE_LOOP_ENVS.
    env_name: str = "close_loop_block_reaching"
    num_processes: int = 0  # 0 = SingleRunner, N = MultiRunner with N workers
    seed: int = 0
    obs_size: int = 128
    max_steps: int = 50  # per-episode cap

    # Training budget and cadence.
    total_steps: int = 50_000
    warmup_steps: int = 1_000  # random/expert collection before learning starts
    batch_size: int = 64
    buffer_capacity: int = 100_000

    # Cadences in env steps.
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 10_000

    # Eval rollouts use a separate EnvWrapper seeded from eval_seed so they
    # don't perturb the training env's RNG.
    eval_episodes: int = 5
    eval_seed: int = 10_000

    # RunLogger creates a timestamped subdir under this root.
    output_dir: str = "outputs"

    # None auto-selects cuda if available, else cpu.
    device: Optional[str] = None
