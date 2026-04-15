"""Trainer-loop knobs shared by every agent variant: env name, seed, step
budget, log cadence, device, output dir. Agent-specific hyperparameters
live on subclasses so the trainer stays update-rule-agnostic.

Every field defaults. Dataclass inheritance puts base fields before subclass
fields, and a non-default after a default is a TypeError, so keeping
everything defaulted makes subclassing safe.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # Environment selection. Must be one of the close-loop task names
    # enumerated in envs/wrapper.py:_CLOSE_LOOP_ENVS.
    env_name: str = "close_loop_block_reaching"
    num_processes: int = 0  # 0 = SingleRunner, N = MultiRunner (N worker subprocesses)
    seed: int = 0
    obs_size: int = 128
    max_steps: int = 50  # per-episode step cap

    # Training budget and cadence.
    total_steps: int = 50_000  # total env steps to collect
    warmup_steps: int = 1_000  # random/expert collection before learning starts
    batch_size: int = 64
    buffer_capacity: int = 100_000

    # Logging / checkpoint cadence. All counts are in env steps.
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 10_000

    # Eval knobs. eval_seed is used to build a separate EnvWrapper so eval
    # rollouts don't perturb the training env's RNG.
    eval_episodes: int = 5
    eval_seed: int = 10_000

    # Output directory root. RunLogger creates a timestamped subdir per run.
    output_dir: str = "outputs"

    # None auto-selects cuda if available, else cpu, at run start.
    device: Optional[str] = None
