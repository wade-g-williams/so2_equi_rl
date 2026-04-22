"""Trainer-loop knobs shared by every agent variant. Agent-specific
hyperparameters live on subclasses so the trainer stays update-rule-agnostic.

Every field defaults so dataclass inheritance stays safe (a non-default
after a default is a TypeError).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    # Env selection. env_name is the task identity (shared across backends),
    # env_backend picks the simulator.
    env_name: str = "close_loop_block_reaching"
    env_backend: str = "bulletarm"  # "bulletarm" or "maniskill"
    num_processes: int = 0  # BulletArm only: 0 = SingleRunner, N = MultiRunner
    num_envs: int = 1  # ManiSkill only: GPU-vectorized env batch size
    seed: int = 0
    obs_size: int = 128
    max_steps: int = 50  # per-episode cap

    # ManiSkill orthographic-approximation knobs; ortho keeps SO(2) equivariance.
    ms3_camera_height: float = 1.4  # meters above workspace
    ms3_camera_fov: float = 15.0  # degrees, smaller = closer to ortho
    ms3_depth_max: float = 0.4  # meters, depth clip
    ms3_control_mode: str = "pd_ee_delta_pose"
    ms3_sim_backend: str = "gpu"
    # MS3 reward mode. Defaults to 'normalized_dense' (MS3's own default),
    # which is shaped reward bounded to roughly [0, 1] per step. Other
    # valid options: 'dense' (unnormalized, unbounded), 'sparse' ({0, 1}
    # on success only, matches the paper's BulletArm regime), 'none'.
    ms3_reward_mode: str = "normalized_dense"

    # Training budget and cadence.
    total_steps: int = 50_000
    warmup_steps: int = 1_000  # random/expert collection before learning starts
    batch_size: int = 64
    buffer_capacity: int = 100_000

    # Cadences in env steps. All multiples of 160 (LCM of likely batch
    # sizes: BulletArm num_processes 1/5, MS3 num_envs 8/16/32). The
    # trainer enforces cadence % batch_size == 0, so these must stay
    # divisible by whatever batch size the env ends up with.
    log_every: int = 160
    eval_every: int = 4_800
    ckpt_every: int = 9_600

    # Eval rollouts use a separate EnvWrapper seeded from eval_seed so they
    # don't perturb the training env's RNG.
    eval_episodes: int = 5
    eval_seed: int = 10_000

    # RunLogger creates a timestamped subdir under this root.
    output_dir: str = "outputs"

    # None auto-selects cuda if available, else cpu.
    device: Optional[str] = None
