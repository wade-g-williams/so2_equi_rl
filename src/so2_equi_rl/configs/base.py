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

    # ManiSkill overhead camera knobs. Matches Joey's eq_sac maniskill branch:
    # 1 m camera, 60 deg FOV, symmetric gripper-relative depth clamp.
    ms3_camera_height: float = 1.0  # meters above workspace
    ms3_camera_fov: float = 60.0  # degrees, wide enough to cover ~1.15 m at 1 m
    ms3_depth_max: float = 2.0  # meters, symmetric clamp on gripper-relative depth
    ms3_control_mode: str = "pd_ee_delta_pose"
    ms3_sim_backend: str = "gpu"
    # MS3 reward mode. Defaults to 'normalized_dense' (MS3's own default),
    # which is shaped reward bounded to roughly [0, 1] per step. Other
    # valid options: 'dense' (unnormalized, unbounded), 'sparse' ({0, 1}
    # on success only, matches the paper's BulletArm regime), 'none'.
    ms3_reward_mode: str = "normalized_dense"

    # Training budget and cadence. total_steps counts UPDATE iterations
    # (gradient steps), matching paper repo's max_train_step and the x-axis
    # of paper Figures 6/7/8. 20000 is paper's spec for both DQN and SAC.
    total_steps: int = 20_000
    warmup_steps: int = 1_000  # kept for backward compat, superseded by warmup_episodes
    # Paper Appendix F: SAC warmup is 20 episodes, DQN is 100 episodes.
    warmup_episodes: int = 20
    batch_size: int = 64
    buffer_capacity: int = 100_000

    # Cadences in update count (matching paper repo main.py). Trainer enforces
    # cadence % n_updates_per_step == 0, which is trivially true for UTD=1.
    log_every: int = 32
    eval_every: int = (
        500  # paper spec (Fig 6/7/8: "evaluation every 500 training steps")
    )
    ckpt_every: int = 2_000

    # Eval rollouts use a separate EnvWrapper seeded from eval_seed so they
    # don't perturb the training env's RNG.
    eval_episodes: int = 5
    eval_seed: int = 10_000

    # RunLogger creates a timestamped subdir under this root.
    output_dir: str = "outputs"

    # None auto-selects cuda if available, else cpu.
    device: Optional[str] = None
