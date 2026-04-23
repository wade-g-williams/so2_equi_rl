"""make_env() dispatches to BulletArm or ManiSkill from a TrainConfig.

EnvStep lives here, not in wrapper.py, so maniskill_wrapper.py can import it
without pulling in helping_hands_rl_envs.
"""

from typing import NamedTuple

import torch


class EnvStep(NamedTuple):
    """env.step return shape. Accessed by attribute so adding fields is safe."""

    state: torch.Tensor
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    success: torch.Tensor = None  # task-internal success, (B,) float32 0/1


def make_env(cfg, *, seed: int, num_processes: int, num_envs: int = 1):
    """Build the train or eval env for cfg.env_backend.

    num_processes is BulletArm workers (0 = SingleRunner); num_envs is
    ManiSkill's vec count. Only one matters per backend.
    """
    backend = getattr(cfg, "env_backend", "bulletarm")
    if backend == "bulletarm":
        from so2_equi_rl.envs.wrapper import EnvWrapper

        return EnvWrapper(
            env_name=cfg.env_name,
            num_processes=num_processes,
            seed=seed,
            obs_size=cfg.obs_size,
            max_steps=cfg.max_steps,
            dpos=getattr(cfg, "dpos", None),
            drot=getattr(cfg, "drot", None),
        )
    if backend == "maniskill":
        # Lazy import so BulletArm-only workflows don't need ManiSkill installed.
        from so2_equi_rl.envs.maniskill_experts import get_expert
        from so2_equi_rl.envs.maniskill_wrapper import ManiSkillWrapper

        wrapper = ManiSkillWrapper(cfg=cfg, seed=seed, num_envs=num_envs)
        # Fail loud for unwired tasks rather than silent warmup no-op.
        wrapper.set_expert(get_expert(cfg.env_name))
        return wrapper
    raise ValueError(f"unknown env_backend: {backend!r}")
