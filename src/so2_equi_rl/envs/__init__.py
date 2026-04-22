"""make_env() dispatches to the right backend (BulletArm or ManiSkill)
from a TrainConfig, so train_*.py scripts stay backend-agnostic.

EnvStep lives here rather than in wrapper.py so that maniskill_wrapper.py
can import it without dragging in helping_hands_rl_envs. The hhe __file__
patch that wrapper.py needs also stays out of this __init__'s import
path, so ms3-only envs don't need hhe installed.
"""

from typing import NamedTuple

import torch


class EnvStep(NamedTuple):
    """Return shape for env.step in both backends. Trainers access fields
    by attribute, not positional unpacking, so adding fields is safe.
    """

    state: torch.Tensor
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    success: torch.Tensor = None  # task-internal success, (B,) float32 0/1


def make_env(cfg, *, seed: int, num_processes: int, num_envs: int = 1):
    """Build the training or eval env for cfg.env_backend.

    num_processes is BulletArm's worker count (0 = SingleRunner).
    num_envs is ManiSkill's GPU-vectorized env count.
    Only one is meaningful per backend; the other is ignored.
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
        # Lazy-imported so BulletArm-only workflows don't need ManiSkill installed.
        from so2_equi_rl.envs.maniskill_experts import get_expert
        from so2_equi_rl.envs.maniskill_wrapper import ManiSkillWrapper

        wrapper = ManiSkillWrapper(cfg=cfg, seed=seed, num_envs=num_envs)
        # Register the scripted expert; raises for tasks we haven't wired yet
        # (e.g. drawer_opening). Fail-loud is preferred to silent no-op at warmup.
        wrapper.set_expert(get_expert(cfg.env_name))
        return wrapper
    raise ValueError(f"unknown env_backend: {backend!r}")
