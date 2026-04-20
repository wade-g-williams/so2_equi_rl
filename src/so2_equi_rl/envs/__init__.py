"""make_env() dispatches to the right backend (BulletArm or ManiSkill)
from a TrainConfig so the train_*.py scripts stay backend-agnostic.

The hhe __file__ patch lives at the top of envs/wrapper.py; importing
this __init__ no longer forces a helping_hands_rl_envs install, which
lets the ms3 env run these scripts without BulletArm in scope.
"""


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
