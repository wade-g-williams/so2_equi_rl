"""CLI entry point for DrQ-SAC training. Mirrors scripts/train_sac.py
with the DrQ config and agent.
"""

import argparse
from pathlib import Path

from so2_equi_rl.agents.sac_drq import SACDrQAgent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.sac_drq import SACDrQConfig
from so2_equi_rl.envs.wrapper import EnvWrapper
from so2_equi_rl.networks import (
    CNNActor,
    CNNCritic,
    CNNEncoder,
    EquiActor,
    EquiCritic,
    EquiEncoder,
)
from so2_equi_rl.trainers import SACTrainer
from so2_equi_rl.utils import set_seed
from so2_equi_rl.utils.cli_args import add_dataclass_args, extract_dataclass_kwargs
from so2_equi_rl.utils.logging import RunLogger

# DrQ reuses the vanilla SAC heads; only the update rule differs.
_ENCODER_VARIANTS = {
    "equi": (EquiEncoder, EquiActor, EquiCritic),
    "cnn": (CNNEncoder, CNNActor, CNNCritic),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DrQ-SAC on so2_equi_rl")
    add_dataclass_args(parser, SACDrQConfig)
    parser.add_argument(
        "--run-name", type=str, default=None, help="suffix appended to the run dir"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to a checkpoint .pt to resume from",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="equi",
        choices=sorted(_ENCODER_VARIANTS.keys()),
        help="encoder/head variant (default: equi)",
    )
    args = parser.parse_args()

    cfg = SACDrQConfig(**extract_dataclass_kwargs(args, SACDrQConfig))

    # Seed before any global-RNG draws. The aug RNG seeds itself from cfg.seed + a fixed offset.
    set_seed(cfg.seed)

    train_env = EnvWrapper(
        env_name=cfg.env_name,
        num_processes=cfg.num_processes,
        seed=cfg.seed,
        obs_size=cfg.obs_size,
        max_steps=cfg.max_steps,
    )
    eval_env = EnvWrapper(
        env_name=cfg.env_name,
        num_processes=0,
        seed=cfg.eval_seed,
        obs_size=cfg.obs_size,
        max_steps=cfg.max_steps,
    )

    buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        state_dim=1,
        obs_shape=(1, cfg.obs_size, cfg.obs_size),
        action_dim=cfg.action_dim,
        seed=cfg.seed,
    )

    encoder_cls, actor_cls, critic_cls = _ENCODER_VARIANTS[args.encoder]
    agent = SACDrQAgent(
        cfg,
        encoder_cls=encoder_cls,
        actor_cls=actor_cls,
        critic_cls=critic_cls,
    )

    logger = RunLogger(cfg, run_name=args.run_name)
    print(f"[train_sac_drq] run dir: {logger.run_dir}")

    trainer = SACTrainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_sac_drq] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
