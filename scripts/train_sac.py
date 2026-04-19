"""CLI entry point for SAC training. Auto-registers SACConfig fields as
--kebab-case flags, builds the six components, hands them to SACTrainer.
Loop logic lives in trainers/.
"""

import argparse
from pathlib import Path

from so2_equi_rl.agents.sac import SACAgent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.sac import SACConfig
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

# Maps --encoder to the (encoder, actor, critic) triple SACAgent's DI ctor expects.
_ENCODER_VARIANTS = {
    "equi": (EquiEncoder, EquiActor, EquiCritic),
    "cnn": (CNNEncoder, CNNActor, CNNCritic),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on so2_equi_rl")
    add_dataclass_args(parser, SACConfig)
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

    cfg = SACConfig(**extract_dataclass_kwargs(args, SACConfig))

    # Seed before constructing anything that draws from the global torch RNG
    # (network init, env, buffer). Resume restores RNG state from the ckpt.
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
        num_processes=0,  # eval is single-slot
        seed=cfg.eval_seed,
        obs_size=cfg.obs_size,
        max_steps=cfg.max_steps,
    )

    buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        state_dim=1,  # scalar gripper open/close
        obs_shape=(1, cfg.obs_size, cfg.obs_size),
        action_dim=cfg.action_dim,
        seed=cfg.seed,
    )

    encoder_cls, actor_cls, critic_cls = _ENCODER_VARIANTS[args.encoder]
    agent = SACAgent(
        cfg,
        encoder_cls=encoder_cls,
        actor_cls=actor_cls,
        critic_cls=critic_cls,
    )

    logger = RunLogger(cfg, run_name=args.run_name)
    print(f"[train_sac] run dir: {logger.run_dir}")

    trainer = SACTrainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_sac] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
