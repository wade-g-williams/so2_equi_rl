"""CLI entry point for FERM-SAC training. Mirrors scripts/train_sac.py
with the FERM config and agent. No --encoder flag because FERM's InfoNCE
pretext learns the rotation-invariance that equi already bakes in.
"""

# ruff: noqa: E402  (so2_equi_rl imports must come after the sys.path fix below)

import argparse
import os
import sys
from pathlib import Path

# `python -m scripts.train_sac_ferm` puts the repo root on sys.path, which
# makes helping_hands_rl_envs resolve to the namespace dir instead of the
# editable install. Drop it so the real package wins (see tests/conftest.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]

from so2_equi_rl.agents.sac_ferm import SACFERMAgent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.sac_ferm import SACFERMConfig
from so2_equi_rl.envs.wrapper import EnvWrapper
from so2_equi_rl.networks import CNNActor, CNNCritic, CNNEncoder
from so2_equi_rl.trainers import SACTrainer
from so2_equi_rl.utils import set_seed
from so2_equi_rl.utils.cli_args import add_dataclass_args, extract_dataclass_kwargs
from so2_equi_rl.utils.logging import RunLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FERM-SAC on so2_equi_rl")
    add_dataclass_args(parser, SACFERMConfig)
    parser.add_argument(
        "--run-name", type=str, default=None, help="suffix appended to the run dir"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to a checkpoint .pt to resume from",
    )
    args = parser.parse_args()

    cfg = SACFERMConfig(**extract_dataclass_kwargs(args, SACFERMConfig))

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

    # CNN triple is hardcoded; SACFERMAgent raises on anything else.
    agent = SACFERMAgent(
        cfg,
        encoder_cls=CNNEncoder,
        actor_cls=CNNActor,
        critic_cls=CNNCritic,
    )

    logger = RunLogger(cfg, run_name=args.run_name)
    print(f"[train_sac_ferm] run dir: {logger.run_dir}")

    trainer = SACTrainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_sac_ferm] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
