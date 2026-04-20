"""CLI entry point for RAD-SAC training. Mirrors scripts/train_sac.py
with the RAD config and agent. One shared SO(2) rotation per transition,
no K/M averaging (that's DrQ).
"""

# ruff: noqa: E402  (so2_equi_rl imports must come after the sys.path fix below)

import argparse
import os
import sys
from pathlib import Path

# `python -m scripts.train_sac_rad` puts the repo root on sys.path, which
# makes helping_hands_rl_envs resolve to the namespace dir instead of the
# editable install. Drop it so the real package wins (see tests/conftest.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]

from so2_equi_rl.agents.sac_rad import SACRADAgent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.sac_rad import SACRADConfig
from so2_equi_rl.envs import make_env
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

# RAD reuses vanilla SAC heads; only the transition-rotation step is extra.
# equi+RAD stays as an ablation even though the paper only uses RAD on CNN.
_ENCODER_VARIANTS = {
    "equi": (EquiEncoder, EquiActor, EquiCritic),
    "cnn": (CNNEncoder, CNNActor, CNNCritic),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RAD-SAC on so2_equi_rl")
    add_dataclass_args(parser, SACRADConfig)
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

    cfg = SACRADConfig(**extract_dataclass_kwargs(args, SACRADConfig))

    # Seed before any global-RNG draws. The aug RNG seeds itself from cfg.seed + a fixed offset.
    set_seed(cfg.seed)

    train_env = make_env(
        cfg,
        seed=cfg.seed,
        num_processes=cfg.num_processes,
        num_envs=cfg.num_envs,
    )
    eval_env = make_env(cfg, seed=cfg.eval_seed, num_processes=0, num_envs=1)

    buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        state_dim=1,
        obs_shape=(1, cfg.obs_size, cfg.obs_size),
        action_dim=cfg.action_dim,
        seed=cfg.seed,
    )

    encoder_cls, actor_cls, critic_cls = _ENCODER_VARIANTS[args.encoder]
    agent = SACRADAgent(
        cfg,
        encoder_cls=encoder_cls,
        actor_cls=actor_cls,
        critic_cls=critic_cls,
    )

    logger = RunLogger(cfg, run_name=args.run_name)
    print(f"[train_sac_rad] run dir: {logger.run_dir}")

    trainer = SACTrainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_sac_rad] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
