"""CLI entry point for DQN training. Auto-registers DQNConfig fields as
--kebab-case flags, builds the six components, hands them to DQNTrainer.
Loop logic lives in trainers/.
"""

# ruff: noqa: E402  (so2_equi_rl imports must come after the sys.path fix below)

import argparse
import os
import sys
from pathlib import Path

# `python -m scripts.train_dqn` puts the repo root on sys.path, which makes
# helping_hands_rl_envs resolve to the namespace dir instead of the
# editable install. Drop it so the real package wins (see tests/conftest.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]

from so2_equi_rl.agents.dqn import DQNAgent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.dqn import DQNConfig
from so2_equi_rl.envs.wrapper import EnvWrapper
from so2_equi_rl.networks import CNNDQNNet, EquiDQNNet
from so2_equi_rl.trainers import DQNTrainer
from so2_equi_rl.utils import set_seed
from so2_equi_rl.utils.cli_args import add_dataclass_args, extract_dataclass_kwargs
from so2_equi_rl.utils.logging import RunLogger

# Maps --network to the Q-net class DQNAgent's DI ctor expects.
_Q_VARIANTS = {
    "equi": EquiDQNNet,
    "cnn": CNNDQNNet,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN on so2_equi_rl")
    add_dataclass_args(parser, DQNConfig)
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
        "--network",
        type=str,
        default="equi",
        choices=sorted(_Q_VARIANTS.keys()),
        help="Q-net variant (default: equi)",
    )
    args = parser.parse_args()

    cfg = DQNConfig(**extract_dataclass_kwargs(args, DQNConfig))

    # Seed before constructing anything that draws from the global torch RNG.
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

    # DQN stores integer grid indices (cast to float32), not [-1, 1] actions,
    # so the buffer's range guard has to be off.
    buffer = ReplayBuffer(
        capacity=cfg.buffer_capacity,
        state_dim=1,  # scalar gripper open/close
        obs_shape=(1, cfg.obs_size, cfg.obs_size),
        action_dim=cfg.action_dim,
        seed=cfg.seed,
        enforce_unscaled_action_range=False,
    )

    net_cls = _Q_VARIANTS[args.network]
    agent = DQNAgent(cfg, net_cls=net_cls)

    logger = RunLogger(cfg, run_name=args.run_name)
    print(f"[train_dqn] run dir: {logger.run_dir}")

    trainer = DQNTrainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_dqn] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
