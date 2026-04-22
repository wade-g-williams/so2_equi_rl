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
from so2_equi_rl.envs import make_env
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
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
        help="path to a pretrained encoder .pt from scripts/pretrain_ferm.py",
    )
    args = parser.parse_args()

    cfg = SACFERMConfig(**extract_dataclass_kwargs(args, SACFERMConfig))

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
        so2_aug_k=cfg.so2_aug_k,
    )

    # CNN triple is hardcoded; SACFERMAgent raises on anything else.
    agent = SACFERMAgent(
        cfg,
        encoder_cls=CNNEncoder,
        actor_cls=CNNActor,
        critic_cls=CNNCritic,
    )

    logger = RunLogger(
        cfg, run_name=args.run_name, alg_family="sac", alg_variant="ferm"
    )
    print(f"[train_sac_ferm] run dir: {logger.run_dir}")

    if args.pretrained_encoder is not None:
        # Paper Appendix F pretrains the FERM encoder for 1600 InfoNCE steps
        # before SAC kicks in. scripts/pretrain_ferm.py produces the .pt
        # we load here. actor and critic share agent.q_encoder by reference,
        # so updating q_encoder's state_dict propagates to them automatically.
        import torch

        state = torch.load(args.pretrained_encoder, map_location=agent.device)
        agent.q_encoder.load_state_dict(state["q_encoder"])
        agent.k_encoder.load_state_dict(state["q_encoder"])
        with torch.no_grad():
            agent.W.copy_(state["W"].to(agent.device))
        print(
            f"[train_sac_ferm] loaded pretrained encoder from {args.pretrained_encoder}"
        )

    trainer = SACTrainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_sac_ferm] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
