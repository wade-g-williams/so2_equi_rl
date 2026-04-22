"""FERM InfoNCE pretraining. Paper Appendix F, Table 9.

Runs 1600 InfoNCE steps on expert-demo observations before SAC training
starts. Paper reports this pretrain is what lets FERM learn block_pulling
at all, so SAC-FERM runs without it tend to flatline on that task.

Mirrors the InfoNCE loop in agents/sac_ferm.py (random crops, bilinear W,
Polyak key encoder) but skips actor, critic, and alpha updates. The saved
encoder weights load into train_sac_ferm.py via --pretrained-encoder.
"""

# ruff: noqa: E402  (so2_equi_rl imports must come after the sys.path fix below)

import argparse
import copy
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]

import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_equi_rl.configs.sac_ferm import SACFERMConfig
from so2_equi_rl.envs import make_env
from so2_equi_rl.networks import CNNEncoder
from so2_equi_rl.utils import augmentation as aug_mod
from so2_equi_rl.utils import set_seed, tile_state
from so2_equi_rl.utils.cli_args import add_dataclass_args, extract_dataclass_kwargs
from so2_equi_rl.utils.logging import RunLogger

# Paper Appendix F uses 1600 pretrain steps.
_DEFAULT_PRETRAIN_STEPS = 1600

# Same offset as agents/sac_ferm.py so the aug RNG lines up when the
# pretrained encoder is loaded and training continues.
_AUG_SEED_OFFSET = 3407


def collect_demo_obs(env, warmup_steps: int):
    # Rolls the scripted expert for `warmup_steps` env steps, stashing the
    # post-step (obs, state) tuples. Only these two fields feed InfoNCE;
    # action, reward, next_obs, and done are not needed for pretrain.
    obs_chunks = []
    state_chunks = []

    env.reset()
    for _ in range(warmup_steps):
        action_physical = env.get_expert_action()
        step = env.step(action_physical)
        obs_chunks.append(step.obs.cpu())
        state_chunks.append(step.state.cpu())

    obs_pool = torch.cat(obs_chunks, dim=0)
    state_pool = torch.cat(state_chunks, dim=0)
    return obs_pool, state_pool


def pretrain(
    q_encoder: nn.Module,
    k_encoder: nn.Module,
    W: nn.Parameter,
    obs_pool: torch.Tensor,
    state_pool: torch.Tensor,
    cfg: SACFERMConfig,
    n_steps: int,
    device: torch.device,
    aug_gen: torch.Generator,
    logger: RunLogger,
):
    # 1600 InfoNCE steps on random mini-batches of the demo pool. Matches
    # the step in agents/sac_ferm.py::update but with no critic/actor/alpha.
    optimizer = torch.optim.Adam(
        list(q_encoder.parameters()) + [W],
        lr=cfg.encoder_lr,
    )
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(int(cfg.seed) + 7777)

    pool_size = obs_pool.shape[0]
    B = cfg.batch_size

    for step in range(n_steps):
        idx = torch.randint(0, pool_size, (B,), generator=sampler)
        obs_batch = obs_pool[idx]
        state_batch = state_pool[idx].to(device)

        # Two independent random crops, same mechanics as the SAC-FERM update.
        q_obs = aug_mod.random_crop(obs_batch, pad=cfg.ferm_pad, generator=aug_gen).to(
            device
        )
        k_obs = aug_mod.random_crop(obs_batch, pad=cfg.ferm_pad, generator=aug_gen).to(
            device
        )
        q_tiled = tile_state(q_obs, state_batch)
        k_tiled = tile_state(k_obs, state_batch)

        q_feat = q_encoder(q_tiled).view(B, -1)
        with torch.no_grad():
            k_feat = k_encoder(k_tiled).view(B, -1)

        # Bilinear logits (B, B). Diagonal positives, off-diagonal negatives.
        Wk = torch.matmul(W, k_feat.t())
        logits = torch.matmul(q_feat, Wk) / cfg.curl_temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        (cfg.curl_lambda * loss).backward()
        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(q_encoder.parameters()) + [W], cfg.grad_clip_norm
            )
        optimizer.step()

        # Polyak update the frozen key encoder.
        with torch.no_grad():
            for p_q, p_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                p_k.mul_(1.0 - cfg.curl_tau).add_(p_q.data, alpha=cfg.curl_tau)

        if step % cfg.log_every == 0:
            logger.log_scalars({"pretrain/infonce_loss": loss.item()}, step=step)
            print(f"[pretrain_ferm] step {step}/{n_steps}  loss={loss.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FERM InfoNCE pretrain on so2_equi_rl")
    add_dataclass_args(parser, SACFERMConfig)
    parser.add_argument(
        "--run-name", type=str, default=None, help="suffix appended to the run dir"
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=_DEFAULT_PRETRAIN_STEPS,
        help=f"InfoNCE pretrain steps (paper Appendix F: {_DEFAULT_PRETRAIN_STEPS})",
    )
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=None,
        help="expert-rollout env steps used to build the demo pool (default: cfg.warmup_steps)",
    )
    args = parser.parse_args()

    cfg = SACFERMConfig(**extract_dataclass_kwargs(args, SACFERMConfig))
    demo_steps = args.demo_steps if args.demo_steps is not None else cfg.warmup_steps

    set_seed(cfg.seed)

    env = make_env(
        cfg,
        seed=cfg.seed,
        num_processes=cfg.num_processes,
        num_envs=cfg.num_envs,
    )

    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Build the encoder + momentum key + bilinear W. Same construction as
    # agents/sac_ferm.py so the saved weights drop into SAC-FERM cleanly.
    enc_kwargs = {
        "obs_channels": cfg.obs_channels,
        "n_hidden": cfg.n_hidden,
        "group_order": cfg.group_order,
    }
    q_encoder = CNNEncoder(**enc_kwargs).to(device)
    k_encoder = copy.deepcopy(q_encoder)
    for p in k_encoder.parameters():
        p.requires_grad_(False)
    W = nn.Parameter(
        torch.empty(q_encoder.output_dim, q_encoder.output_dim, device=device)
    )
    nn.init.orthogonal_(W)

    logger = RunLogger(
        cfg,
        run_name=args.run_name,
        alg_family="sac",
        alg_variant="ferm_pretrain",
    )
    print(f"[pretrain_ferm] run dir: {logger.run_dir}")
    print(f"[pretrain_ferm] collecting {demo_steps} env steps of expert demos...")

    obs_pool, state_pool = collect_demo_obs(env, demo_steps)
    print(
        f"[pretrain_ferm] demo pool: "
        f"obs={tuple(obs_pool.shape)} state={tuple(state_pool.shape)}"
    )

    aug_gen = torch.Generator(device="cpu")
    aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    print(f"[pretrain_ferm] running {args.pretrain_steps} InfoNCE steps...")
    pretrain(
        q_encoder,
        k_encoder,
        W,
        obs_pool,
        state_pool,
        cfg,
        args.pretrain_steps,
        device,
        aug_gen,
        logger,
    )

    # Save the trained encoder + bilinear projection. train_sac_ferm.py
    # loads q_encoder into both q_encoder and k_encoder (same weights), and
    # copies W over.
    save_path = logger.ckpt_dir / "pretrained_encoder.pt"
    torch.save(
        {
            "q_encoder": q_encoder.state_dict(),
            "W": W.detach().cpu(),
            "pretrain_steps": args.pretrain_steps,
            "demo_steps": demo_steps,
            "env_name": cfg.env_name,
            "env_backend": cfg.env_backend,
            "seed": cfg.seed,
        },
        save_path,
    )

    env.close()
    print(f"[pretrain_ferm] saved pretrained encoder to: {save_path}")
    print(
        "[pretrain_ferm] load into SAC-FERM with: "
        f"python scripts/train_sac_ferm.py --pretrained-encoder {save_path} ..."
    )


if __name__ == "__main__":
    main()
