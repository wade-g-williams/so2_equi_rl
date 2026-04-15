"""CLI entry point for SAC training.

Auto-registers every SACConfig field as a --kebab-case flag, builds the six
components, hands them to Trainer, and runs. Thin on purpose; loop logic
lives in trainers/trainer.py.
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Union

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
from so2_equi_rl.trainers.trainer import Trainer
from so2_equi_rl.utils import set_seed
from so2_equi_rl.utils.logging import RunLogger

# Maps the --encoder flag to the (encoder, actor, critic) triple that
# SACAgent's DI constructor expects. One entry per baseline variant.
_ENCODER_VARIANTS = {
    "equi": (EquiEncoder, EquiActor, EquiCritic),
    "cnn": (CNNEncoder, CNNActor, CNNCritic),
}


def _str2bool(s: str) -> bool:
    # argparse's default bool cast is broken: bool("False") == True.
    # Accept the usual truthy/falsey tokens case-insensitively.
    low = s.lower()
    if low in ("true", "t", "yes", "y", "1"):
        return True
    if low in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {s!r}")


def _parse_with_none(inner_cast: Callable[[str], Any]) -> Callable[[str], Any]:
    # Wraps an inner cast so the string "none"/"null" yields Python None.
    # Lets the CLI explicitly override an Optional field back to None.
    def parse(s: str) -> Any:
        if s.lower() in ("none", "null"):
            return None
        return inner_cast(s)

    return parse


def _unwrap_optional(ftype: Any):
    # Returns (inner_type, is_optional). For Optional[X] = Union[X, None]:
    # pulls X out and flags True. Leaves bare types untouched.
    origin = getattr(ftype, "__origin__", None)
    if origin is Union:
        non_none = [a for a in ftype.__args__ if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return non_none[0], True
    return ftype, False


def _is_tuple_type(ftype: Any) -> bool:
    # Skipped by the auto-register loop: Tuple fields (only p_range today)
    # are unlikely one-off CLI overrides and would need nargs plumbing.
    return getattr(ftype, "__origin__", None) is tuple


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    # Register --kebab-name flags for every supported field on cls.
    # Supported: int / float / str / bool, with Optional[X] for each.
    # Silently skips unsupported shapes (tuples, nested dataclasses, etc).
    for f in dataclasses.fields(cls):
        ftype = f.type
        if _is_tuple_type(ftype):
            continue

        inner, is_optional = _unwrap_optional(ftype)

        if inner is bool:
            cast: Callable[[str], Any] = _str2bool
        elif inner in (int, float, str):
            cast = inner
        else:
            continue  # skip unknown types rather than crashing

        if is_optional:
            cast = _parse_with_none(cast)

        flag = "--" + f.name.replace("_", "-")
        parser.add_argument(
            flag,
            type=cast,
            default=f.default,
            help=f"{inner.__name__} (default: {f.default!r})",
        )


def _extract_dataclass_kwargs(args: argparse.Namespace, cls: type) -> Dict[str, Any]:
    # Build a kwargs dict the dataclass ctor accepts. Fields we skipped in
    # _add_dataclass_args (tuples) get filtered out by hasattr.
    return {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(cls)
        if hasattr(args, f.name)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on so2_equi_rl")
    _add_dataclass_args(parser, SACConfig)
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

    cfg = SACConfig(**_extract_dataclass_kwargs(args, SACConfig))

    # Seed before constructing anything that draws from the global torch RNG
    # (network init, env, buffer). Resume restores RNG state from the ckpt.
    set_seed(cfg.seed)

    # --- collaborators ---
    train_env = EnvWrapper(
        env_name=cfg.env_name,
        num_processes=cfg.num_processes,
        seed=cfg.seed,
        obs_size=cfg.obs_size,
        max_steps=cfg.max_steps,
    )
    eval_env = EnvWrapper(
        env_name=cfg.env_name,
        num_processes=0,  # eval is single-slot; cfg.num_processes is training only
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

    trainer = Trainer(cfg, agent, train_env, eval_env, buffer, logger)
    trainer.run(resume_path=args.resume)

    print(f"[train_sac] done. artifacts at: {logger.run_dir}")


if __name__ == "__main__":
    main()
