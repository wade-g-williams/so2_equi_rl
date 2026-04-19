"""Auto-registers the fields of a config dataclass as --kebab-case argparse
flags. Shared by scripts/train_sac.py and scripts/train_sac_drq.py so the
two CLIs stay in lockstep as configs evolve.
"""

import argparse
import dataclasses
from typing import Any, Callable, Dict, Union


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


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
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


def extract_dataclass_kwargs(args: argparse.Namespace, cls: type) -> Dict[str, Any]:
    # Build a kwargs dict the dataclass ctor accepts. Fields skipped in
    # add_dataclass_args (tuples) get filtered out by hasattr.
    return {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(cls)
        if hasattr(args, f.name)
    }
