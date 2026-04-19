"""Auto-registers config-dataclass fields as --kebab-case argparse flags.
Shared by the train_*.py scripts so adding a config field doesn't mean
editing every CLI by hand.
"""

import argparse
import dataclasses
from typing import Any, Callable, Dict, Union


def _str2bool(s: str) -> bool:
    # argparse's default bool cast is broken: bool("False") == True.
    low = s.lower()
    if low in ("true", "t", "yes", "y", "1"):
        return True
    if low in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {s!r}")


def _parse_with_none(inner_cast: Callable[[str], Any]) -> Callable[[str], Any]:
    # Wraps a cast so "none"/"null" yield Python None. Lets the CLI
    # explicitly override an Optional field back to None.
    def parse(s: str) -> Any:
        if s.lower() in ("none", "null"):
            return None
        return inner_cast(s)

    return parse


def _unwrap_optional(ftype: Any):
    # Optional[X] = Union[X, None] -> (X, True). Bare types -> (X, False).
    origin = getattr(ftype, "__origin__", None)
    if origin is Union:
        non_none = [a for a in ftype.__args__ if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return non_none[0], True
    return ftype, False


def _is_tuple_type(ftype: Any) -> bool:
    # Tuple fields (only p_range today) skip auto-registration; would need nargs plumbing.
    return getattr(ftype, "__origin__", None) is tuple


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    # Register --kebab-name flags for int, float, str, bool fields (and
    # Optional[X] for each). Silently skips unsupported shapes.
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
            continue

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
    # Filter Namespace down to fields the dataclass ctor accepts. Skipped
    # fields (tuples) get filtered out by hasattr.
    return {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(cls)
        if hasattr(args, f.name)
    }
