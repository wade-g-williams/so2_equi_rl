"""Learning-curve plots for Fig 6 (DQN family) and Fig 7 (SAC family).

Discovers runs under one or more --roots, groups by
(family, backend, task, variant, seed), aggregates seeds into mean + std
band, writes one PDF per (family, backend). Reads both the legacy
schema (info/parameters.json + info/eval_rewards.npy) and the new local
schema (config.yaml + metrics.jsonl), so outputs from either pipeline
aggregate cleanly.

    python scripts/plot_results.py --roots outputs/matrix --out outputs/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Poster typography, sized to stay legible at 36x24 in.
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
    }
)

# RAD dropped from both figures (scope lock 2026-04-20). The agents and
# runs still exist on disk, we just don't plot them.
VARIANT_ORDER_SAC = ["equi", "cnn", "drq", "ferm"]
VARIANT_ORDER_DQN = ["equi", "cnn", "drq", "curl"]

VARIANT_LABEL = {
    "equi": "Equi",
    "cnn": "CNN",
    "drq": "DrQ",
    "curl": "CURL",
    "ferm": "FERM",
}
# Equi drawn saturated; baselines muted. Stable palette across Fig 6/7.
VARIANT_COLOR = {
    "equi": "#2A9D8F",
    "cnn": "#64748B",
    "drq": "#E9C46A",
    "curl": "#8E44AD",
    "ferm": "#1B2A4E",
}

# BulletArm tasks are named close_loop_<task>; MS3 is keyed by ms3_task.
TASK_ORDER_BULLETARM = ["block_picking", "block_pulling", "drawer_opening"]
TASK_ORDER_MANISKILL = ["pickcube_v1", "pullcube_v1", "stackcube_v1"]
TASK_LABEL = {
    "block_picking": "Block Picking",
    "block_pulling": "Block Pulling",
    "drawer_opening": "Drawer Opening",
    "pickcube_v1": "PickCube-v1",
    "pullcube_v1": "PullCube-v1",
    "stackcube_v1": "StackCube-v1",
}

BACKEND_LABEL = {"bulletarm": "BulletArm", "maniskill": "ManiSkill3"}


_KNOWN_VARIANTS = {"equi", "cnn", "drq", "curl", "ferm", "rad"}


def _parse_alg(alg: str):
    # "equi_sac" -> ("sac", "equi"), "drq_dqn" -> ("dqn", "drq").
    # Filters out non-paper variants like "canon_sac" that showed up in
    # exploratory runs from earlier iterations of the codebase.
    parts = alg.split("_")
    if len(parts) != 2:
        return None
    variant, family = parts
    if family not in ("sac", "dqn"):
        return None
    if variant not in _KNOWN_VARIANTS:
        return None
    return family, variant


def _parse_backend(simulator: str):
    if simulator == "pybullet":
        return "bulletarm"
    if simulator == "maniskill3":
        return "maniskill"
    return None


def _parse_task(params: dict, backend: str):
    # BulletArm: strip close_loop_ prefix from env.
    # MS3: legacy runs store a placeholder env and the real task in ms3_task.
    if backend == "bulletarm":
        return params["env"].replace("close_loop_", "")
    if backend == "maniskill":
        ms3_task = params.get("ms3_task", "")
        return ms3_task.lower().replace("-", "_")
    return None


def _discover_legacy(run_dir: Path):
    # Legacy schema: info/parameters.json + info/eval_rewards.npy.
    # Returns a single run dict or None if the dir doesn't match.
    info = run_dir / "info"
    params_path = info / "parameters.json"
    eval_path = info / "eval_rewards.npy"
    if not (params_path.exists() and eval_path.exists()):
        return None

    eval_rewards = np.load(eval_path)
    if eval_rewards.size == 0:
        return None

    with params_path.open() as f:
        params = json.load(f)

    fam = _parse_alg(params.get("alg", ""))
    if fam is None:
        return None
    family, variant = fam

    backend = _parse_backend(params.get("simulator", ""))
    if backend is None:
        return None

    task = _parse_task(params, backend)
    if not task:
        return None

    seed_raw = params.get("seed")
    if seed_raw in (None, "None"):
        return None
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError):
        return None

    return {
        "run_dir": run_dir.name,
        "family": family,
        "variant": variant,
        "backend": backend,
        "task": task,
        "seed": seed,
        "eval_freq": int(params["eval_freq"]),
        "eval_rewards": eval_rewards,
        "n_evals": int(eval_rewards.size),
    }


def _discover_new(run_dir: Path):
    # New so2_equi_rl schema: config.yaml + metrics.jsonl. alg_family /
    # alg_variant are written into config.yaml by RunLogger (since the
    # TrainConfig dataclass doesn't include the CLI --network/--encoder).
    cfg_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not (cfg_path.exists() and metrics_path.exists()):
        return None

    try:
        import yaml
    except ImportError:
        # PyYAML is already a hard dep of RunLogger, so this is defensive.
        return None

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    family = cfg.get("alg_family")
    variant = cfg.get("alg_variant")
    if family not in ("sac", "dqn") or variant not in _KNOWN_VARIANTS:
        return None

    # cfg.env_backend is "bulletarm" / "maniskill" (already normalized).
    backend = cfg.get("env_backend")
    if backend not in ("bulletarm", "maniskill"):
        return None

    env_name = cfg.get("env_name", "")
    if backend == "bulletarm":
        task = env_name.replace("close_loop_", "")
    else:  # maniskill
        task = env_name.lower().replace("-", "_")
    if not task:
        return None

    seed_raw = cfg.get("seed")
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError):
        return None

    # Parse eval rows from metrics.jsonl. eval_every is the cadence.
    eval_returns = []
    with metrics_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval/return_disc_mean" in row:
                eval_returns.append(float(row["eval/return_disc_mean"]))
    if not eval_returns:
        return None

    eval_rewards = np.asarray(eval_returns, dtype=np.float32)

    return {
        "run_dir": run_dir.name,
        "family": family,
        "variant": variant,
        "backend": backend,
        "task": task,
        "seed": seed,
        "eval_freq": int(cfg.get("eval_every", 500)),
        "eval_rewards": eval_rewards,
        "n_evals": int(eval_rewards.size),
    }


def discover_runs(root: Path):
    # Walk a single output root. Yields dicts shaped for aggregation.
    # Tries legacy schema first, falls back to new schema
    # (local so2_equi_rl). Skips dirs that match neither.
    runs = []
    if not root.is_dir():
        return runs
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run = _discover_legacy(run_dir) or _discover_new(run_dir)
        if run is None:
            continue
        run["root"] = root
        runs.append(run)
    return runs


def dedup_by_identity(runs):
    # Dedup key: (family, variant, backend, task, seed). First root in --roots
    # wins on collisions; within a root, the longest curve wins. Lets us
    # override stale legacy-schema runs with fresh runs when needed.
    chosen: dict = {}
    for r in runs:
        key = (r["family"], r["variant"], r["backend"], r["task"], r["seed"])
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = r
            continue
        # Same-root collisions: keep the longer curve.
        if prev["root"] == r["root"] and r["n_evals"] > prev["n_evals"]:
            chosen[key] = r
    return list(chosen.values())


def group_for_plot(runs, family: str, backend: str):
    # {task: {variant: {"curves": (n_seeds, min_len), "steps": ndarray}}}
    filtered = [r for r in runs if r["family"] == family and r["backend"] == backend]
    by_task_variant: dict = {}
    for r in filtered:
        d = by_task_variant.setdefault(r["task"], {}).setdefault(
            r["variant"],
            {"seeds": [], "eval_freq": r["eval_freq"]},
        )
        d["seeds"].append(r["eval_rewards"])

    # Truncate to shortest curve per (task, variant) so stack is rectangular.
    for task, variants in by_task_variant.items():
        for variant, d in variants.items():
            min_len = min(len(s) for s in d["seeds"])
            d["curves"] = np.stack([s[:min_len] for s in d["seeds"]], axis=0)
            d["steps"] = np.arange(1, min_len + 1) * d["eval_freq"]
            del d["seeds"]
    return by_task_variant


def _variant_order(family: str):
    return VARIANT_ORDER_DQN if family == "dqn" else VARIANT_ORDER_SAC


def plot_panel(ax, variants_data, task_name, family, show_legend=False):
    for variant in _variant_order(family):
        if variant not in variants_data:
            continue
        d = variants_data[variant]
        curves, steps = d["curves"], d["steps"]
        mean = curves.mean(axis=0)
        std = curves.std(axis=0) if curves.shape[0] > 1 else np.zeros_like(mean)
        is_equi = variant == "equi"
        ax.plot(
            steps,
            mean,
            label=VARIANT_LABEL[variant],
            color=VARIANT_COLOR[variant],
            linewidth=3.0 if is_equi else 2.0,
            alpha=1.0 if is_equi else 0.8,
            zorder=3 if is_equi else 2,
        )
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color=VARIANT_COLOR[variant],
            alpha=0.22 if is_equi else 0.12,
            linewidth=0,
        )

    ax.set_title(TASK_LABEL.get(task_name, task_name))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Discounted eval return")
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.92)


def save_figure(grouped, family: str, backend: str, out_dir: Path):
    # One figure per (family, backend): 1xN panels across present tasks.
    task_order = (
        TASK_ORDER_BULLETARM if backend == "bulletarm" else TASK_ORDER_MANISKILL
    )
    tasks_present = [t for t in task_order if t in grouped]
    if not tasks_present:
        print(f"  skip {family}/{backend}: no runs found")
        return

    fig, axes = plt.subplots(
        1,
        len(tasks_present),
        figsize=(5.8 * len(tasks_present), 4.2),
        sharey=True,
    )
    if len(tasks_present) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks_present):
        plot_panel(ax, grouped[task], task, family, show_legend=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"Fig {'6' if family == 'dqn' else '7'} {family.upper()} on {BACKEND_LABEL[backend]}",
        y=1.02,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(len(handles), 1),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    fig_id = "figure6_dqn" if family == "dqn" else "figure7_sac"
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{fig_id}_{backend}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  wrote {out_path}")
    plt.close(fig)


def summarize(runs, out_dir: Path):
    # Small inventory table so you can tell at a glance which (family, backend,
    # task, variant) have how many seeds before looking at the PDFs.
    lines = ["family\tbackend\ttask\tvariant\tseeds"]
    counts: dict = {}
    for r in runs:
        key = (r["family"], r["backend"], r["task"], r["variant"])
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        family, backend, task, variant = key
        lines.append(f"{family}\t{backend}\t{task}\t{variant}\t{counts[key]}")
    out_path = out_dir / "run_inventory.tsv"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[Path("outputs")],
        help="one or more output roots to aggregate; first-root wins on dup identity",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for root in args.roots:
        print(f"Scanning {root}/ for runs...")
        found = discover_runs(root)
        # Also walk one level deep for roots like outputs/sac, outputs/ms3
        # that keep runs in a subdirectory rather than flat at the top.
        for sub in sorted(root.iterdir()) if root.is_dir() else []:
            if sub.is_dir() and not (sub / "info").exists():
                found += discover_runs(sub)
        print(f"  found {len(found)} plottable runs")
        all_runs.extend(found)

    runs = dedup_by_identity(all_runs)
    print(f"after dedup: {len(runs)} unique runs")

    summarize(runs, args.out)

    for family in ("dqn", "sac"):
        for backend in ("bulletarm", "maniskill"):
            grouped = group_for_plot(runs, family, backend)
            save_figure(grouped, family, backend, args.out)

    print("done.")


if __name__ == "__main__":
    main()
