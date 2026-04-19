"""Learning-curve plots from outputs/<run>/info/eval_rewards.npy. Auto-discovers
run dirs by reading parameters.json, drops smoke-test runs (empty eval),
aggregates seeds into mean + 1-std band per task.

    python scripts/plot_results.py --outputs outputs --out outputs/plots
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

# Canonical plot order. Equi SAC drawn saturated; baselines muted.
ALG_ORDER = ["equi_sac", "cnn_sac", "drq_sac", "ferm_sac", "rad_sac"]
ALG_LABEL = {
    "equi_sac": "Equi SAC",
    "cnn_sac": "CNN SAC",
    "drq_sac": "DrQ SAC",
    "ferm_sac": "FERM SAC",
    "rad_sac": "RAD SAC",
}
ALG_COLOR = {
    "equi_sac": "#2A9D8F",  # teal, highlighted
    "cnn_sac": "#64748B",
    "drq_sac": "#E9C46A",
    "ferm_sac": "#1B2A4E",
    "rad_sac": "#E76F51",
}

ENV_ORDER = ["block_picking", "block_pulling", "drawer_opening"]
ENV_LABEL = {
    "block_picking": "Block Picking",
    "block_pulling": "Block Pulling",
    "drawer_opening": "Drawer Opening",
}


def discover_runs(outputs_dir: Path):
    # Walk outputs/, load each run's eval curve and metadata, skip smoke tests.
    runs = []
    for run_dir in sorted(outputs_dir.iterdir()):
        info = run_dir / "info"
        params_path = info / "parameters.json"
        eval_path = info / "eval_rewards.npy"
        if not (params_path.exists() and eval_path.exists()):
            continue

        eval_rewards = np.load(eval_path)
        if eval_rewards.size == 0:
            continue  # smoke test, no eval data

        with params_path.open() as f:
            params = json.load(f)

        # env field is "close_loop_<task>", strip the prefix for grouping.
        env = params["env"].replace("close_loop_", "")
        runs.append(
            {
                "alg": params["alg"],
                "env": env,
                "seed": int(params["seed"]),
                "eval_freq": int(params["eval_freq"]),
                "max_train_step": int(params["max_train_step"]),
                "eval_rewards": eval_rewards,
            }
        )
    return runs


def group_by_env_alg(runs):
    # {env: {alg: {"steps": ndarray, "seeds": [arr_per_seed], "eval_freq": int}}}
    grouped = {}
    for r in runs:
        env_d = grouped.setdefault(r["env"], {})
        alg_d = env_d.setdefault(
            r["alg"],
            {
                "seeds": [],
                "eval_freq": r["eval_freq"],
                "n_evals": r["eval_rewards"].size,
            },
        )
        alg_d["seeds"].append(r["eval_rewards"])

    # Truncate to the shortest curve per (env, alg) so np.stack works if a run was cut short.
    for env, algs in grouped.items():
        for alg, d in algs.items():
            min_len = min(len(s) for s in d["seeds"])
            stacked = np.stack([s[:min_len] for s in d["seeds"]], axis=0)
            d["curves"] = stacked
            d["n_evals"] = min_len
            d["steps"] = np.arange(1, min_len + 1) * d["eval_freq"]
            del d["seeds"]
    return grouped


def plot_env(ax, env_data, env_name, show_legend=True):
    # One panel per env. Each algorithm draws a mean curve with a 1-std
    # shaded band, in canonical order.
    for alg in ALG_ORDER:
        if alg not in env_data:
            continue
        d = env_data[alg]
        curves = d["curves"]
        steps = d["steps"]
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)

        is_ours = alg == "equi_sac"
        ax.plot(
            steps,
            mean,
            label=ALG_LABEL[alg],
            color=ALG_COLOR[alg],
            linewidth=3.0 if is_ours else 2.0,
            alpha=1.0 if is_ours else 0.75,
            zorder=3 if is_ours else 2,
        )
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color=ALG_COLOR[alg],
            alpha=0.22 if is_ours else 0.12,
            linewidth=0,
        )

    ax.set_title(ENV_LABEL[env_name])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.92)


def save_per_env_figures(grouped, out_dir: Path):
    # One PDF per task, full legend on each.
    for env in ENV_ORDER:
        if env not in grouped:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        plot_env(ax, grouped[env], env, show_legend=True)
        fig.tight_layout()
        out_path = out_dir / f"learning_curve_{env}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  wrote {out_path}")


def save_combined_figure(grouped, out_dir: Path):
    # 1x3 panel for the report. Single shared legend below the row.
    envs_present = [e for e in ENV_ORDER if e in grouped]
    fig, axes = plt.subplots(
        1, len(envs_present), figsize=(5.8 * len(envs_present), 4.2), sharey=True
    )
    if len(envs_present) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs_present):
        plot_env(ax, grouped[env], env, show_legend=False)

    # Pull legend handles from the first axis (all algs share styling)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = out_dir / "learning_curves_combined.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.outputs}/ for runs...")
    runs = discover_runs(args.outputs)
    print(f"  found {len(runs)} plottable runs")

    grouped = group_by_env_alg(runs)
    for env in ENV_ORDER:
        if env not in grouped:
            continue
        algs_in = [a for a in ALG_ORDER if a in grouped[env]]
        seed_counts = [grouped[env][a]["curves"].shape[0] for a in algs_in]
        print(f"  {env}: {dict(zip(algs_in, seed_counts))} seeds per alg")

    print(f"Writing figures to {args.out}/")
    save_per_env_figures(grouped, args.out)
    save_combined_figure(grouped, args.out)
    print("done.")


if __name__ == "__main__":
    main()
