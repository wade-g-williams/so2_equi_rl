"""Matrix launch orchestrator. Loads a YAML spec of reproduction cells,
gates each launch on free VRAM, runs sequentially or through a worker pool.
Lives here (not scripts/) so tests can import it without sys.path games.

Gate rationale: Popen returns before CUDA actually allocates VRAM, so two
workers polling nvidia-smi could both see a large free number and
overcommit. VRAMGate tracks reserved MiB in-process and consults
free-reserved before each launch.
"""

import argparse
import itertools
import json
import subprocess
import sys
import threading
import warnings
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import yaml

# algo key -> (python module to invoke with `-m`, CLI flag that picks the
# network variant). Keys must match the YAML `matrix.algos` list.
_ALGO_SCRIPTS: Dict[str, Tuple[str, str]] = {
    "dqn": ("scripts.train_dqn", "--network"),
    "dqn_drq": ("scripts.train_dqn", "--network"),
    "dqn_curl": ("scripts.train_dqn", "--network"),
    "dqn_rad": ("scripts.train_dqn", "--network"),
    "sac": ("scripts.train_sac", "--encoder"),
    "sac_drq": ("scripts.train_sac_drq", "--encoder"),
    "sac_ferm": ("scripts.train_sac_ferm", "--encoder"),
    "sac_rad": ("scripts.train_sac_rad", "--encoder"),
}


class Cell(NamedTuple):
    """One training run. run_tag is the stable identifier used for
    resume markers and per-cell output directory naming."""

    algo: str
    model: str
    task: str
    backend: str
    seed: int
    total_steps: int
    per_job_vram_mib: int

    @property
    def run_tag(self) -> str:
        return f"{self.backend}_{self.task}_{self.algo}_{self.model}_s{self.seed}"


def load_spec(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


# Algos whose architectures assume a CNN encoder. The agent file itself
# rejects non-CNN at init time, but skipping invalid combos here keeps
# the expanded cell list honest (no dead entries to puzzle over in logs).
_CNN_ONLY_ALGOS = {
    "dqn_drq",
    "dqn_curl",
    "dqn_rad",
    "sac_drq",
    "sac_rad",
    "sac_ferm",
}


def expand_cells(spec: dict) -> List[Cell]:
    # Cartesian product of matrix axes, filtered on algo-model compatibility
    # (aug variants are CNN-only) and task-backend compatibility (spec lists
    # tasks per backend so MS3 only wires its own tasks). VRAM budget per
    # cell is looked up by "<backend>_<model>" in spec["per_job_vram_mib"];
    # missing keys fail loud before any launch.
    m = spec["matrix"]
    defaults = spec.get("defaults", {})
    vram_table = spec.get("per_job_vram_mib", {})
    total_steps = int(defaults.get("total_steps", 20000))

    # New (preferred) shape: tasks_per_backend: {backend: [task, ...]}.
    # Old shape (tasks + backends at top level) is still supported for
    # existing tests but emits a deprecation warning.
    if "tasks_per_backend" in m:
        backend_tasks = m["tasks_per_backend"]
    else:
        # Legacy: flat lists produce the full Cartesian (task, backend)
        # product, which for this repo means invalid MS3 tasks like
        # close_loop_block_pulling. Kept working for back-compat.
        warnings.warn(
            "matrix spec uses legacy flat 'tasks' + 'backends' lists. "
            "Prefer 'tasks_per_backend' to avoid invalid MS3 task names.",
            DeprecationWarning,
            stacklevel=2,
        )
        backend_tasks = {backend: m["tasks"] for backend in m["backends"]}

    cells: List[Cell] = []
    for algo, model, seed in itertools.product(m["algos"], m["models"], m["seeds"]):
        if algo not in _ALGO_SCRIPTS:
            raise KeyError(f"unknown algo {algo!r}. Known: {sorted(_ALGO_SCRIPTS)}")
        # Skip augmentation variants paired with a non-CNN backbone.
        if algo in _CNN_ONLY_ALGOS and model != "cnn":
            continue
        for backend, tasks in backend_tasks.items():
            vram_key = f"{backend}_{model}"
            if vram_key not in vram_table:
                raise KeyError(
                    f"missing per_job_vram_mib[{vram_key!r}] in spec. "
                    f"Known keys: {sorted(vram_table)}"
                )
            for task in tasks:
                cells.append(
                    Cell(
                        algo=algo,
                        model=model,
                        task=task,
                        backend=backend,
                        seed=int(seed),
                        total_steps=total_steps,
                        per_job_vram_mib=int(vram_table[vram_key]),
                    )
                )
    return cells


# DQN aug variants share train_dqn.py but select their variant via --network
# (not --network cnn). sac_ferm has no --encoder flag at all (CNN-hardcoded).
# Keeping the translation here so matrix.yaml stays declarative.
_DQN_AUG_NETWORK: Dict[str, str] = {
    "dqn_drq": "drq",
    "dqn_curl": "curl",
    "dqn_rad": "rad",
}


# Each backend runs under its own conda env. BulletArm needs equi_rl
# (Python 3.7 + pybullet + helping_hands_rl_envs); ManiSkill needs ms3_equi
# (Python 3.10 + gymnasium + maniskill3). build_command wraps every cell
# invocation in `conda run -n <env>` so one launcher can dispatch both.
_BACKEND_TO_CONDA_ENV: Dict[str, str] = {
    "bulletarm": "equi_rl",
    "maniskill": "ms3_equi",
}


def build_command(cell: Cell, output_dir: Path) -> List[str]:
    # Cell -> argv. `conda run -n <env>` wraps each invocation so each
    # backend runs under the right env (equi_rl for BulletArm, ms3_equi
    # for MS3). Flags are kebab-case (see utils/cli_args.py); --run-name
    # is cell.run_tag so the logger's output dir carries the tag.
    script, model_flag = _ALGO_SCRIPTS[cell.algo]
    env = _BACKEND_TO_CONDA_ENV[cell.backend]
    argv = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env,
        "python",
        "-m",
        script,
        "--env-name",
        cell.task,
        "--env-backend",
        cell.backend,
    ]

    # Variant selection differs per algo:
    #   - dqn (vanilla):       --network {cnn, equi}        (cell.model)
    #   - dqn_drq/curl/rad:    --network {drq, curl, rad}   (derived from algo)
    #   - sac (vanilla):       --encoder {cnn, equi}        (cell.model)
    #   - sac_drq/sac_rad:     --encoder {cnn, equi}        (cell.model; cnn per paper)
    #   - sac_ferm:            no flag (CNN hardcoded in train_sac_ferm.py)
    if cell.algo == "sac_ferm":
        pass  # no variant flag
    elif cell.algo in _DQN_AUG_NETWORK:
        argv += [model_flag, _DQN_AUG_NETWORK[cell.algo]]
    else:
        argv += [model_flag, cell.model]

    # Backend parallelism. BulletArm: MultiRunner with 5 workers (paper
    # spec; TrainConfig defaults to SingleRunner which is ~3x slower).
    # MS3: num_envs=32 for GPU-batched vectorization. eval_every=500 works
    # for both since global_step counts updates (no batch-size constraint).
    if cell.backend == "bulletarm":
        argv += ["--num-processes", "5"]
    elif cell.backend == "maniskill":
        argv += ["--num-envs", "32"]

    argv += [
        "--seed",
        str(cell.seed),
        "--total-steps",
        str(cell.total_steps),
        "--output-dir",
        str(output_dir),
        "--run-name",
        cell.run_tag,
    ]
    return argv


def query_free_vram_mib() -> int:
    # Minimum free VRAM across all visible GPUs. Conservative for
    # multi-GPU machines; fails loudly if nvidia-smi isn't on PATH.
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    values = [int(line.strip()) for line in out.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no memory.free values")
    return min(values)


class VRAMGate:
    """Thread-safe launch gate. wait_and_reserve(N) blocks until
    free-reserved >= N, then atomically reserves N MiB; release(N) frees
    it. The reservation counter covers the gap between Popen returning
    and CUDA actually allocating, which would otherwise race.
    """

    def __init__(self, vram_query: Callable[[], int] = query_free_vram_mib) -> None:
        self._lock = threading.Lock()
        self._reserved_mib = 0
        self._query = vram_query

    def reserved_mib(self) -> int:
        with self._lock:
            return self._reserved_mib

    def wait_and_reserve(
        self,
        required_mib: int,
        poll_interval: float = 30.0,
        max_wait: float = 3600.0,
    ) -> bool:
        # Returns True after reserving required_mib. Returns False if the
        # max_wait deadline expires with insufficient free+reserved.
        deadline = time.monotonic() + max_wait
        backoff = poll_interval
        while True:
            with self._lock:
                free = self._query()
                available = free - self._reserved_mib
                if available >= required_mib:
                    self._reserved_mib += required_mib
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
            # Exponential-ish backoff, capped so we don't wait forever
            # between polls on long deadlines.
            backoff = min(backoff * 1.5, 120.0)

    def release(self, mib: int) -> None:
        with self._lock:
            self._reserved_mib = max(0, self._reserved_mib - mib)


def marker_path(output_dir: Path, cell: Cell) -> Path:
    # Done markers live in a dedicated subdir so they survive even if
    # the cell's run directory is rotated or archived.
    return output_dir / ".markers" / f"{cell.run_tag}.done"


def is_cell_done(output_dir: Path, cell: Cell) -> bool:
    return marker_path(output_dir, cell).exists()


def mark_cell_done(output_dir: Path, cell: Cell) -> None:
    path = marker_path(output_dir, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def run_one(
    cell: Cell,
    output_dir: Path,
    gate: VRAMGate,
    dry_run: bool = False,
    vram_override: Optional[int] = None,
    poll_interval: float = 30.0,
    max_wait: float = 3600.0,
) -> Tuple[Cell, str, Optional[str]]:
    """Returns (cell, status, message). status in {ok, failed, timeout, dry_run}."""
    required_mib = vram_override if vram_override is not None else cell.per_job_vram_mib

    if dry_run:
        cmd = build_command(cell, output_dir)
        print(f"[dry-run] {cell.run_tag}: {' '.join(cmd)}")
        return cell, "dry_run", None

    acquired = gate.wait_and_reserve(
        required_mib, poll_interval=poll_interval, max_wait=max_wait
    )
    if not acquired:
        return cell, "timeout", f"VRAM gate timeout waiting for {required_mib} MiB"

    try:
        cmd = build_command(cell, output_dir)
        log_dir = output_dir / cell.run_tag
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "launch.log"
        print(f"[launch] {cell.run_tag} (need {required_mib} MiB)")
        with log_path.open("w") as log_f:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            mark_cell_done(output_dir, cell)
            return cell, "ok", None
        return (
            cell,
            "failed",
            f"exit={proc.returncode}; log={log_path}",
        )
    finally:
        gate.release(required_mib)


def run_matrix(
    cells: List[Cell],
    output_dir: Path,
    parallel: int = 1,
    dry_run: bool = False,
    vram_override: Optional[int] = None,
    gate: Optional[VRAMGate] = None,
    poll_interval: float = 30.0,
    max_wait: float = 3600.0,
) -> Dict[str, Tuple[str, Optional[str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = [c for c in cells if not is_cell_done(output_dir, c)]
    skipped = len(cells) - len(pending)
    print(
        f"[matrix] {len(cells)} cells total, {skipped} already done, "
        f"{len(pending)} pending"
    )

    if gate is None:
        gate = VRAMGate()

    results: Dict[str, Tuple[str, Optional[str]]] = {}

    def _record(cell: Cell, status: str, msg: Optional[str]) -> None:
        results[cell.run_tag] = (status, msg)

    if parallel <= 1:
        for cell in pending:
            _, status, msg = run_one(
                cell,
                output_dir,
                gate,
                dry_run=dry_run,
                vram_override=vram_override,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
            _record(cell, status, msg)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(
                    run_one,
                    c,
                    output_dir,
                    gate,
                    dry_run,
                    vram_override,
                    poll_interval,
                    max_wait,
                ): c
                for c in pending
            }
            for fut in as_completed(futures):
                cell, status, msg = fut.result()
                _record(cell, status, msg)

    _write_summary(output_dir, results)
    return results


def _write_summary(
    output_dir: Path, results: Dict[str, Tuple[str, Optional[str]]]
) -> None:
    counts: Dict[str, int] = {}
    for status, _msg in results.values():
        counts[status] = counts.get(status, 0) + 1
    print("\n[matrix] summary:")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")
    for tag, (status, msg) in sorted(results.items()):
        if status in ("failed", "timeout"):
            print(f"  {status}: {tag} / {msg}")

    summary_path = output_dir / "matrix_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "counts": counts,
                "results": {
                    k: {"status": v[0], "message": v[1]} for k, v in results.items()
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[matrix] summary at {summary_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the so2_equi_rl reproduction matrix"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to matrix.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matrix"),
        help="root for per-cell output dirs and done markers",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="concurrent workers; 1 = sequential (default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the launch command for each cell, don't launch",
    )
    parser.add_argument(
        "--vram-override",
        type=int,
        default=None,
        help="force per-job VRAM threshold (MiB); overrides spec",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between VRAM polls (default: 30)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=3600.0,
        help="seconds before VRAM gate gives up (default: 3600)",
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    cells = expand_cells(spec)
    results = run_matrix(
        cells=cells,
        output_dir=args.output_dir,
        parallel=args.parallel,
        dry_run=args.dry_run,
        vram_override=args.vram_override,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
    )

    # Non-zero exit if anything failed, so CI/scripting can detect it.
    if any(s in ("failed", "timeout") for s, _ in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
