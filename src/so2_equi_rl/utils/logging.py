"""Run-directory logging: TB + CSV mirror, stdout tee, git hash, config dump.

One RunLogger per run. Owns a timestamped dir under cfg.output_dir and is the
single sink for metrics, checkpoints, and provenance. Trainer-agnostic; SAC
and DQN instantiate it the same way.
"""

import csv
import dataclasses
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from so2_equi_rl.configs.base import TrainConfig


def _git_info(repo_root: Path) -> str:
    # Returns "<sha>", "<sha>-dirty", or "no-git". Never raises:
    # a missing .git or missing git binary shouldn't kill a run.
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


class _Tee:
    """Duplicates writes to multiple streams. Mirrors stdout into stdout.log
    without losing the terminal. Only .write and .flush are forwarded.
    """

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


class RunLogger:
    """Timestamped run directory + metric sink + checkpoint saver."""

    def __init__(self, cfg: TrainConfig, run_name: Optional[str] = None) -> None:
        # Build the run dir name. Timestamp first so `ls outputs/` sorts chronologically.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [stamp, cfg.env_name]
        if run_name:
            parts.append(run_name)
        self.run_dir: Path = Path(cfg.output_dir) / "_".join(parts)
        self.ckpt_dir: Path = self.run_dir / "ckpts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)

        # Provenance: config snapshot + git state. Written once at construction.
        cfg_path = self.run_dir / "config.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
        (self.run_dir / "git.txt").write_text(_git_info(Path.cwd()) + "\n")

        # TB for live curves, CSV for offline diffing. CSV uses a growing
        # schema: new keys (eval/* typically shows up mid-run) rewrite the
        # file with the wider header so it stays rectangular.
        self.tb = SummaryWriter(log_dir=str(self.run_dir))
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_columns: List[str] = []  # grows as new metric keys show up
        self._csv_rows: List[Dict[str, Any]] = []  # cached for schema-growth rewrites
        self._csv_file = self._csv_path.open(
            "w", newline="", buffering=1
        )  # line-buffered
        self._csv_writer = csv.writer(self._csv_file)

        # Stdout tee. Stash original so close() can restore and tests don't
        # accumulate a chain of tees across multiple RunLoggers.
        self._stdout_log = (self.run_dir / "stdout.log").open("w", buffering=1)
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._stdout_log)

        self._closed = False

    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
        to_stdout: bool = False,
    ) -> None:
        # TB is schema-free, so one add_scalar per key and we're done.
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, step)

        # CSV: cache every row as a dict so we can rewrite the file from
        # scratch whenever a new metric key arrives.
        row = {"step": step, **metrics}
        self._csv_rows.append(row)

        new_keys = [k for k in metrics if k not in self._csv_columns]
        if new_keys:
            self._csv_columns = ["step"] + sorted(
                set(self._csv_columns[1:]) | set(metrics.keys())
            )
            self._rewrite_csv()
        else:
            self._csv_writer.writerow([row.get(col, "") for col in self._csv_columns])

        if to_stdout:
            pretty = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"[step {step}] {pretty}")

    def _rewrite_csv(self) -> None:
        # Reopen the file and re-emit every cached row under the current
        # (wider) column list. Only runs when the schema grows.
        self._csv_file.close()
        self._csv_file = self._csv_path.open("w", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._csv_columns)
        for cached in self._csv_rows:
            self._csv_writer.writerow(
                [cached.get(col, "") for col in self._csv_columns]
            )

    def save_checkpoint(self, name: str, payload: Dict[str, Any]) -> Path:
        # Atomic save: write to .pt.tmp then rename. A SIGINT mid-save
        # leaves the previous good checkpoint intact instead of a truncated file.
        path = self.ckpt_dir / f"{name}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def close(self) -> None:
        if self._closed:
            return
        self.tb.flush()
        self.tb.close()
        self._csv_file.flush()
        self._csv_file.close()
        sys.stdout = self._orig_stdout  # restore before closing the log file
        self._stdout_log.close()
        self._closed = True
