"""Run-directory logging: TB + JSONL metrics, git hash, config dump.

One RunLogger per run. Owns a timestamped dir under cfg.output_dir and is the
single sink for metrics, checkpoints, and provenance.
"""

import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

        # TB for live curves, JSONL for offline diffing. Each line is its own
        # dict, so new metric keys (e.g. eval/* appearing mid-run) just start
        # showing up in later rows without a schema rewrite.
        self.tb = SummaryWriter(log_dir=str(self.run_dir))
        self._jsonl_file = (self.run_dir / "metrics.jsonl").open("w", buffering=1)

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

        # JSONL: append one standalone row per call. Float-cast so numpy
        # scalars / 0-d tensors serialize cleanly.
        row = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self._jsonl_file.write(json.dumps(row) + "\n")

        if to_stdout:
            pretty = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"[step {step}] {pretty}")

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
        self._jsonl_file.flush()
        self._jsonl_file.close()
        self._closed = True
