"""Agent-agnostic trainer. Owns the env rollout, buffer writes, update
scheduling, eval, and checkpointing. Trainer talks to the Agent contract
(select_action / update / state_dict) and to EnvWrapper, ReplayBuffer,
RunLogger. Nothing SAC-specific lives here, so the DQN baseline reuses
the same class.
"""

import dataclasses
import math
import random
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from so2_equi_rl.agents.base import Agent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.base import TrainConfig
from so2_equi_rl.utils.logging import RunLogger

if TYPE_CHECKING:
    # Deferred so trainer.py imports cleanly without BulletArm + lets tests
    # pass in env mocks. EnvWrapper pulls in helping_hands_rl_envs which has
    # a load-time bug we patch in envs/__init__.py.
    from so2_equi_rl.envs.wrapper import EnvWrapper


class Trainer:
    """Drives the train/eval/ckpt loop for any Agent subclass."""

    def __init__(
        self,
        cfg: TrainConfig,
        agent: Agent,
        train_env: "EnvWrapper",
        eval_env: "EnvWrapper",
        buffer: ReplayBuffer,
        logger: RunLogger,
    ) -> None:
        self.cfg = cfg
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        # Rollout state on self so _warmup / _train_loop don't pass it around.
        self._state: Optional[torch.Tensor] = None
        self._obs: Optional[torch.Tensor] = None

        self.global_step: int = 0
        self.best_success: float = -math.inf  # first eval always wins best.pt

    def run(self, resume_path: Optional[Path] = None) -> None:
        try:
            if resume_path is not None:
                self._load(resume_path)

            # Fresh rollout state. On resume this drops the in-flight episode
            # the ckpt was captured in; training resumes from a new episode.
            self._state, self._obs = self.train_env.reset()

            # Warmup triggers on buffer length, not global_step, so a resume
            # with save_buffer_on_ckpt=False still refills before training.
            if len(self.buffer) < self.cfg.warmup_steps:
                self._warmup()

            self._train_loop()

            # Final eval + last ckpt so the end-of-run artifacts always exist.
            final_metrics = self._evaluate()
            self.logger.log_scalars(
                final_metrics, step=self.global_step, to_stdout=True
            )
            self._save("last")
        finally:
            self.logger.close()

    def _warmup(self) -> None:
        # Scripted-expert demos into the buffer. No updates, no eval, no
        # global_step advance; matches Wang et al. BulletArm auto-resets
        # on done so step.state/obs is already the next episode.
        print(f"[warmup] filling buffer to {self.cfg.warmup_steps} transitions")
        while len(self.buffer) < self.cfg.warmup_steps:
            physical = self.train_env.get_expert_action()
            unscaled = self.agent.encode_action(physical)
            step = self.train_env.step(physical)
            self.buffer.push(
                self._state,
                self._obs,
                unscaled,
                step.reward,
                step.state,
                step.obs,
                step.done,
            )
            self._state, self._obs = step.state, step.obs
        print(f"[warmup] done, buffer size = {len(self.buffer)}")

    def _evaluate(self) -> Dict[str, float]:
        # Deterministic rollouts on a separate EnvWrapper (different seed).
        # Uses its own state/obs so training rollout progress isn't clobbered.
        state, obs = self.eval_env.reset()
        Be = self.eval_env.batch_size

        ep_return = torch.zeros(Be)
        ep_len = torch.zeros(Be)
        returns: List[float] = []
        lengths: List[float] = []

        while len(returns) < self.cfg.eval_episodes:
            act = self.agent.select_action(state, obs, deterministic=True)
            step = self.eval_env.step(act.physical)
            ep_return += step.reward
            ep_len += 1

            # Close out any episodes that just finished. BulletArm auto-resets,
            # so the next iteration's state/obs for that slot is already fresh.
            for i in range(Be):
                if step.done[i].item() > 0.5:
                    returns.append(ep_return[i].item())
                    lengths.append(ep_len[i].item())
                    ep_return[i] = 0.0
                    ep_len[i] = 0.0
            state, obs = step.state, step.obs

        # Sparse-reward close-loop tasks: return > 0 <=> task completed.
        success_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in returns]))
        return {
            "eval/return_mean": float(np.mean(returns)),
            "eval/success_rate": success_rate,
            "eval/length_mean": float(np.mean(lengths)),
        }

    def _save(self, name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        # Full resume payload: agent state + optional buffer + RNGs + bookkeeping.
        buffer_state = (
            self.buffer.state_dict() if self.cfg.save_buffer_on_ckpt else None
        )
        payload: Dict[str, Any] = {
            "global_step": self.global_step,
            "best_success": self.best_success,
            "agent": self.agent.state_dict(),
            "buffer": buffer_state,
            "rng": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "cfg_snapshot": dataclasses.asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        self.logger.save_checkpoint(name, payload)

    def _load(self, path: Path) -> None:
        # map_location='cpu' so a checkpoint saved on a cuda box still loads
        # on a cpu-only machine; the agent re-moves tensors to its device.
        payload = torch.load(path, map_location="cpu")

        # Soft cfg diff: warn, don't raise. Dangerous shape mismatches get
        # caught by ReplayBuffer.load_state_dict anyway.
        current_cfg = dataclasses.asdict(self.cfg)
        saved_cfg = payload.get("cfg_snapshot", {})
        diffs = {
            k: (saved_cfg.get(k), current_cfg.get(k))
            for k in set(saved_cfg) | set(current_cfg)
            if saved_cfg.get(k) != current_cfg.get(k)
        }
        if diffs:
            warnings.warn(
                f"Trainer._load: cfg differs from checkpoint on fields {sorted(diffs)}",
                stacklevel=2,
            )

        self.agent.load_state_dict(payload["agent"])

        if payload.get("buffer") is not None:
            self.buffer.load_state_dict(payload["buffer"])
        else:
            warnings.warn(
                "Trainer._load: checkpoint has no buffer (save_buffer_on_ckpt was False). "
                "Warmup will refill before training resumes.",
                stacklevel=2,
            )

        rng = payload["rng"]
        torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

        self.global_step = int(payload["global_step"])
        self.best_success = float(payload["best_success"])

    def _train_loop(self) -> None:
        # 100-episode rolling window, matching Wang et al. (utils/logger.py:107).
        # Warmup episodes are NOT seeded in: expert returns would inflate the
        # early curve and hide the real policy's learning.
        cfg = self.cfg
        B = self.train_env.batch_size

        # The `% cadence < B` fire trick is only deterministic when B divides
        # each cadence. Enforce up front instead of drifting silently.
        for cadence_name, cadence in (
            ("log_every", cfg.log_every),
            ("eval_every", cfg.eval_every),
            ("ckpt_every", cfg.ckpt_every),
        ):
            if cadence % B != 0:
                raise ValueError(
                    f"{cadence_name}={cadence} must be a multiple of batch size B={B}"
                )

        ep_return = torch.zeros(B)
        ep_len = torch.zeros(B)
        recent_returns: deque = deque(maxlen=100)
        recent_successes: deque = deque(maxlen=100)
        loss_accum: Dict[str, list] = defaultdict(list)

        # SPS window anchor. Reset on every log so SPS is the rate over the
        # last log_every window, not since run start.
        sps_anchor_step = self.global_step
        sps_anchor_time = time.time()

        print(
            f"[train] starting loop at step={self.global_step}, target={cfg.total_steps}"
        )

        while self.global_step < cfg.total_steps:
            # --- env step on the current policy (exploration on) ---
            act = self.agent.select_action(self._state, self._obs, deterministic=False)
            step = self.train_env.step(act.physical)

            self.buffer.push(
                self._state,
                self._obs,
                act.unscaled,  # buffer stores [-1, 1], not physical
                step.reward,
                step.state,
                step.obs,
                step.done,
            )

            ep_return += step.reward
            ep_len += 1

            # Episode bookkeeping. return > 0 <=> success on sparse tasks.
            for i in range(B):
                if step.done[i].item() > 0.5:
                    r = ep_return[i].item()
                    recent_returns.append(r)
                    recent_successes.append(1.0 if r > 0 else 0.0)
                    ep_return[i] = 0.0
                    ep_len[i] = 0.0

            # --- gradient updates (UTD = n_updates_per_step) ---
            for _ in range(cfg.n_updates_per_step):
                batch = self.buffer.sample(cfg.batch_size)
                metrics = self.agent.update(batch)
                for k, v in metrics.items():
                    loss_accum[k].append(v)

            # Advance rollout state and step counter. Overshoot by up to B-1
            # on the final iter is accepted; matches Wang et al. main.py:270.
            self._state, self._obs = step.state, step.obs
            self.global_step += B

            # --- cadenced logging / eval / ckpt using the `% cadence < B` trick ---
            if self.global_step % cfg.log_every < B:
                now = time.time()
                window_steps = self.global_step - sps_anchor_step
                window_secs = max(now - sps_anchor_time, 1e-9)  # guard div-by-zero
                sps = window_steps / window_secs

                log: Dict[str, float] = {
                    f"loss/{k}": float(np.mean(vs)) for k, vs in loss_accum.items()
                }
                log["train/return_mean"] = (
                    float(np.mean(recent_returns)) if recent_returns else 0.0
                )
                log["train/success_rate"] = (
                    float(np.mean(recent_successes)) if recent_successes else 0.0
                )
                log["train/buffer_size"] = float(len(self.buffer))
                log["time/sps"] = sps
                self.logger.log_scalars(log, step=self.global_step, to_stdout=True)

                loss_accum.clear()
                sps_anchor_step = self.global_step
                sps_anchor_time = now

            if self.global_step % cfg.eval_every < B:
                eval_metrics = self._evaluate()
                self.logger.log_scalars(eval_metrics, step=self.global_step)
                if eval_metrics["eval/success_rate"] > self.best_success:
                    self.best_success = eval_metrics["eval/success_rate"]
                    self._save("best")

            if self.global_step % cfg.ckpt_every < B:
                self._save("last")
