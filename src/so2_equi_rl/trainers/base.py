"""Agent-agnostic trainer skeleton.

Owns env rollout, buffer writes, update scheduling, eval, and checkpointing.
Two abstract hooks (_warmup_action, _explore) cover the per-RL-family differences.
"""

import dataclasses
import math
import random
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from so2_equi_rl.agents.base import ActionPair, Agent
from so2_equi_rl.buffers.replay import ReplayBuffer
from so2_equi_rl.configs.base import TrainConfig
from so2_equi_rl.utils.logging import RunLogger

if TYPE_CHECKING:
    # Deferred so base.py imports without BulletArm; tests can pass mocks.
    from so2_equi_rl.envs.wrapper import EnvWrapper


class BaseTrainer(ABC):
    """Train/eval/checkpoint loop for any Agent subclass."""

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

        # Rollout state on self so _warmup + _train_loop don't pass it around.
        self._state: Optional[torch.Tensor] = None
        self._obs: Optional[torch.Tensor] = None

        self.global_step: int = 0
        self.best_success: float = -math.inf  # first eval wins best.pt

    @abstractmethod
    def _warmup_action(self, state: torch.Tensor, obs: torch.Tensor) -> ActionPair:
        """Action pushed into the buffer during warmup."""

    @abstractmethod
    def _explore(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        global_step: int,
    ) -> ActionPair:
        """Action for one rollout step. global_step is for DQN's epsilon decay."""

    def run(self, resume_path: Optional[Path] = None) -> None:
        try:
            if resume_path is not None:
                self._load(resume_path)

            # Fresh rollout state. On resume this drops the in-flight episode.
            self._state, self._obs = self.train_env.reset()

            # Warmup runs on empty buffer only. Resume from a buffer sidecar
            # skips warmup by virtue of len(buffer) > 0.
            if len(self.buffer) == 0:
                self._warmup()

            self._train_loop()

            # Final eval + full resume bundle (last.pt + buffer.pt) so a
            # follow-up run can resume without re-warming.
            final_metrics = self._evaluate()
            self.logger.log_scalars(
                final_metrics, step=self.global_step, to_stdout=True
            )
            self._save_policy("last")
            self.logger.save_checkpoint("buffer", self.buffer.state_dict())
        finally:
            self.logger.close()

    def _warmup(self) -> None:
        # Seed buffer via _warmup_action; no updates, no eval, no global_step advance.
        # Counts completed episodes (paper App F: SAC=20, DQN=100).
        target_episodes = int(self.cfg.warmup_episodes)
        completed_episodes = 0
        B = self.train_env.batch_size
        print(
            f"[warmup] collecting {target_episodes} expert episodes "
            f"across {B} parallel envs"
        )
        while completed_episodes < target_episodes:
            act = self._warmup_action(self._state, self._obs)
            step = self.train_env.step(act.physical)
            self.buffer.push(
                self._state,
                self._obs,
                act.unscaled,
                step.reward,
                step.state,
                step.obs,
                step.done,
            )
            self._state, self._obs = step.state, step.obs
            completed_episodes += int(step.done.sum().item())
        print(
            f"[warmup] done, collected {completed_episodes} expert episodes, "
            f"buffer size = {len(self.buffer)}"
        )

    def _evaluate(self) -> Dict[str, float]:
        # Deterministic rollouts on a separate EnvWrapper so training state isn't clobbered.
        # Paper uses discounted eval return, so we keep per-env reward traces
        # and fold with gamma on episode end.
        state, obs = self.eval_env.reset()
        Be = self.eval_env.batch_size
        gamma = self.cfg.gamma

        traces: List[List[float]] = [[] for _ in range(Be)]
        ep_len = torch.zeros(Be)
        # t-index of first step.success == 1 (None until it fires).
        # Used to compute the reward-mode-agnostic gamma^t metric.
        ep_success_step: List[Optional[int]] = [None] * Be
        disc_returns: List[float] = []
        disc_success_per_ep: List[float] = []
        lengths: List[float] = []
        successes: List[bool] = []

        while len(disc_returns) < self.cfg.eval_episodes:
            act = self.agent.select_action(state, obs, deterministic=True)
            step = self.eval_env.step(act.physical)
            ep_len += 1

            for i in range(Be):
                traces[i].append(float(step.reward[i].item()))
                # Record first-success step only; later ones ignored so gamma^t stays monotone.
                if (
                    step.success is not None
                    and step.success[i].item() > 0.5
                    and ep_success_step[i] is None
                ):
                    ep_success_step[i] = int(ep_len[i].item()) - 1
                if step.done[i].item() > 0.5:
                    # Reverse fold: R_t = r_t + gamma*R_{t+1}.
                    R = 0.0
                    for r in reversed(traces[i]):
                        R = r + gamma * R
                    disc_returns.append(R)
                    lengths.append(ep_len[i].item())
                    # Paper-style: gamma^t_success if solved, else 0.
                    # Comparable across sparse/dense runs and across backends.
                    t = ep_success_step[i]
                    disc_success_per_ep.append(gamma**t if t is not None else 0.0)
                    successes.append(t is not None)
                    traces[i] = []
                    ep_len[i] = 0.0
                    ep_success_step[i] = None
                    if len(disc_returns) >= self.cfg.eval_episodes:
                        break
            state, obs = step.state, step.obs

        # BulletArm sets step.success from reward>0; MS3 reads info['success'].
        success_rate = float(np.mean([1.0 if s else 0.0 for s in successes]))
        return {
            "eval/return_disc_mean": float(np.mean(disc_returns)),
            "eval/disc_success_mean": float(np.mean(disc_success_per_ep)),
            "eval/success_rate": success_rate,
            "eval/length_mean": float(np.mean(lengths)),
        }

    def _save_policy(self, name: str) -> None:
        # Policy-only payload (MB-scale); buffer goes in buffer.pt at run end.
        payload: Dict[str, Any] = {
            "global_step": self.global_step,
            "best_success": self.best_success,
            "agent": self.agent.state_dict(),
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
        self.logger.save_checkpoint(name, payload)

    def _load(self, path: Path) -> None:
        # map_location='cpu' so cuda-saved ckpts load on cpu-only too.
        payload = torch.load(path, map_location="cpu")

        # Soft cfg diff: warn, don't raise, so intentional changes aren't blocked.
        current_cfg = dataclasses.asdict(self.cfg)
        saved_cfg = payload.get("cfg_snapshot", {})
        diffs = {
            k: (saved_cfg.get(k), current_cfg.get(k))
            for k in set(saved_cfg) | set(current_cfg)
            if saved_cfg.get(k) != current_cfg.get(k)
        }
        if diffs:
            warnings.warn(
                f"BaseTrainer._load: cfg differs from checkpoint (saved vs current): {diffs}",
                stacklevel=2,
            )

        self.agent.load_state_dict(payload["agent"])

        # Buffer sidecar is only written at run end; cadenced ckpts won't have one.
        buffer_path = Path(path).parent / "buffer.pt"
        if buffer_path.exists():
            buffer_state = torch.load(buffer_path, map_location="cpu")
            self.buffer.load_state_dict(buffer_state)
        else:
            warnings.warn(
                f"BaseTrainer._load: no buffer sidecar at {buffer_path}. "
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
        # 100-ep rolling window matches Wang et al. utils/logger.py:107.
        # global_step counts update iters (paper main.py:46 logger.num_training_steps).
        # One iter = one env.step + n_updates_per_step gradient updates.
        cfg = self.cfg
        B = self.train_env.batch_size
        step_per_iter = int(cfg.n_updates_per_step)

        # Cadences must divide step_per_iter to fire inside the `% cadence < step_per_iter` window.
        for cadence_name, cadence in (
            ("log_every", cfg.log_every),
            ("eval_every", cfg.eval_every),
            ("ckpt_every", cfg.ckpt_every),
        ):
            if cadence % step_per_iter != 0:
                raise ValueError(
                    f"{cadence_name}={cadence} must be a multiple of "
                    f"n_updates_per_step={step_per_iter}"
                )

        ep_return = torch.zeros(B)
        ep_len = torch.zeros(B)
        recent_returns: deque = deque(maxlen=100)
        recent_successes: deque = deque(maxlen=100)
        loss_accum: Dict[str, list] = defaultdict(list)

        # SPS anchor resets every log so SPS covers the last log_every window.
        sps_anchor_step = self.global_step
        sps_anchor_time = time.time()

        print(
            f"[train] starting loop at step={self.global_step}, target={cfg.total_steps}"
        )

        while self.global_step < cfg.total_steps:
            # _explore owns the exploration decision (SAC stochastic, DQN eps-greedy).
            act = self._explore(self._state, self._obs, self.global_step)
            step = self.train_env.step(act.physical)

            self.buffer.push(
                self._state,
                self._obs,
                act.unscaled,  # buffer stores unscaled
                step.reward,
                step.state,
                step.obs,
                step.done,
            )

            ep_return += step.reward
            ep_len += 1

            # Episode bookkeeping. return > 0 = success on sparse tasks.
            for i in range(B):
                if step.done[i].item() > 0.5:
                    r = ep_return[i].item()
                    recent_returns.append(r)
                    recent_successes.append(1.0 if r > 0 else 0.0)
                    ep_return[i] = 0.0
                    ep_len[i] = 0.0

            # Gradient updates (UTD = n_updates_per_step).
            for _ in range(cfg.n_updates_per_step):
                batch = self.buffer.sample(cfg.batch_size)
                metrics = self.agent.update(batch)
                for k, v in metrics.items():
                    loss_accum[k].append(v)

            # Advance by update count to match paper's training-step counter.
            self._state, self._obs = step.state, step.obs
            self.global_step += step_per_iter

            # Cadenced log/eval/ckpt. `< step_per_iter` catches the single boundary crossing.
            if self.global_step % cfg.log_every < step_per_iter:
                now = time.time()
                window_steps = self.global_step - sps_anchor_step
                window_secs = max(now - sps_anchor_time, 1e-9)
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

            if self.global_step % cfg.eval_every < step_per_iter:
                eval_metrics = self._evaluate()
                self.logger.log_scalars(eval_metrics, step=self.global_step)
                if eval_metrics["eval/success_rate"] > self.best_success:
                    self.best_success = eval_metrics["eval/success_rate"]
                    self._save_policy("best")

            if self.global_step % cfg.ckpt_every < step_per_iter:
                self._save_policy("last")
