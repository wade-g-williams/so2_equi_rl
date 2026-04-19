"""DQN with Polyak target updates and a 4-axis discrete action decoder.

net_cls is injected so one update() body covers the equivariant run and
the CNN baseline. Buffer stores 4-D grid indices as float32, decode_action
runs at the env.step boundary to convert back to 5-D physical (pxyzr).
"""

import copy
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from so2_equi_rl.agents.base import Agent, ActionPair
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.dqn import DQNConfig
from so2_equi_rl.utils import tile_state


class DQNAgent(Agent):
    """DQN with Polyak target updates and a discrete pxyzr action decoder."""

    def __init__(
        self,
        cfg: DQNConfig,
        net_cls: Type[nn.Module],
    ) -> None:
        self._init_hyperparams(cfg)
        self._init_action_grid(cfg)

        net_kwargs = {
            "obs_channels": cfg.obs_channels,
            "n_hidden": cfg.n_hidden,
            "group_order": cfg.group_order,  # ignored by CNNDQNNet
            "n_p": cfg.n_p,
            "n_xy": cfg.n_xy,
            "n_z": cfg.n_z,
            "n_theta": cfg.n_theta,
        }
        self.policy_net = net_cls(**net_kwargs).to(self.device)
        # deepcopy after .to() so e2cnn doesn't rebuild the kernel basis on the wrong device.
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.requires_grad_(False)

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

    def _init_hyperparams(self, cfg: DQNConfig) -> None:
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.action_dim = cfg.action_dim
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.grad_clip_norm = cfg.grad_clip_norm
        self.n_p = cfg.n_p
        self.n_xy = cfg.n_xy
        self.n_z = cfg.n_z
        self.n_theta = cfg.n_theta

    def _init_action_grid(self, cfg: DQNConfig) -> None:
        # Lookup tables held on CPU. encode/decode move them to the input
        # device per call, which is free for tensors this small.
        self._p_table = torch.tensor(
            [cfg.p_range[0], cfg.p_range[1]], dtype=torch.float32
        )  # (n_p,)
        # xy ordering matches Wang et al.'s getActionFromPlan: dx slow, dy fast.
        # Row-major flatten of (3, 3) -> 9 with row=dx_idx, col=dy_idx.
        dpos = cfg.dpos
        self._xy_table = torch.tensor(
            [
                [-dpos, -dpos],
                [-dpos, 0.0],
                [-dpos, dpos],
                [0.0, -dpos],
                [0.0, 0.0],
                [0.0, dpos],
                [dpos, -dpos],
                [dpos, 0.0],
                [dpos, dpos],
            ],
            dtype=torch.float32,
        )  # (n_xy, 2)
        self._z_table = torch.tensor(
            [-cfg.dz, 0.0, cfg.dz], dtype=torch.float32
        )  # (n_z,)
        self._theta_table = torch.tensor(
            [-cfg.drot, 0.0, cfg.drot], dtype=torch.float32
        )  # (n_theta,)

    def decode_action(self, indices: Tensor) -> Tensor:
        # indices: (B, 4) long, columns (p_id, xy_id, z_id, theta_id).
        # Returns (B, 5) float [p, dx, dy, dz, dtheta] on the input device.
        device = indices.device
        idx = indices.long()
        p = self._p_table.to(device)[idx[:, 0]].unsqueeze(-1)  # (B, 1)
        dxy = self._xy_table.to(device)[idx[:, 1]]  # (B, 2)
        dz = self._z_table.to(device)[idx[:, 2]].unsqueeze(-1)  # (B, 1)
        dtheta = self._theta_table.to(device)[idx[:, 3]].unsqueeze(-1)  # (B, 1)
        return torch.cat([p, dxy, dz, dtheta], dim=1)

    def encode_action(self, physical: Tensor) -> Tensor:
        # physical: (B, 5) float [p, dx, dy, dz, dtheta].
        # Returns (B, 4) long indices via argmin distance per axis. Mirrors
        # the paper repo's getActionFromPlan quantization.
        device = physical.device
        p_table = self._p_table.to(device)
        xy_table = self._xy_table.to(device)
        z_table = self._z_table.to(device)
        theta_table = self._theta_table.to(device)

        p_id = (physical[:, 0:1] - p_table.unsqueeze(0)).abs().argmin(dim=1)
        # Joint argmin over (dx, dy) so off-grid commands snap to the nearest cell.
        dxy = physical[:, 1:3]
        dist_xy = (dxy.unsqueeze(1) - xy_table.unsqueeze(0)).pow(2).sum(dim=-1)
        xy_id = dist_xy.argmin(dim=1)
        z_id = (physical[:, 3:4] - z_table.unsqueeze(0)).abs().argmin(dim=1)
        theta_id = (physical[:, 4:5] - theta_table.unsqueeze(0)).abs().argmin(dim=1)
        return torch.stack([p_id, xy_id, z_id, theta_id], dim=1)

    def select_action(
        self,
        state: Tensor,
        obs: Tensor,
        eps: float = 0.0,
        deterministic: bool = False,
    ) -> ActionPair:
        # deterministic=True forces eps=0 (eval rollouts use pure greedy).
        if deterministic:
            eps = 0.0

        with torch.no_grad():
            state = state.to(self.device)
            obs = obs.to(self.device)
            tiled = tile_state(obs, state)
            q_all = self.policy_net(tiled)  # (B, n_xy, n_z, n_theta, n_p)
            B = q_all.shape[0]

            # Greedy: argmax over flattened action space, then unravel.
            flat_argmax = q_all.reshape(B, -1).argmax(dim=1)
            indices = self._unravel(flat_argmax)  # (B, 4) long

            if eps > 0.0:
                # Per-row Bernoulli mask, replace masked rows with uniform random per axis.
                mask = torch.bernoulli(torch.full((B,), eps, device=self.device)).bool()
                if mask.any():
                    rand = torch.stack(
                        [
                            torch.randint(0, self.n_p, (B,), device=self.device),
                            torch.randint(0, self.n_xy, (B,), device=self.device),
                            torch.randint(0, self.n_z, (B,), device=self.device),
                            torch.randint(0, self.n_theta, (B,), device=self.device),
                        ],
                        dim=1,
                    )
                    indices = torch.where(mask.unsqueeze(1), rand, indices)

            physical = self.decode_action(indices)

        # Buffer stores indices as float32, env.step gets physical pxyzr.
        return ActionPair(
            unscaled=indices.float().cpu(),
            physical=physical.cpu(),
        )

    def _unravel(self, flat_idx: Tensor) -> Tensor:
        # Inverse of view(B, -1) on (B, n_xy, n_z, n_theta, n_p). p is fastest.
        p_id = flat_idx % self.n_p
        theta_id = (flat_idx // self.n_p) % self.n_theta
        z_id = (flat_idx // (self.n_p * self.n_theta)) % self.n_z
        xy_id = flat_idx // (self.n_p * self.n_theta * self.n_z)
        return torch.stack([p_id, xy_id, z_id, theta_id], dim=1)

    def update(self, batch: Transition) -> Dict[str, float]:
        batch = batch.to(self.device, non_blocking=True)

        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        obs_tiled = tile_state(batch.obs, batch.state)
        next_obs_tiled = tile_state(batch.next_obs, batch.next_state)

        # Buffer stored indices as float32, gather needs long.
        action_idx = batch.action.long()
        p_id = action_idx[:, 0]
        xy_id = action_idx[:, 1]
        z_id = action_idx[:, 2]
        theta_id = action_idx[:, 3]
        B = action_idx.shape[0]
        b_arange = torch.arange(B, device=self.device)

        q_all = self.policy_net(obs_tiled)  # (B, n_xy, n_z, n_theta, n_p)
        q_pred = q_all[b_arange, xy_id, z_id, theta_id, p_id].unsqueeze(-1)  # (B, 1)

        # Bellman target. Greedy over the full flattened action space.
        with torch.no_grad():
            q_all_next = self.target_net(next_obs_tiled)
            q_next = q_all_next.reshape(B, -1).max(dim=1)[0].unsqueeze(-1)  # (B, 1)
            y = reward + self.gamma * (1.0 - done) * q_next

        td_loss = F.smooth_l1_loss(q_pred, y)
        td_error = (q_pred - y).detach()

        self.optim.zero_grad()
        td_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optim.step()

        # Polyak target update.
        with torch.no_grad():
            for p, p_target in zip(
                self.policy_net.parameters(), self.target_net.parameters()
            ):
                p_target.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "td_loss": td_loss.item(),
            "td_error_mean": td_error.mean().item(),
            "q_mean": q_pred.mean().item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        for m in (self.policy_net, self.target_net):
            self._unalias_buffers(m)
        self.policy_net.load_state_dict(d["policy_net"])
        self.target_net.load_state_dict(d["target_net"])
        self.optim.load_state_dict(d["optim"])
