"""CURL-DQN. Adds a CURL InfoNCE loss over two random-crop views, sharing
the Q-net's conv stack with the contrastive head (pad=7, 142 -> 128).

The conv stack trains from TD (self.optim) and InfoNCE (self.encoder_optim).
Both run per update with distinct Adam states so the two gradient scales
stay decoupled. Momentum key-net is a frozen Polyak copy of policy_net at
cfg.curl_tau; since the key side isn't bootstrapped, curl_tau can run
faster than cfg.tau without destabilising the TD target.

CNN-only (same reason as sac_ferm.py): the bilinear head expects a flat
feature vector and EquiDQNNet's structured output doesn't plug in cleanly.
"""

import copy
from typing import Any, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_equi_rl.agents.dqn import DQNAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs.dqn_curl import DQNCURLConfig
from so2_equi_rl.networks import CNNDQNNet
from so2_equi_rl.utils import augmentation as aug_mod
from so2_equi_rl.utils import tile_state

# Fixed offset so the CURL aug RNG doesn't overlap DQN-RAD (2023),
# SAC-DrQ (1337), RAD-SAC (2022), FERM (3407), or DQN-DrQ (4242).
_AUG_SEED_OFFSET = 5150


class DQNCURLAgent(DQNAgent):
    """DQN with a CURL InfoNCE loss sharing the policy_net conv stack."""

    def __init__(
        self,
        cfg: DQNCURLConfig,
        net_cls: Type[nn.Module],
    ) -> None:
        # CNN-only. InfoNCE expects flat features; EquiDQNNet's structured
        # output would need a GroupPooling + flatten that the paper's CURL
        # baseline doesn't use. Fail fast instead of silently misbehaving.
        if net_cls is not CNNDQNNet:
            raise TypeError(f"DQNCURLAgent is CNN-only; got net_cls={net_cls.__name__}")

        super().__init__(cfg, net_cls)

        self.curl_tau = float(cfg.curl_tau)
        self.curl_lambda = float(cfg.curl_lambda)
        self.curl_temperature = float(cfg.curl_temperature)
        self.curl_pad = int(cfg.curl_pad)

        # Momentum key-net. Full net deepcopy (not just .conv) keeps the
        # checkpoint format uniform and avoids the buffer-aliasing pitfalls
        # that bit SAC-DrQ's state_dict path. Only .features() is ever
        # called on it, so the fc head is dead weight but cheap.
        self.k_net = copy.deepcopy(self.policy_net)
        for p in self.k_net.parameters():
            p.requires_grad_(False)

        # Bilinear projection. logits[i, j] = <q_i, W @ k_j>. Orthogonal
        # init keeps initial logits O(1) so cross-entropy doesn't saturate.
        self.feat_dim = int(self.policy_net.feat_dim)
        self.W = nn.Parameter(
            torch.empty(self.feat_dim, self.feat_dim, device=self.device)
        )
        nn.init.orthogonal_(self.W)

        # Separate encoder_optim over the conv stack + W. self.optim already
        # owns all policy_net params (conv included); both optimizers touch
        # the same conv tensors but keep distinct Adam moments.
        self.encoder_optim = torch.optim.Adam(
            list(self.policy_net.conv.parameters()) + [self.W],
            lr=cfg.encoder_lr,
        )

        # CPU generator. random_crop uses torch.randint which needs cpu.
        self._aug_gen = torch.Generator(device="cpu")
        self._aug_gen.manual_seed(int(cfg.seed) + _AUG_SEED_OFFSET)

    def update(self, batch: Transition) -> Dict[str, float]:
        # TD loss, then InfoNCE, then two Polyak averages.
        batch = batch.to(self.device, non_blocking=True)

        reward = batch.reward
        done = batch.done
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        obs_tiled = tile_state(batch.obs, batch.state)
        next_obs_tiled = tile_state(batch.next_obs, batch.next_state)

        # TD step. Identical to DQNAgent.update's body; encoder grads from
        # TD flow through self.optim, InfoNCE will add a second pass below.
        action_idx = batch.action.long()
        p_id = action_idx[:, 0]
        xy_id = action_idx[:, 1]
        z_id = action_idx[:, 2]
        theta_id = action_idx[:, 3]
        B = action_idx.shape[0]
        b_arange = torch.arange(B, device=self.device)

        q_all = self.policy_net(obs_tiled)  # (B, n_xy, n_z, n_theta, n_p)
        q_pred = q_all[b_arange, xy_id, z_id, theta_id, p_id].unsqueeze(-1)

        with torch.no_grad():
            q_all_next = self.target_net(next_obs_tiled)
            q_next = q_all_next.reshape(B, -1).max(dim=1)[0].unsqueeze(-1)
            y = reward + self.gamma * (1.0 - done) * q_next

        td_loss = F.smooth_l1_loss(q_pred, y)
        td_error = (q_pred - y).detach()

        self.optim.zero_grad()
        td_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optim.step()

        # InfoNCE step. Paper sec E CURL: two independent random crops (142 ->
        # 128) for the query and key. Pulls (q, k) from the same row together
        # and pushes other pairings apart.
        obs_cpu = batch.obs.cpu()
        q_obs = aug_mod.random_crop(
            obs_cpu, pad=self.curl_pad, generator=self._aug_gen
        ).to(self.device)
        k_obs = aug_mod.random_crop(
            obs_cpu, pad=self.curl_pad, generator=self._aug_gen
        ).to(self.device)
        q_obs_tiled = tile_state(q_obs, batch.state)
        k_obs_tiled = tile_state(k_obs, batch.state)

        # Key side is frozen via requires_grad=False; no_grad also drops key activations.
        q_feat = self.policy_net.features(q_obs_tiled)  # (B, feat_dim)
        with torch.no_grad():
            k_feat = self.k_net.features(k_obs_tiled)  # (B, feat_dim)

        # Bilinear logits (B, B). Diagonal = positives, off-diagonal = negatives.
        # Row-max subtract for numerical stability, doesn't change cross-entropy.
        Wk = torch.matmul(self.W, k_feat.t())  # (feat_dim, B)
        logits = torch.matmul(q_feat, Wk) / self.curl_temperature  # (B, B)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = torch.arange(B, device=self.device)
        infonce_loss = F.cross_entropy(logits, labels)

        # zero_grad clears stale conv grads from the TD backward earlier.
        self.encoder_optim.zero_grad()
        (self.curl_lambda * infonce_loss).backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.policy_net.conv.parameters()) + [self.W],
                self.grad_clip_norm,
            )
        self.encoder_optim.step()

        # Two Polyak updates. target_net at cfg.tau (slow), k_net at curl_tau (fast).
        with torch.no_grad():
            for p, p_target in zip(
                self.policy_net.parameters(), self.target_net.parameters()
            ):
                p_target.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

            for p, p_k in zip(self.policy_net.parameters(), self.k_net.parameters()):
                p_k.mul_(1.0 - self.curl_tau).add_(p.data, alpha=self.curl_tau)

        return {
            "td_loss": td_loss.item(),
            "td_error_mean": td_error.mean().item(),
            "q_mean": q_pred.mean().item(),
            "infonce_loss": infonce_loss.item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "k_net": self.k_net.state_dict(),
            "W": self.W.detach().cpu(),
            "optim": self.optim.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        for m in (self.policy_net, self.target_net, self.k_net):
            self._unalias_buffers(m)
        self.policy_net.load_state_dict(d["policy_net"])
        self.target_net.load_state_dict(d["target_net"])
        self.k_net.load_state_dict(d["k_net"])
        with torch.no_grad():
            self.W.copy_(d["W"].to(self.device))
        self.optim.load_state_dict(d["optim"])
        self.encoder_optim.load_state_dict(d["encoder_optim"])
