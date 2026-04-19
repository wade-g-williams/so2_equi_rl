"""Sanity tests for agents/sac_rad.py.

Pins the SACRADAgent contract: update() runs end-to-end on both the equi
and CNN variants with finite losses and moving params, the RAD = SAC
degenerate case (zero theta) matches vanilla SACAgent.update within
float tolerance, and aug RNG reproducibility gives identical losses on
two identically-seeded agents.

- Finite losses and nonzero param delta on equi RAD.
- Finite losses and nonzero param delta on CNN RAD.
- Zero theta matches SACAgent exactly.
- Two identically-seeded agents produce identical losses on the same batch.
"""

import math

import torch

from so2_equi_rl.agents import SACAgent, SACRADAgent
from so2_equi_rl.buffers.replay import Transition
from so2_equi_rl.configs import SACConfig, SACRADConfig
from so2_equi_rl.networks import (
    CNNActor,
    CNNCritic,
    CNNEncoder,
    EquiActor,
    EquiCritic,
    EquiEncoder,
)
from so2_equi_rl.utils import augmentation as aug_mod

# Tiny nets so the tests stay fast and run CPU-only on the CI machine.
_N_HIDDEN = 32
_GROUP_ORDER = 8


def _rad_cfg(**overrides) -> SACRADConfig:
    base = dict(
        obs_channels=2,
        action_dim=5,
        n_hidden=_N_HIDDEN,
        group_order=_GROUP_ORDER,
        device="cpu",
        seed=0,
    )
    base.update(overrides)
    return SACRADConfig(**base)


def _sac_cfg(**overrides) -> SACConfig:
    base = dict(
        obs_channels=2,
        action_dim=5,
        n_hidden=_N_HIDDEN,
        group_order=_GROUP_ORDER,
        device="cpu",
        seed=0,
    )
    base.update(overrides)
    return SACConfig(**base)


def _make_equi_rad(cfg: SACRADConfig) -> SACRADAgent:
    # Seed before constructing so network init is reproducible.
    torch.manual_seed(cfg.seed)
    return SACRADAgent(
        cfg=cfg,
        encoder_cls=EquiEncoder,
        actor_cls=EquiActor,
        critic_cls=EquiCritic,
    )


def _make_cnn_rad(cfg: SACRADConfig) -> SACRADAgent:
    torch.manual_seed(cfg.seed)
    return SACRADAgent(
        cfg=cfg,
        encoder_cls=CNNEncoder,
        actor_cls=CNNActor,
        critic_cls=CNNCritic,
    )


def _make_synthetic_batch(batch_size: int = 4) -> Transition:
    torch.manual_seed(1)
    return Transition(
        state=torch.randn(batch_size, 1),
        obs=torch.randn(batch_size, 1, 128, 128),
        action=torch.randn(batch_size, 5).clamp(-1.0, 1.0),
        reward=torch.randn(batch_size),
        next_state=torch.randn(batch_size, 1),
        next_obs=torch.randn(batch_size, 1, 128, 128),
        done=torch.zeros(batch_size),
    )


def _param_l2_delta(before: list, after_params) -> float:
    # Squared-sum delta across every parameter tensor. A nonzero value
    # proves the optimizer step actually moved weights this update.
    total = 0.0
    for p_before, p_after in zip(before, after_params):
        total += (p_after.detach() - p_before).pow(2).sum().item()
    return total


def _run_one_update_assert_moves(agent) -> dict:
    batch = _make_synthetic_batch()

    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]

    losses = agent.update(batch)

    expected_keys = {
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "q1_mean",
        "q2_mean",
    }
    assert set(losses.keys()) == expected_keys
    for k, v in losses.items():
        assert isinstance(v, float), f"loss {k} not float: {type(v)}"
        assert math.isfinite(v), f"loss {k} not finite: {v}"

    actor_delta = _param_l2_delta(actor_before, agent.actor.parameters())
    critic_delta = _param_l2_delta(critic_before, agent.critic.parameters())
    assert actor_delta > 0.0, f"actor params did not move (delta={actor_delta})"
    assert critic_delta > 0.0, f"critic params did not move (delta={critic_delta})"

    return losses


def test_equi_rad_update_finite_and_moves():
    # End-to-end: equi RAD agent, one update, finite losses, params moved.
    agent = _make_equi_rad(_rad_cfg())
    _run_one_update_assert_moves(agent)
    print("equi RAD update OK")


def test_cnn_rad_update_finite_and_moves():
    # Same end-to-end check on the CNN variant.
    agent = _make_cnn_rad(_rad_cfg())
    _run_one_update_assert_moves(agent)
    print("cnn RAD update OK")


def test_rad_degenerate_matches_sac(monkeypatch):
    # Zero theta collapses RAD's rotation to the identity, so the update
    # should match vanilla SACAgent.update byte-close after we match
    # initial weights.
    def _zero_angles(batch, mode, group_order, *, generator):  # noqa: ARG001
        return torch.zeros(batch, dtype=torch.float32)

    monkeypatch.setattr(aug_mod, "sample_so2_angles", _zero_angles)

    cfg_rad = _rad_cfg()
    cfg_sac = _sac_cfg()

    torch.manual_seed(0)
    sac = SACAgent(
        cfg=cfg_sac,
        encoder_cls=EquiEncoder,
        actor_cls=EquiActor,
        critic_cls=EquiCritic,
    )

    torch.manual_seed(0)
    rad = SACRADAgent(
        cfg=cfg_rad,
        encoder_cls=EquiEncoder,
        actor_cls=EquiActor,
        critic_cls=EquiCritic,
    )

    # Hard-sync every trainable tensor. state_dict copying is avoided here
    # because e2cnn's basis-expansion buffers alias memory under the hood
    # and trip load_state_dict's in-place copy. Parameter-wise .data.copy_
    # sidesteps the alias path, and the basis buffers are determined by
    # construction config (not RNG) so they already match across the two
    # fresh builds.
    with torch.no_grad():
        for dst, src in zip(rad.actor.parameters(), sac.actor.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(rad.critic.parameters(), sac.critic.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(
            rad.critic_target.parameters(), sac.critic_target.parameters()
        ):
            dst.data.copy_(src.data)
        rad.log_alpha.copy_(sac.log_alpha.detach())

    batch = _make_synthetic_batch()

    # Both updates draw from the global torch RNG via actor.sample, so
    # sac.update() advances the state that rad.update() would otherwise
    # read. Reset the seed before each call so they see identical RNG.
    torch.manual_seed(1234)
    sac_losses = sac.update(batch)
    torch.manual_seed(1234)
    rad_losses = rad.update(batch)

    # Tolerance is 1e-4 rather than tighter because rotate_obs at theta=0
    # is bilinear-exact only to ~2e-5 due to align_corners=False, and that
    # drift propagates through the critic and actor forward passes.
    for key in ("critic_loss", "actor_loss", "alpha_loss", "q1_mean", "q2_mean"):
        assert math.isclose(
            sac_losses[key], rad_losses[key], rel_tol=0.0, abs_tol=1e-4
        ), f"{key}: SAC {sac_losses[key]} vs RAD {rad_losses[key]}"
    print("RAD = SAC degenerate case OK")


def test_rad_deterministic_given_seed():
    # Two identically-seeded agents on the same batch must produce the
    # same losses. Pins that the aug RNG path is reproducible under the
    # fixed seed offset.
    cfg_a = _rad_cfg(seed=42)
    cfg_b = _rad_cfg(seed=42)

    agent_a = _make_equi_rad(cfg_a)
    agent_b = _make_equi_rad(cfg_b)

    # The second _make_equi_rad call resets manual_seed(42), so the network
    # init should match. Hard-sync parameter-wise anyway to guard against
    # any lurking global-RNG consumption during the first agent's init.
    # See test_rad_degenerate_matches_sac for why this avoids state_dict.
    with torch.no_grad():
        for dst, src in zip(agent_b.actor.parameters(), agent_a.actor.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(agent_b.critic.parameters(), agent_a.critic.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(
            agent_b.critic_target.parameters(), agent_a.critic_target.parameters()
        ):
            dst.data.copy_(src.data)
        agent_b.log_alpha.copy_(agent_a.log_alpha.detach())

    batch = _make_synthetic_batch()

    # Same global-RNG reset rule as the degenerate test: both updates
    # draw from torch.manual_seed state via actor.sample, so reset
    # between the two calls to give them identical RNG input.
    torch.manual_seed(1234)
    losses_a = agent_a.update(batch)
    torch.manual_seed(1234)
    losses_b = agent_b.update(batch)

    for key in ("critic_loss", "actor_loss", "alpha_loss", "q1_mean", "q2_mean"):
        assert (
            losses_a[key] == losses_b[key]
        ), f"{key}: {losses_a[key]} vs {losses_b[key]}"
    print("RAD determinism OK")
