"""MVP sanity check for networks/equi_encoder.py -- shapes, dtypes,
C_4-subgroup equivariance, param count, and an env.reset() smoke test.

No full C_8 test: 45-deg rotations on a square grid need bilinear
interp, which doesn't commute with R2Conv pixel-exactly. The C_4
subgroup is pixel-exact via torch.rot90 and catches any real bug.
"""

import torch

from so2_equi_rl.env import EnvWrapper
from so2_equi_rl.networks import EquiEncoder, tile_state


def test_shape_and_dtype():
    encoder = EquiEncoder(obs_channels=2, n_hidden=128, group_order=8).eval()
    assert encoder.output_dim == 1024, "expected output_dim 1024, got {}".format(
        encoder.output_dim
    )

    obs = torch.randn(4, 1, 128, 128)
    state = torch.rand(4, 1)
    tiled = tile_state(obs, state)
    assert tiled.shape == (4, 2, 128, 128), "tile_state shape wrong: {}".format(
        tiled.shape
    )

    with torch.no_grad():
        out = encoder(tiled)
    assert out.tensor.shape == (4, 1024, 1, 1), "encoder tensor shape: {}".format(
        out.tensor.shape
    )
    assert out.tensor.dtype == torch.float32
    assert out.type == encoder.output_type, "output type drift"
    print("shape + dtype check OK  (output_dim={})".format(encoder.output_dim))


def test_c4_subgroup_equivariance():
    # torch.rot90 is pixel-exact for the C_4 subgroup of C_8. Since C_4 is
    # a subgroup, an encoder that is C_8-equivariant must also be C_4-
    # equivariant -- if this test fails, something is fundamentally wrong.
    torch.manual_seed(0)
    batch = 2
    n_hidden = 32
    group_order = 8
    encoder = EquiEncoder(
        obs_channels=2, n_hidden=n_hidden, group_order=group_order
    ).eval()

    obs = torch.randn(batch, 2, 128, 128)

    with torch.no_grad():
        out_plain = encoder(obs).tensor
        obs_rot = torch.rot90(obs, k=1, dims=(-2, -1))
        out_rot = encoder(obs_rot).tensor

    # C_8 regular rep: each of the n_hidden "slots" holds group_order values
    # indexed by group element. A 90 deg rotation (= 2 steps in C_8)
    # cyclically shifts each slot's 8 values by 2 positions. Spatial dims
    # are (1,1), so only the channel permutation matters.
    # NOTE: shifts=+2 is the empirical match for torch.rot90(k=1) + e2cnn's
    # regular_repr convention. If this test ever fails with a large max err,
    # try shifts=-2 first -- that's a sign-convention mismatch, not an
    # encoder bug.
    assert out_plain.shape[1] == n_hidden * group_order
    shifted = (
        out_plain.view(batch, n_hidden, group_order, 1, 1)
        .roll(shifts=2, dims=2)
        .view_as(out_plain)
    )

    err = (shifted - out_rot).abs().max().item()
    # Tight tolerance -- C_4 subgroup rotations are pixel-exact on a
    # square grid, so residual error should be floating-point noise.
    assert err < 1e-4, "C_4 subgroup equivariance broken, max err={:.2e}".format(err)
    print("C_4 subgroup equivariance OK  (max err={:.2e})".format(err))


def test_param_count():
    encoder = EquiEncoder(obs_channels=2, n_hidden=128, group_order=8)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # Sanity bounds only. Not asserting an exact value -- e2cnn basis
    # expansion param counts depend on internal filter choices.
    assert 1e5 < n_params < 5e7, "param count out of range: {}".format(n_params)
    print(
        "param count OK             ({:.2f} M trainable params)".format(n_params / 1e6)
    )


def test_env_integration():
    # Smoke test: push one real env.reset() output through tile_state
    # and the encoder, same pattern as test_buffer.py.
    env = EnvWrapper("close_loop_block_reaching", num_processes=0, seed=0)
    encoder = EquiEncoder(obs_channels=2, n_hidden=64, group_order=8).eval()

    states, obs = env.reset()
    tiled = tile_state(obs, states)
    assert tiled.shape == (env.batch_size, 2, env.obs_size, env.obs_size)

    with torch.no_grad():
        out = encoder(tiled)
    assert out.tensor.shape == (env.batch_size, 64 * 8, 1, 1)
    env.close()
    print("env integration check OK")


if __name__ == "__main__":
    test_shape_and_dtype()
    test_c4_subgroup_equivariance()
    test_param_count()
    test_env_integration()
    print("equi_encoder.py MVP OK")
