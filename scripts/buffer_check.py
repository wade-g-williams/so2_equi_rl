"""MVP sanity check for buffer.py -- push/sample round-trip, shapes,
dtypes, wrap-around, and a single integration push from env.py."""

import torch

from so2_equi_rl.buffer import ReplayBuffer, Transition
from so2_equi_rl.env import EnvWrapper


def test_shapes_and_dtypes():
    capacity = 256
    state_dim = 1
    obs_shape = (1, 128, 128)
    action_dim = 5

    buf = ReplayBuffer(capacity, state_dim, obs_shape, action_dim)
    assert len(buf) == 0

    # Push 300 transitions in 3-wide batches -- overflows capacity (256),
    # exercising the wrap-around path at least once.
    batch = 3
    for _ in range(100):
        buf.push(
            states=torch.zeros(batch, state_dim),
            obs=torch.zeros(batch, *obs_shape),
            actions=torch.zeros(batch, action_dim),
            rewards=torch.zeros(batch),
            next_states=torch.zeros(batch, state_dim),
            next_obs=torch.zeros(batch, *obs_shape),
            dones=torch.zeros(batch),
        )

    assert len(buf) == capacity, "len should saturate at capacity, got {}".format(
        len(buf)
    )

    sample = buf.sample(64)
    assert isinstance(sample, Transition)
    assert sample.state.shape == (64, state_dim)
    assert sample.obs.shape == (64, 1, 128, 128)
    assert sample.action.shape == (64, action_dim)
    assert sample.reward.shape == (64,)
    assert sample.next_state.shape == (64, state_dim)
    assert sample.next_obs.shape == (64, 1, 128, 128)
    assert sample.done.shape == (64,)
    assert sample.step_left.shape == (64,)
    assert sample.expert.shape == (64,)
    assert sample.state.dtype == torch.float32
    assert sample.obs.dtype == torch.float32
    assert sample.expert.dtype == torch.bool
    print("shape + dtype check OK")


def test_round_trip():
    # One unique transition in, one out -- every field must match bitwise.
    buf = ReplayBuffer(capacity=4, state_dim=1, obs_shape=(1, 4, 4), action_dim=5)

    s = torch.tensor([[0.7]])
    o = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16.0
    a = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5]])
    r = torch.tensor([1.0])
    ns = torch.tensor([[0.9]])
    no = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 32.0
    d = torch.tensor([1.0])

    buf.push(
        states=s,
        obs=o,
        actions=a,
        rewards=r,
        next_states=ns,
        next_obs=no,
        dones=d,
    )
    assert len(buf) == 1

    # _size == 1, so randint(0, 1) deterministically returns 0.
    batch = buf.sample(batch_size=1)
    assert torch.equal(batch.state, s), "state mismatch"
    assert torch.equal(batch.obs, o), "obs mismatch"
    assert torch.equal(batch.action, a), "action mismatch"
    assert torch.equal(batch.reward, r), "reward mismatch"
    assert torch.equal(batch.next_state, ns), "next_state mismatch"
    assert torch.equal(batch.next_obs, no), "next_obs mismatch"
    assert torch.equal(batch.done, d), "done mismatch"
    print("round-trip check OK")


def test_empty_sample_raises():
    buf = ReplayBuffer(capacity=4, state_dim=1, obs_shape=(1, 4, 4), action_dim=5)
    try:
        buf.sample(1)
    except ValueError:
        print("empty-sample guard OK")
        return
    raise AssertionError("sample() on empty buffer should have raised ValueError")


def test_env_integration():
    # Smoke test: push one real env.step output.
    env = EnvWrapper("close_loop_block_reaching", num_processes=0, seed=0)
    buf = ReplayBuffer(
        capacity=64,
        state_dim=env.state_dim,
        obs_shape=(1, env.obs_size, env.obs_size),
        action_dim=env.action_dim,
    )

    states, obs = env.reset()
    actions = torch.zeros(env.batch_size, env.action_dim)
    next_states, next_obs, rewards, dones = env.step(actions)
    buf.push(
        states=states,
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        next_obs=next_obs,
        dones=dones,
    )
    assert len(buf) == env.batch_size
    sample = buf.sample(1)
    assert sample.obs.shape == (1, 1, env.obs_size, env.obs_size)
    env.close()
    print("env integration check OK")


if __name__ == "__main__":
    test_shapes_and_dtypes()
    test_round_trip()
    test_empty_sample_raises()
    test_env_integration()
    print("buffer.py MVP OK")
