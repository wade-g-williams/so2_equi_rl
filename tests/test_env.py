"""Smoke test for EnvWrapper.

Runs reset + 10 random steps + close, first single-process then multi-process,
and asserts every tensor shape matches the always-batched contract.
"""

import numpy as np
import torch

from so2_equi_rl.env import EnvWrapper


def run(num_processes: int) -> None:
    label = "num_processes={}".format(num_processes)
    print("\n=== {} ===".format(label))

    env = EnvWrapper(
        "close_loop_block_picking",
        num_processes=num_processes,
        seed=0,
    )

    n = env.batch_size
    print(
        "batch_size={}, action_dim={}, state_dim={}, obs_size={}".format(
            n, env.action_dim, env.state_dim, env.obs_size
        )
    )

    H = env.obs_size
    states, obs = env.reset()
    assert states.shape == (n, 1), "states: got {}, want {}".format(
        tuple(states.shape), (n, 1)
    )
    assert obs.shape == (n, 1, H, H), "obs: got {}, want {}".format(
        tuple(obs.shape), (n, 1, H, H)
    )
    assert states.dtype == torch.float32 and obs.dtype == torch.float32
    print("reset ok -- states {}, obs {}".format(tuple(states.shape), tuple(obs.shape)))

    rng = np.random.default_rng(0)
    for i in range(10):
        action_np = rng.uniform(-0.02, 0.02, size=(n, env.action_dim)).astype(
            np.float32
        )
        action = torch.from_numpy(action_np)
        states, obs, rewards, dones = env.step(action)
        assert states.shape == (n, 1)
        assert obs.shape == (n, 1, H, H)
        assert rewards.shape == (n,)
        assert dones.shape == (n,)
        print(
            "  step {}: reward={} done={}".format(i, rewards.tolist(), dones.tolist())
        )

    env.close()
    print("{} OK".format(label))


if __name__ == "__main__":
    run(num_processes=0)
    run(num_processes=3)
    print("\nenv.py MVP OK (single + multi)")
