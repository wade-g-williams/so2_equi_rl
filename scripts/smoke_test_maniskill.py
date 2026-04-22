"""Smoke test for Joey's ManiSkill patch. Verifies the full wrapper path:
gym.make the EquiPickCube-v1 subclass, reset, run 5 expert-driven steps,
assert shapes / dtypes / value ranges. Exit 0 on success, nonzero on failure.

Run with:
    conda activate ms3_equi
    python scripts/smoke_test_maniskill.py

Expected wall time: ~15-30 s (dominated by MS3 GPU init).
"""

from __future__ import annotations

import sys
import traceback

import torch

from so2_equi_rl.configs.base import TrainConfig
from so2_equi_rl.envs import make_env


def main() -> int:
    cfg = TrainConfig(
        env_name="EquiPickCube-v1",
        env_backend="maniskill",
        num_envs=2,
        obs_size=128,
        ms3_reward_mode="normalized_dense",
        ms3_control_mode="pd_ee_delta_pose",
        ms3_sim_backend="gpu",
    )

    print(f"[smoke] building env  task={cfg.env_name}  num_envs={cfg.num_envs}")
    env = make_env(cfg, seed=0, num_processes=0, num_envs=cfg.num_envs)

    # Reset returns (state, obs) per _extract's contract.
    print("[smoke] reset")
    state, obs = env.reset()

    # Shape checks
    assert state.shape == (2, 1), f"state shape {state.shape} != (2, 1)"
    assert obs.shape == (2, 1, 128, 128), f"obs shape {obs.shape} != (2,1,128,128)"
    assert state.dtype == torch.float32, f"state dtype {state.dtype}"
    assert obs.dtype == torch.float32, f"obs dtype {obs.dtype}"

    # Obs values should be gripper-relative depth clamped to [-depth_max, depth_max].
    assert torch.isfinite(obs).all(), "obs has non-finite values"
    assert (
        obs.abs().max() <= cfg.ms3_depth_max + 1e-4
    ), f"obs max |v|={obs.abs().max().item():.3f} > depth_max={cfg.ms3_depth_max}"
    # Depth should vary across pixels. If it's all-zero the camera is broken or
    # _extract is subtracting something it shouldn't.
    assert obs.std() > 1e-5, f"obs std={obs.std().item():.2e} (depth is constant)"

    print(
        f"[smoke] reset OK  "
        f"state.sum={state.sum().item():.1f}  "
        f"obs.min={obs.min().item():.3f}  "
        f"obs.max={obs.max().item():.3f}  "
        f"obs.mean={obs.mean().item():.3f}  "
        f"obs.std={obs.std().item():.3f}"
    )

    # Step 5 times using the scripted expert. This exercises _physical_to_ms3,
    # env.step, and the rgbd -> heightmap pipeline end-to-end.
    print("[smoke] stepping 5 expert actions")
    for i in range(5):
        action = env.get_expert_action()
        assert action.shape == (2, 5), f"expert action shape {action.shape}"
        assert torch.isfinite(action).all(), "expert action has non-finite values"

        out = env.step(action)

        assert out.state.shape == (2, 1)
        assert out.obs.shape == (2, 1, 128, 128)
        assert out.reward.shape == (2,)
        assert out.done.shape == (2,)
        assert out.success.shape == (2,)
        assert torch.isfinite(out.obs).all(), f"step {i} obs non-finite"
        assert torch.isfinite(out.reward).all(), f"step {i} reward non-finite"

        print(
            f"  step {i}  "
            f"r={out.reward.tolist()}  "
            f"done={out.done.tolist()}  "
            f"success={out.success.tolist()}  "
            f"obs.std={out.obs.std().item():.3f}"
        )

    env.close()
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("[smoke] FAIL")
        sys.exit(1)
