# so2_equi_rl

Reimplementation of Wang et al., *SO(2)-Equivariant Reinforcement Learning* (ICLR 2022), on the three close-loop BulletArm tasks from the paper, plus a ManiSkill3 extension covering PickCube-v1, PullCube-v1, and StackCube-v1. CS 5180 final project, Spring 2026.

## Setup

Two conda envs, one per simulator. BulletArm needs Python 3.7 to match the paper, ManiSkill3 needs 3.10. The BulletArm sim is vendored as a submodule.

```bash
git clone --recursive git@github.com:wade-g-williams/so2_equi_rl.git
cd so2_equi_rl
```

BulletArm env:

```bash
conda env create -f environment.yml
conda activate equi_rl
uv pip install -e .
uv pip install -e helping_hands_rl_envs
```

Patch the submodule. Upstream's `SingleRunner.step` accepts an `auto_reset` kwarg but silently ignores it, so eval episodes never actually reset.

```bash
cd helping_hands_rl_envs
git apply ../patches/helping_hands_auto_reset.patch
cd ..
```

ManiSkill3 env:

```bash
conda env create -f environment_ms3.yml
conda activate ms3_equi
uv pip install -e .
```

## Usage

Every training script auto-registers its config dataclass fields as `--kebab-case` CLI flags, so most hyperparameters are overridable without touching config files. Pass `--help` on any script for the full list.

Example, Equi-SAC on block_picking:

```bash
python scripts/train_sac.py \
    --encoder equi \
    --env-name close_loop_block_picking \
    --env-backend bulletarm \
    --total-steps 20000 \
    --seed 0 \
    --run-name equi_sac
```

Training scripts:

- `train_sac.py`, vanilla SAC, `--encoder {equi, cnn}`
- `train_sac_drq.py`, DrQ-SAC
- `train_sac_rad.py`, RAD-SAC
- `train_sac_ferm.py`, FERM-SAC, CNN hardcoded
- `train_dqn.py`, DQN plus DrQ, RAD, and CURL variants, selected via `--network {equi, cnn, drq, rad, curl}`

Paper scope is three close-loop BulletArm tasks:

- `close_loop_block_picking`
- `close_loop_block_pulling`
- `close_loop_drawer_opening`

Plus the ManiSkill3 extension, run with `--env-backend maniskill` in the `ms3_equi` env:

- `PickCube-v1`
- `PullCube-v1`
- `StackCube-v1`

PickCube and PullCube are the direct MS3 analogues of block_picking and block_pulling. StackCube stands in for drawer_opening because MS3's OpenCabinetDrawer uses a mobile Fetch robot, not a tabletop Panda.

The BulletArm wrapper supports more tasks than these three, see [envs/wrapper.py](src/so2_equi_rl/envs/wrapper.py). Reproduction sticks to the paper's.

Each run writes a timestamped directory under `outputs/` with the resolved config, a `metrics.jsonl` eval log, TensorBoard events, and checkpoints. Resume with `--resume outputs/<run>/ckpts/latest.pt`.

Reproducing the full matrix means looping over variants, tasks, and seeds. Each `train_*.py` script is standalone, so run them however fits your hardware.

Plots:

```bash
python scripts/plot_results.py --roots outputs --out outputs/plots
```

Writes one PDF per (alg family, backend). Four total, Fig 6 DQN and Fig 7 SAC on both BulletArm and ManiSkill3.

## Layout

```text
so2_equi_rl/
|-- environment.yml
|-- environment_ms3.yml
|-- pyproject.toml
|-- .pre-commit-config.yaml
|-- helping_hands_rl_envs/
|-- patches/
|-- reference/
|-- report/
|   |-- poster.tex
|   |-- poster.pdf
|   `-- figures/
|-- scripts/
|   |-- train_sac.py
|   |-- train_sac_drq.py
|   |-- train_sac_rad.py
|   |-- train_sac_ferm.py
|   |-- train_dqn.py
|   |-- plot_results.py
|   `-- render_env_panel.py
|-- outputs/
`-- src/so2_equi_rl/
    |-- agents/
    |   |-- base.py
    |   |-- sac.py
    |   |-- sac_drq.py
    |   |-- sac_rad.py
    |   |-- sac_ferm.py
    |   |-- dqn.py
    |   |-- dqn_drq.py
    |   |-- dqn_rad.py
    |   `-- dqn_curl.py
    |-- buffers/
    |   |-- replay.py
    |   `-- so2_aug.py
    |-- configs/
    |   |-- base.py
    |   |-- sac.py
    |   |-- sac_drq.py
    |   |-- sac_rad.py
    |   |-- sac_ferm.py
    |   |-- dqn.py
    |   |-- dqn_drq.py
    |   |-- dqn_rad.py
    |   `-- dqn_curl.py
    |-- envs/
    |   |-- wrapper.py
    |   |-- maniskill_wrapper.py
    |   `-- maniskill_experts.py
    |-- networks/
    |   |-- encoders.py
    |   |-- sac_heads.py
    |   `-- dqn_heads.py
    |-- trainers/
    |   |-- base.py
    |   |-- sac.py
    |   `-- dqn.py
    `-- utils/
        |-- augmentation.py
        |-- cli_args.py
        |-- logging.py
        |-- preprocessing.py
        `-- seeding.py
```

## Results

Paper reproduction covers Fig 6 (DQN plus DrQ, RAD, CURL) and Fig 7 (SAC plus DrQ, RAD, FERM) on the three close-loop BulletArm tasks, 2 seeds each, 20k env steps. The ManiSkill3 extension runs the same variants on PickCube-v1, PullCube-v1, and StackCube-v1 under the same budget.

- Fig 6 BulletArm, [outputs/plots/figure6_dqn_bulletarm.pdf](outputs/plots/figure6_dqn_bulletarm.pdf)
- Fig 7 BulletArm, [outputs/plots/figure7_sac_bulletarm.pdf](outputs/plots/figure7_sac_bulletarm.pdf)
- MS3 extension, [outputs/plots/figure6_dqn_maniskill.pdf](outputs/plots/figure6_dqn_maniskill.pdf), [outputs/plots/figure7_sac_maniskill.pdf](outputs/plots/figure7_sac_maniskill.pdf)
- Environment panel, [report/figures/env_panel.png](report/figures/env_panel.png)
- Full writeup and poster, [report/poster.pdf](report/poster.pdf)

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
