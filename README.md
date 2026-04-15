# so2_equi_rl

Scratch implementation of Wang et al., *SO(2)-Equivariant Reinforcement Learning* (ICLR 2022), for the close-loop BulletArm tasks. CS 5180 final project, Spring 2026.

## Setup

Clone with submodules:

```bash
git clone --recursive git@github.com:wade-g-williams/so2_equi_rl.git
cd so2_equi_rl
```

Create the conda env and install both packages editable:

```bash
conda env create -f environment.yml
conda activate equi_rl
uv pip install -e .
uv pip install -e helping_hands_rl_envs
```

## Usage

Train SAC on one of the BulletArm close-loop tasks:

```bash
python scripts/train_sac.py \
    --env-name close_loop_block_picking \
    --total-steps 50000 \
    --seed 0 \
    --run-name equi_sac
```

`train_sac.py` auto-registers every field on `SACConfig` as a `--kebab-case` flag, so anything in [configs/base.py](src/so2_equi_rl/configs/base.py) or [configs/sac.py](src/so2_equi_rl/configs/sac.py) is overridable from the CLI (`--gamma`, `--batch-size`, `--actor-lr`, `--group-order`, ...). Run `python scripts/train_sac.py --help` for the full list.

Supported tasks (from [envs/wrapper.py](src/so2_equi_rl/envs/wrapper.py)): `close_loop_block_reaching`, `close_loop_block_picking`, `close_loop_block_pulling`, `close_loop_block_stacking`, `close_loop_block_picking_corner`, `close_loop_drawer_opening`, `close_loop_house_building_1`, `close_loop_household_picking`.

Each run drops a timestamped directory under `outputs/` containing the resolved config, TensorBoard event file, per-eval metrics (`.npy`/`.json`), and checkpoints. Resume with `--resume outputs/<run>/ckpts/latest.pt`.

Plot learning curves across runs:

```bash
python scripts/plot_results.py
```

This reads every tfevents file under `outputs/` and writes per-task and combined figures to `outputs/plots/`.

## Layout

```text
so2_equi_rl/
|-- environment.yml                 conda env pinned to the versions the paper used
|-- pyproject.toml                  package metadata and deps
|-- .pre-commit-config.yaml         format + lint on commit
|-- helping_hands_rl_envs/          BulletArm sim, pulled in as a submodule
|-- reference/                      paper PDF, project description, poster
|-- report/
|   |-- poster.tex                  AAAI-style poster source
|   |-- poster.pdf                  compiled poster
|   `-- figures/                    plots and renders used in the poster and report
|-- scripts/
|   |-- train_sac.py                CLI entry point, builds everything and hands it to Trainer
|   `-- plot_results.py             reads tfevents from outputs/ and plots learning curves
|-- outputs/                        per-run logs, metrics, and checkpoints (gitignored except tfevents/npy/json/pdf)
`-- src/so2_equi_rl/
    |-- agents/
    |   |-- base.py                 agent interface (act, update, save/load)
    |   `-- sac.py                  SAC: actor + critic updates, entropy tuning
    |-- buffers/
    |   `-- replay.py               fixed-capacity replay buffer, uniform-sampled batches
    |-- configs/
    |   |-- base.py                 shared training settings (env, seed, steps, device, out dir)
    |   `-- sac.py                  SAC hyperparameters (lrs, gamma, tau, batch size, ...)
    |-- envs/
    |   `-- wrapper.py              wraps BulletArm to return batched torch tensors
    |-- networks/
    |   |-- encoders.py             equivariant CNN that turns the heightmap into features
    |   `-- sac_heads.py            equivariant actor + critic heads
    |-- trainers/
    |   `-- trainer.py              agent-agnostic rollout + update + eval + checkpoint loop
    `-- utils/
        |-- logging.py              metrics, configs, checkpoints to disk
        |-- preprocessing.py        folds gripper state into the image tensor
        `-- seeding.py              seeds every RNG for reproducibility
```

## Results

Compared the SO(2)-equivariant agent against plain-CNN, DrQ, RAD, and FERM baselines on three close-loop BulletArm tasks (`block_picking`, `block_pulling`, `drawer_opening`), 2 seeds each, 50k env steps.

Under SAC, equivariant reaches ~0.93–0.94 final eval success on all three tasks, while every non-equivariant baseline stays near zero at the same budget. One DrQ seed on `block_pulling` is the only exception. The DQN comparison shows the same ordering on `block_pulling`.

- Combined SAC curves: [outputs/plots/learning_curves_combined_sac.pdf](outputs/plots/learning_curves_combined_sac.pdf)
- Combined DQN curves: [outputs/plots/learning_curves_combined_dqn.pdf](outputs/plots/learning_curves_combined_dqn.pdf)
- Per-task curves: [outputs/plots/](outputs/plots/)
- Environment panel: [report/figures/env_panel.png](report/figures/env_panel.png)
- Full write-up and poster: [report/poster.pdf](report/poster.pdf)

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
