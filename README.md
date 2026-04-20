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

Train the equivariant SAC agent on one of the BulletArm close-loop tasks:

```bash
python scripts/train_sac.py \
    --encoder equi \
    --env-name close_loop_block_picking \
    --total-steps 50000 \
    --seed 0 \
    --run-name equi_sac
```

Every training script auto-registers its `Config` dataclass fields as `--kebab-case` flags, so anything in [configs/base.py](src/so2_equi_rl/configs/base.py) and the per-variant config (e.g. [configs/sac.py](src/so2_equi_rl/configs/sac.py)) is CLI-overridable (`--gamma`, `--batch-size`, `--actor-lr`, `--group-order`, ...). Run any script with `--help` for the full list.

Variants and their backbone flags:

| Script | Agent | Backbone flag |
| --- | --- | --- |
| [train_sac.py](scripts/train_sac.py) | vanilla SAC | `--encoder {equi,cnn}` |
| [train_sac_drq.py](scripts/train_sac_drq.py) | DrQ-SAC | `--encoder {equi,cnn}` |
| [train_sac_rad.py](scripts/train_sac_rad.py) | RAD-SAC | `--encoder {equi,cnn}` (equi+RAD is an ablation) |
| [train_sac_ferm.py](scripts/train_sac_ferm.py) | FERM-SAC | CNN-only (FERM's InfoNCE pretext is the invariance analog) |
| [train_dqn.py](scripts/train_dqn.py) | DQN | `--network {equi,cnn,rad}` (`rad` = CNN + RAD agent) |

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
|   |-- train_sac.py                vanilla SAC, --encoder {equi,cnn}
|   |-- train_sac_drq.py            DrQ-SAC, --encoder {equi,cnn}
|   |-- train_sac_rad.py            RAD-SAC, --encoder {equi,cnn}
|   |-- train_sac_ferm.py           FERM-SAC, CNN-only
|   |-- train_dqn.py                DQN, --network {equi,cnn,rad}
|   |-- plot_results.py             reads tfevents from outputs/ and plots learning curves
|   `-- render_env_panel.py         renders the 1x3 BulletArm panel for the poster
|-- outputs/                        per-run logs, metrics, and checkpoints (gitignored except tfevents/npy/json/pdf)
`-- src/so2_equi_rl/
    |-- agents/
    |   |-- base.py                 agent interface (act, update, save/load)
    |   |-- sac.py                  vanilla SAC: actor + critic updates, entropy tuning
    |   |-- sac_drq.py              DrQ-SAC: K/M augmented targets and Q regression
    |   |-- sac_rad.py              RAD-SAC: one shared SO(2) rotation per transition
    |   |-- sac_ferm.py             FERM-SAC: InfoNCE pretext + SAC fine-tune, CNN-only
    |   |-- dqn.py                  vanilla DQN: eps-greedy, target net, Huber loss
    |   `-- dqn_rad.py              RAD-DQN: same as dqn with per-transition rotation
    |-- buffers/
    |   `-- replay.py               fixed-capacity replay buffer, uniform-sampled batches
    |-- configs/
    |   |-- base.py                 shared training settings (env, seed, steps, device, out dir)
    |   |-- sac.py                  vanilla SAC hyperparameters
    |   |-- sac_drq.py              DrQ-SAC hyperparameters (adds K, M)
    |   |-- sac_rad.py              RAD-SAC hyperparameters
    |   |-- sac_ferm.py             FERM-SAC hyperparameters (pretext batch, InfoNCE temperature)
    |   |-- dqn.py                  DQN hyperparameters (eps schedule, target sync)
    |   `-- dqn_rad.py              RAD-DQN hyperparameters
    |-- envs/
    |   `-- wrapper.py              wraps BulletArm to return batched torch tensors
    |-- networks/
    |   |-- encoders.py             equivariant and plain-CNN encoders (heightmap -> features)
    |   |-- sac_heads.py            equivariant and plain-CNN actor + critic heads
    |   `-- dqn_heads.py            equivariant and plain-CNN discrete Q-heads
    |-- trainers/
    |   |-- base.py                 agent-agnostic rollout + update + eval + checkpoint loop
    |   |-- sac.py                  SACTrainer: expert warmup + always-stochastic exploration
    |   `-- dqn.py                  DQNTrainer: eps-greedy rollout + target-net sync
    `-- utils/
        |-- augmentation.py         SO(2) rotation augmentations shared by DrQ/RAD/FERM
        |-- cli_args.py             dataclass <-> argparse bridge (auto-register --kebab-case flags)
        |-- logging.py              metrics, configs, checkpoints to disk
        |-- preprocessing.py        folds gripper state into the image tensor
        `-- seeding.py              seeds every RNG for reproducibility
```

## Results

Compared the SO(2)-equivariant agent against plain-CNN, DrQ, RAD, and FERM baselines on three close-loop BulletArm tasks (`block_picking`, `block_pulling`, `drawer_opening`), 2 seeds each, 50k env steps.

Under SAC, equivariant reaches ~0.93-0.94 final eval success on all three tasks, while every non-equivariant baseline stays near zero at the same budget. One DrQ seed on `block_pulling` is the only exception. The DQN comparison shows the same ordering on `block_pulling`.

- Combined SAC curves: [outputs/plots/learning_curves_combined_sac.pdf](outputs/plots/learning_curves_combined_sac.pdf)
- Combined DQN curves: [outputs/plots/learning_curves_combined_dqn.pdf](outputs/plots/learning_curves_combined_dqn.pdf)
- Per-task curves: [outputs/plots/](outputs/plots/)
- Environment panel: [report/figures/env_panel.png](report/figures/env_panel.png)
- Full write-up and poster: [report/poster.pdf](report/poster.pdf)

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
