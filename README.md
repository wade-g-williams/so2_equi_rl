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

## Layout

```text
so2_equi_rl/
|-- environment.yml                 conda env pinned to the versions the paper used
|-- pyproject.toml                  package metadata and deps
|-- .pre-commit-config.yaml         format + lint on commit
|-- helping_hands_rl_envs/          BulletArm sim, pulled in as a submodule
|-- scripts/
|   `-- train_sac.py                CLI entry point, builds everything and hands it to Trainer
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

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
