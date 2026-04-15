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
├── environment.yml
├── pyproject.toml
├── .pre-commit-config.yaml
├── helping_hands_rl_envs/        (submodule)
└── src/so2_equi_rl/
    ├── agents/
    ├── buffers/
    ├── configs/
    ├── envs/
    ├── networks/
    ├── trainers/
    └── utils/
```

### Root

- `environment.yml`: conda env with the package versions the paper used
- `pyproject.toml`: Python package info and dependencies
- `.pre-commit-config.yaml`: auto-formatting and linting on every commit
- `helping_hands_rl_envs/`: the paper's BulletArm simulator, pulled in as a submodule

### `agents/`

The learner.

- `base.py`: the interface every agent has to implement (pick an action, learn from a batch, save/load)
- `sac.py`: the SAC algorithm itself. Picks actions, trains the actor and critic, handles entropy tuning

### `buffers/`

Experience replay.

- `replay.py`: stores recent transitions and samples random batches for training

### `configs/`

All the knobs in one place.

- `base.py`: shared training settings (env, seed, steps, device, output dir)
- `sac.py`: SAC-specific hyperparameters (learning rates, γ, τ, batch size, etc.)

### `envs/`

The simulator.

- `wrapper.py`: wraps BulletArm so the agent gets batched PyTorch tensors

### `networks/`

The neural nets.

- `encoders.py`: equivariant CNN that turns the heightmap image into features
- `sac_heads.py`: the actor (picks actions) and critic (scores them), both equivariant

### `trainers/`

The main training loop. *(not written yet)*

### `utils/`

- `logging.py`: saves metrics, configs, and checkpoints to disk *(not written yet)*
- `preprocessing.py`: merges the gripper state into the image so the encoder sees one tensor
- `schedules.py`: things that change over training, like exploration decay *(not written yet)*
- `seeding.py`: sets every random seed so runs are reproducible

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
