# so2_equi_rl

Reimplementation of Wang et al., *SO(2)-Equivariant Reinforcement Learning* (ICLR 2022), on the three close-loop BulletArm tasks from the paper, plus a ManiSkill3 PickCube-v1 extension. CS 5180 final project, Spring 2026.

## Setup

Two conda envs because BulletArm is pinned to Python 3.7 (paper's version) and ManiSkill3 needs 3.10. The BulletArm sim is vendored as a submodule, so clone recursively.

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

Patch the BulletArm submodule. Upstream's `SingleRunner.step` accepts an `auto_reset` kwarg but silently ignores it, which breaks the eval loop (the env stays terminal after the first done and `eval/length_mean` collapses). The patch makes it match `MultiRunner`.

```bash
cd helping_hands_rl_envs
git apply ../patches/helping_hands_auto_reset.patch
cd ..
```

ManiSkill3 env, only needed for the PickCube-v1 extension:

```bash
conda env create -f environment_ms3.yml
conda activate ms3_equi
uv pip install -e .
```

## Usage

Every training script auto-registers its config dataclass fields as `--kebab-case` flags, so almost every hyperparameter is overridable from the CLI without editing config files. Pass `--help` for the full list.

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
- `train_sac_drq.py`, DrQ-SAC, same encoder flag (paper runs cnn, equi is an ablation)
- `train_sac_rad.py`, RAD-SAC, same encoder flag
- `train_sac_ferm.py`, FERM-SAC, no encoder flag (CNN hardcoded)
- `train_dqn.py`, DQN plus DrQ, RAD, and CURL, selected via `--network {equi, cnn, drq, rad, curl}`

Paper scope is three close-loop BulletArm tasks plus the ManiSkill3 extension:

- `close_loop_block_picking`
- `close_loop_block_pulling`
- `close_loop_drawer_opening`
- `PickCube-v1` (run with `--env-backend maniskill` in the `ms3_equi` env)

The BulletArm wrapper can run more tasks than these, see [envs/wrapper.py](src/so2_equi_rl/envs/wrapper.py). Reproduction sticks to the paper's three.

Each run writes a timestamped directory under `outputs/` with the resolved config, a `metrics.jsonl` eval log, TensorBoard events, and checkpoints. Resume with `--resume outputs/<run>/ckpts/latest.pt`.

Reproducing the full matrix means looping over variants, tasks, and seeds. Each `train_*.py` script is standalone, so run them however fits your hardware (serial, a bash loop, a job scheduler, whatever).

Plots:

```bash
python scripts/plot_results.py --roots outputs --out outputs/plots
```

Writes one PDF per (alg family, backend). Four total, Fig 6 DQN and Fig 7 SAC on both BulletArm and ManiSkill3.

## Layout

```text
so2_equi_rl/
|-- environment.yml                 BulletArm conda env (Python 3.7)
|-- environment_ms3.yml             ManiSkill3 conda env (Python 3.10)
|-- pyproject.toml                  package metadata
|-- .pre-commit-config.yaml         format + lint hooks
|-- helping_hands_rl_envs/          BulletArm sim, pulled in as a submodule
|-- patches/                        submodule patches applied at setup
|-- reference/                      paper PDF, project description, poster
|-- report/
|   |-- poster.tex
|   |-- poster.pdf
|   `-- figures/
|-- scripts/
|   |-- train_sac.py                vanilla SAC, --encoder {equi, cnn}
|   |-- train_sac_drq.py            DrQ-SAC
|   |-- train_sac_rad.py            RAD-SAC
|   |-- train_sac_ferm.py           FERM-SAC, CNN-only
|   |-- train_dqn.py                DQN + DrQ + RAD + CURL
|   |-- plot_results.py             Fig 6 and Fig 7 learning-curve PDFs
|   `-- render_env_panel.py         poster env panel
|-- outputs/                        per-run logs, metrics, checkpoints (gitignored)
`-- src/so2_equi_rl/
    |-- agents/
    |   |-- base.py                 agent interface (act, update, save, load)
    |   |-- sac.py                  twin-Q SAC with entropy tuning
    |   |-- sac_drq.py              DrQ-SAC, random +/-4 pixel shift, K/M averaging
    |   |-- sac_rad.py              RAD-SAC, random crop 142 to 128
    |   |-- sac_ferm.py             FERM-SAC, shared CNN encoder + InfoNCE, CNN-only
    |   |-- dqn.py                  vanilla DQN
    |   |-- dqn_drq.py              DrQ-DQN, shift aug with K/M averaged Bellman
    |   |-- dqn_rad.py              RAD-DQN, random crop
    |   `-- dqn_curl.py             CURL-DQN, shared conv stack + InfoNCE, CNN-only
    |-- buffers/
    |   |-- replay.py               fixed-capacity buffer with uint8 obs quantization
    |   `-- so2_aug.py              SO(2) rotation aug on push (paper Fig 7, k=4)
    |-- configs/
    |   |-- base.py                 shared training settings
    |   |-- sac.py                  SAC hyperparameters (paper Appendix F)
    |   |-- sac_drq.py              DrQ-SAC hyperparameters
    |   |-- sac_rad.py              RAD-SAC hyperparameters
    |   |-- sac_ferm.py             FERM-SAC hyperparameters
    |   |-- dqn.py                  DQN hyperparameters (paper Table 8)
    |   |-- dqn_drq.py              DrQ-DQN hyperparameters
    |   |-- dqn_rad.py              RAD-DQN hyperparameters
    |   `-- dqn_curl.py             CURL-DQN hyperparameters
    |-- envs/
    |   |-- wrapper.py              BulletArm close-loop wrapper
    |   |-- maniskill_wrapper.py    ManiSkill3 wrapper with the same API
    |   `-- maniskill_experts.py    scripted MS3 experts for warmup
    |-- networks/
    |   |-- encoders.py             equivariant and plain-CNN encoders
    |   |-- sac_heads.py            SAC actor and critic heads
    |   `-- dqn_heads.py            DQN Q-heads
    |-- trainers/
    |   |-- base.py                 rollout + update + eval + checkpoint loop
    |   |-- sac.py                  SAC trainer, expert warmup + stochastic exploration
    |   `-- dqn.py                  DQN trainer, eps-greedy + target-net sync
    `-- utils/
        |-- augmentation.py         SO(2) rotation, random_shift, random_crop
        |-- cli_args.py             dataclass to argparse bridge
        |-- logging.py              metrics, configs, checkpoints to disk
        |-- preprocessing.py        folds gripper state into the obs tensor
        `-- seeding.py              RNG seeding
```

## Results

Paper reproduction runs Fig 6 (DQN plus DrQ, RAD, CURL) and Fig 7 (SAC plus DrQ, RAD, FERM) on the three close-loop BulletArm tasks, 2 seeds each, 20k env steps. The ManiSkill3 extension runs the same variants on PickCube-v1 under the same budget.

- Fig 6 BulletArm, [outputs/plots/figure6_dqn_bulletarm.pdf](outputs/plots/figure6_dqn_bulletarm.pdf)
- Fig 7 BulletArm, [outputs/plots/figure7_sac_bulletarm.pdf](outputs/plots/figure7_sac_bulletarm.pdf)
- MS3 extension, [outputs/plots/figure6_dqn_maniskill.pdf](outputs/plots/figure6_dqn_maniskill.pdf), [outputs/plots/figure7_sac_maniskill.pdf](outputs/plots/figure7_sac_maniskill.pdf)
- Environment panel, [report/figures/env_panel.png](report/figures/env_panel.png)
- Full writeup and poster, [report/poster.pdf](report/poster.pdf)

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
