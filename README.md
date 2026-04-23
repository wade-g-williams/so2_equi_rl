# so2_equi_rl

Reimplementation of Wang et al., *SO(2)-Equivariant Reinforcement Learning* (ICLR 2022), on the three close-loop BulletArm tasks from the paper, plus a ManiSkill 3 SAC extension on PickCube-v1. CS 5180 final project, Spring 2026.

## Setup

Two conda envs, one per simulator. BulletArm needs Python 3.7 to match the paper, ManiSkill 3 needs 3.10. The BulletArm sim is vendored as a submodule.

```bash
git clone --recursive https://github.com/wade-g-williams/so2_equi_rl.git
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

ManiSkill 3 env:

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
- `train_sac_rad.py`, RAD-SAC (implemented, not in final results)
- `train_sac_ferm.py`, FERM-SAC, CNN hardcoded. InfoNCE runs concurrently with SAC; the 1600-step contrastive pretrain used in the paper was dropped for scope, so FERM here underperforms the paper's FERM on block_pulling
- `train_dqn.py`, DQN plus DrQ, RAD, and CURL variants, selected via `--network {equi, cnn, drq, rad, curl}`. RAD is implemented but not in final results

Paper reproduction scope:

- SAC was evaluated on `close_loop_block_pulling`, `close_loop_block_picking`, and `close_loop_drawer_opening`
- DQN was evaluated on `close_loop_block_pulling`, `close_loop_household_picking` (the paper's object-picking benchmark), and `close_loop_drawer_opening`

Both at 20,000 env steps with two seeds per cell, except DQN on drawer_opening which has a single seed. Block_picking was used in place of household_picking for SAC because the SAC sweep was locked in before the household-object runs were planned.

The ManiSkill 3 extension runs Equi SAC and CNN SAC on `EquiPickCube-v1`, a subclassed PickCube-v1 that hides the Panda arm from the depth observation so the scene is rotation-symmetric. Run with `--env-backend maniskill` in the `ms3_equi` env. Four trials per variant at 100,000 steps.

The BulletArm wrapper supports more tasks than these three, see [envs/wrapper.py](src/so2_equi_rl/envs/wrapper.py). Reproduction sticks to the paper's.

Each run writes a timestamped directory under `outputs/` with the resolved config, a `metrics.jsonl` eval log, TensorBoard events, and checkpoints. Resume with `--resume outputs/<run>/ckpts/latest.pt`.

### Production matrix

[scripts/launch_matrix.py](scripts/launch_matrix.py) expands [scripts/matrix.yaml](scripts/matrix.yaml) into one training cell per (algo, model, task, backend, seed), gates each launch on free VRAM, and runs them sequentially or through a worker pool. Each cell is wrapped in `conda run -n <env>` so one invocation dispatches both backends.

```bash
python -m scripts.launch_matrix --config scripts/matrix.yaml --output-dir outputs/matrix
```

Add `--parallel N` for N concurrent workers, `--dry-run` to print the command for each cell without launching, and `--vram-override MIB` to force a per-job budget. Cells that have already completed leave a done-marker at `outputs/matrix/.markers/<run_tag>.done`; re-running the launcher skips them, so resumes are idempotent.

### Manual sweep (BulletArm, all variants, both seeds)

```bash
conda activate equi_rl

for task in close_loop_block_picking close_loop_block_pulling close_loop_drawer_opening; do
  for seed in 0 1; do
    for enc in equi cnn; do
      python scripts/train_sac.py --encoder $enc --env-name $task \
          --env-backend bulletarm --seed $seed --total-steps 20000 \
          --num-processes 5 --run-name sac_${enc}_${task}_s$seed
    done
    python scripts/train_sac_drq.py --encoder cnn --env-name $task \
        --env-backend bulletarm --seed $seed --total-steps 20000 \
        --num-processes 5 --run-name sac_drq_cnn_${task}_s$seed
    python scripts/train_sac_ferm.py --env-name $task \
        --env-backend bulletarm --seed $seed --total-steps 20000 \
        --num-processes 5 --run-name sac_ferm_${task}_s$seed
  done
done

for task in close_loop_household_picking close_loop_block_pulling close_loop_drawer_opening; do
  for seed in 0 1; do
    for net in equi cnn drq curl; do
      python scripts/train_dqn.py --network $net --env-name $task \
          --env-backend bulletarm --seed $seed --total-steps 20000 \
          --num-processes 5 --run-name dqn_${net}_${task}_s$seed
    done
  done
done
```

ManiSkill 3 loop on EquiPickCube-v1, 4 seeds, 100k steps:

```bash
conda activate ms3_equi

for seed in 0 1 2 3; do
  for enc in equi cnn; do
    python scripts/train_sac.py --encoder $enc --env-name EquiPickCube-v1 \
        --env-backend maniskill --seed $seed --total-steps 100000 \
        --num-envs 32 --run-name sac_${enc}_pickcube_s$seed
  done
done
```

### Plots + environment panel

```bash
python scripts/plot_results.py --roots outputs --out outputs/plots
python scripts/render_env_panel.py
```

`plot_results.py` aggregates by (alg family, backend), averages seeds, and writes learning-curve PDFs. `render_env_panel.py` produces the three-task PyBullet panel used in the report.

## Layout

```text
so2_equi_rl/
|-- environment.yml
|-- environment_ms3.yml
|-- pyproject.toml
|-- helping_hands_rl_envs/
|-- patches/
|-- reference/
|   `-- wang_2022_so2_equivariant_rl.pdf
|-- report/
|   |-- final_report.tex
|   |-- final_report.pdf
|   |-- references.bib
|   |-- aaai2026.sty
|   |-- aaai2026.bst
|   `-- figures/
|-- scripts/
|   |-- train_sac.py
|   |-- train_sac_drq.py
|   |-- train_sac_rad.py
|   |-- train_sac_ferm.py
|   |-- train_dqn.py
|   |-- launch_matrix.py
|   |-- matrix.yaml
|   |-- plot_results.py
|   `-- render_env_panel.py
|-- outputs/           # training artifacts, gitignored
`-- src/so2_equi_rl/
    |-- launch.py
    |-- agents/
    |-- buffers/
    |-- configs/
    |-- envs/
    |-- networks/
    |-- trainers/
    `-- utils/
```

## Results

Final figures live in `report/figures/`; `final_report.pdf` in the same directory is the full writeup.

- Fig 1, SAC on BulletArm, [report/figures/fig1_sac_bulletarm.pdf](report/figures/fig1_sac_bulletarm.pdf)
- Fig 2, DQN on BulletArm, [report/figures/fig2_dqn_bulletarm.pdf](report/figures/fig2_dqn_bulletarm.pdf)
- Fig 3, SAC on ManiSkill 3 EquiPickCube-v1, [report/figures/maniskill_avg_returns.png](report/figures/maniskill_avg_returns.png)
- Environment panel, [report/figures/env_panel_screenshot.png](report/figures/env_panel_screenshot.png)
- Writeup, [report/final_report.pdf](report/final_report.pdf)

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439). A local copy of the paper lives at [reference/wang_2022_so2_equivariant_rl.pdf](reference/wang_2022_so2_equivariant_rl.pdf).
