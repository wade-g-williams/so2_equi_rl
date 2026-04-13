# so2_equi_rl

SO(2)-equivariant reinforcement learning for robotic manipulation — a reimplementation of Wang et al., *"SO(2)-Equivariant Reinforcement Learning"* (ICLR 2022), built as the final project for CS 5180: Reinforcement Learning at Northeastern University (Spring 2026).


## Setup

Clone the repository with submodules:

```bash
git clone --recursive <repo-url> so2_equi_rl
cd so2_equi_rl
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate equi_rl
```

Install both packages in editable mode:

```bash
uv pip install -e .
uv pip install -e helping_hands_rl_envs
```

Verify the install:

```bash
python -c "import so2_equi_rl, helping_hands_rl_envs; print(so2_equi_rl.__version__)"
```

## Usage

To be added later.

## Reference

Wang, Dian, Robin Walters, and Robert Platt. "SO(2)-Equivariant Reinforcement Learning." *International Conference on Learning Representations (ICLR)*, 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
