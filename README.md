# so2_equi_rl

Scratch implementation of Wang et al., *SO(2)-Equivariant Reinforcement Learning* (ICLR 2022), for the close-loop BulletArm tasks. CS 5180 final project, Spring 2026.

## Setup

Clone with submodules:

```bash
git clone --recursive <repo-url> so2_equi_rl
cd so2_equi_rl
```

Create the conda env and install both packages editable:

```bash
conda env create -f environment.yml
conda activate equi_rl
uv pip install -e .
uv pip install -e helping_hands_rl_envs
```

## Reference

Wang, Walters, Platt. "SO(2)-Equivariant Reinforcement Learning." ICLR 2022. [arXiv:2203.04439](https://arxiv.org/abs/2203.04439)
