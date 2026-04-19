"""FERM-SAC config. Adds the CURL-style contrastive-loss knobs on top of
SACConfig and leaves every other SAC hyperparameter untouched so the
baseline and the contrastive variant share one set of defaults.

Two separate tau fields: cfg.tau for the critic-target Polyak (slow,
0.005) and curl_tau for the key-encoder Polyak (fast, 0.05). The key
encoder targets a direct contrastive loss, not a bootstrapped TD target,
so it can chase the online encoder faster without blowing up.

Aug is SO(2) rotations, not random crops. The task has workspace
rotation symmetry and the rest of the codebase (DrQ, RAD) uses rotations
too, so cross-variant comparisons stay controlled.
"""

from dataclasses import dataclass
from typing import Optional

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACFERMConfig(SACConfig):
    # Key-encoder Polyak rate. Faster than cfg.tau because the key side
    # trains on a direct InfoNCE loss, not a bootstrapped TD target.
    # Matches CURL/FERM.
    curl_tau: float = 0.05

    # Weight on the InfoNCE loss inside the encoder_optim step. Redundant
    # with encoder_lr in theory, but kept as an explicit knob so ablations
    # can hold LR fixed and just vary the loss contribution.
    curl_lambda: float = 1.0

    # Temperature divisor on the (B, B) logit matrix before cross-entropy.
    # CURL paper uses 1.0 (plain dot products); kept tunable for ablations.
    curl_temperature: float = 1.0

    # LR for the dedicated encoder optimizer (q_encoder + W projection).
    # Split from critic_lr because the two losses have different gradient
    # scales and shouldn't share a step size. Default matches critic_lr.
    encoder_lr: float = 1e-3

    # Aug distribution for the two InfoNCE views. 'continuous' = uniform
    # SO(2); 'discrete_cN' = the C_N lattice. Continuous is richer so the
    # negatives are more diverse; default there.
    ferm_aug_mode: str = "continuous"

    # None resolves to cfg.group_order in __post_init__ so one knob drives
    # both the encoder's equivariance group and the FERM aug lattice when
    # the user picks discrete_cN. Unused in continuous mode.
    ferm_group_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ferm_group_order is None:
            self.ferm_group_order = self.group_order
