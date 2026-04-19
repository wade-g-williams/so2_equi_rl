"""FERM-SAC config. Adds the CURL-style contrastive-loss knobs on top of
SACConfig. Two tau fields: cfg.tau (slow) for the critic target,
curl_tau (fast) for the key encoder. Aug is SO(2) rotations not crops,
so DrQ/RAD/FERM stay comparable in ablations.
"""

from dataclasses import dataclass
from typing import Optional

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACFERMConfig(SACConfig):
    # Faster than cfg.tau because the key side trains on a direct InfoNCE
    # loss, not a bootstrapped TD target. Matches CURL and FERM.
    curl_tau: float = 0.05

    # Weight on the InfoNCE loss inside the encoder_optim step. Redundant
    # with encoder_lr in theory; kept as a separate knob so ablations can
    # hold LR fixed and vary the loss contribution.
    curl_lambda: float = 1.0

    # Temperature divisor on the (B, B) logit matrix. CURL paper uses 1.0.
    curl_temperature: float = 1.0

    # LR for the encoder + W projection. Split from critic_lr because the
    # InfoNCE and TD gradients have different scales.
    encoder_lr: float = 1e-3

    # 'continuous' = uniform SO(2). 'discrete_cN' = the C_N lattice.
    # Continuous gives more diverse negatives.
    ferm_aug_mode: str = "continuous"

    # None resolves to cfg.group_order in __post_init__. Unused in continuous mode.
    ferm_group_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ferm_group_order is None:
            self.ferm_group_order = self.group_order
