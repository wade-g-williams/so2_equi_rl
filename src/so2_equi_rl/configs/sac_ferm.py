"""FERM-SAC config. Adds CURL-style contrastive-loss knobs on top of
SACConfig. Paper §E: FERM uses random crop (142x142 -> 128x128) for the
InfoNCE views. Two tau fields: cfg.tau (slow) for the critic target,
curl_tau (fast) for the key encoder.

Paper §E also pretrains the contrastive encoder for 1.6k steps on expert
data before SAC training starts. NOT yet implemented, expected gap from
paper for the Block Pulling FERM result.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.sac import SACConfig


@dataclass
class SACFERMConfig(SACConfig):
    # Faster than cfg.tau because the key side trains on a direct InfoNCE
    # loss, not a bootstrapped TD target. Matches CURL and FERM.
    curl_tau: float = 0.05

    # Weight on the InfoNCE loss inside the encoder_optim step.
    curl_lambda: float = 1.0

    # Temperature divisor on the (B, B) logit matrix. CURL paper uses 1.0.
    curl_temperature: float = 1.0

    # LR for the encoder + W projection. Split from critic_lr because the
    # InfoNCE and TD gradients have different scales.
    encoder_lr: float = 1e-3

    # Paper §E: random crop 142x142 -> 128x128, implemented as pad then
    # random-crop-to-128 with pad=7 on both sides (128 + 14 = 142).
    ferm_pad: int = 7
