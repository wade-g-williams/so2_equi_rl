"""FERM-SAC config. CURL contrastive knobs on top of SACConfig. Paper sec
E: random crop 142x142 -> 128x128 for the InfoNCE views; two tau fields so
the key encoder can chase faster than the critic target.

Paper sec E also pretrains the contrastive encoder for 1.6k steps on
expert data before SAC starts. NOT yet implemented; expected gap from
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

    # Paper sec E: random crop 142x142 -> 128x128, implemented as pad then
    # random-crop-to-128 with pad=7 on both sides (128 + 14 = 142).
    ferm_pad: int = 7

    # Contrastive latent dim. Paper sec E: size 50 (per Zhan et al. 2020).
    # Encoder output (n_hidden=128 CNN, n_hidden*group_order=1024 Equi)
    # projects down to z_dim before the bilinear product. Earlier bug had
    # W at encoder_dim x encoder_dim, giving FERM too much capacity.
    z_dim: int = 50
