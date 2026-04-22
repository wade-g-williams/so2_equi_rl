"""DQN-CURL config. Paper §E: CURL uses random crop (142x142 -> 128x128)
for the two contrastive views. Two tau fields: cfg.tau (slow) for the
DQN target net, curl_tau (fast) for the key encoder.

CNN-only. The bilinear InfoNCE head expects a flat feature vector;
equivariant DQN's GeometricTensor output doesn't plug in cleanly.
"""

from dataclasses import dataclass

from so2_equi_rl.configs.dqn import DQNConfig


@dataclass
class DQNCURLConfig(DQNConfig):
    # Faster than cfg.tau because the key side trains on a direct InfoNCE
    # loss, not a bootstrapped TD target.
    curl_tau: float = 0.05

    # Weight on the InfoNCE loss inside the encoder_optim step.
    curl_lambda: float = 1.0

    # Temperature divisor on the (B, B) logit matrix. CURL paper uses 1.0.
    curl_temperature: float = 1.0

    # LR for the encoder + W projection. Split from cfg.lr because the
    # InfoNCE and TD gradients have different scales.
    encoder_lr: float = 1e-3

    # Paper §E: random crop 142 -> 128, implemented as pad=7 + random-crop-to-128.
    curl_pad: int = 7
