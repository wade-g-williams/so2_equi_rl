#!/bin/bash
# Runs the ManiSkill3 extension on PickCube-v1, PullCube-v1, and StackCube-v1 across all
# SAC and DQN variants, seeds 0 through 3. FERM does its 1600-step InfoNCE
# pretrain before SAC training for each (task, seed).
#
# Usage from the repo root:
#   conda activate ms3_equi
#   bash scripts/run_ms3_extension.sh

set -eu

TASKS=(PickCube-v1 PullCube-v1 StackCube-v1)
SEEDS=(0 1 2 3)
TOTAL_STEPS=20000
NUM_ENVS=32

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # SAC vanilla, both encoders
    for enc in equi cnn; do
      python scripts/train_sac.py \
        --encoder "$enc" --env-name "$task" --env-backend maniskill \
        --seed "$seed" --total-steps "$TOTAL_STEPS" --num-envs "$NUM_ENVS" \
        --run-name "sac_${enc}_${task}_s${seed}"
    done

    # SAC DrQ
    python scripts/train_sac_drq.py \
      --encoder cnn --env-name "$task" --env-backend maniskill \
      --seed "$seed" --total-steps "$TOTAL_STEPS" --num-envs "$NUM_ENVS" \
      --run-name "sac_drq_cnn_${task}_s${seed}"

    # SAC RAD
    python scripts/train_sac_rad.py \
      --encoder cnn --env-name "$task" --env-backend maniskill \
      --seed "$seed" --total-steps "$TOTAL_STEPS" --num-envs "$NUM_ENVS" \
      --run-name "sac_rad_cnn_${task}_s${seed}"

    # FERM pretrain then train
    python scripts/pretrain_ferm.py \
      --env-name "$task" --env-backend maniskill \
      --seed "$seed" --num-envs "$NUM_ENVS" \
      --run-name "ferm_pretrain_${task}_s${seed}"

    ENC=$(ls -td outputs/*_ferm_pretrain_${task}_s${seed}/ckpts/pretrained_encoder.pt | head -1)

    python scripts/train_sac_ferm.py \
      --env-name "$task" --env-backend maniskill \
      --seed "$seed" --total-steps "$TOTAL_STEPS" --num-envs "$NUM_ENVS" \
      --pretrained-encoder "$ENC" \
      --run-name "sac_ferm_${task}_s${seed}"

    # DQN plus DrQ, RAD, CURL
    for net in equi cnn drq rad curl; do
      python scripts/train_dqn.py \
        --network "$net" --env-name "$task" --env-backend maniskill \
        --seed "$seed" --total-steps "$TOTAL_STEPS" --num-envs "$NUM_ENVS" \
        --run-name "dqn_${net}_${task}_s${seed}"
    done
  done
done
