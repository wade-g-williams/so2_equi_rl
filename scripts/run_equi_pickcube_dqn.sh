#!/bin/bash
# Wade's half of the EquiPickCube-v1 paper-extension runs.
# 6 DQN cells, seeds 0 and 1, num_envs=5 (matches Joey's reference setup).
# Joey is running the SAC variants separately.
#
# Cells:
#   dqn_equi   s0, s1
#   dqn_cnn    s0, s1
#   dqn_drq    s0, s1
# (FERM and CURL intentionally excluded per the paper-extension scope.)
#
# Prerequisites:
#   - ms3_equi conda env exists (has gymnasium + mani_skill3 + sapien)
#   - No matrix launcher running that would contend for the GPU
#   - EquiPickCube-v1 registered (import side-effect from maniskill_wrapper.py)
#
# Runtime estimate: num_envs=5 + 20k steps should be ~10-15 min per cell,
# so ~60-90 min serial total on a single GPU.

set -u

# Conda env holding torch + mani_skill + sapien + the repo (editable).
# Laptop: ms3_equi (separate env). Helper: equi_rl (mani_skill installed on top).
# Override with:  CONDA_ENV=equi_rl ./scripts/run_equi_pickcube_dqn.sh
CONDA_ENV=${CONDA_ENV:-ms3_equi}

MATRIX_DIR=/home/wadewilliams/Dev/so2_equi_rl/outputs/matrix
MARKER_DIR=$MATRIX_DIR/.markers
REPO=/home/wadewilliams/Dev/so2_equi_rl
mkdir -p "$MARKER_DIR"

cd "$REPO"

# ---- Zero-interference safeguards (helper01, shared with sakshi) --------
# Mirrors /tmp/cell1.sh's proven isolation setup:
#   - GPU 1 only (driver can't see GPU 0).
#   - Cores 24-31 only (sakshi keeps 0-23 uncontested).
#   - SCHED_IDLE via chrt (runs only when CPU is otherwise idle).
#   - Idle I/O class (disk I/O only when no one else needs it).
#   - BLAS thread pool capped at 2.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# Process-wrap prefix applied to every training launch.
ISOLATE=(taskset -c 24-31 chrt --idle 0 ionice -c 3)

# Safety: don't clash with a running matrix launcher.
if ps -eo pid,cmd | grep -E "python -m scripts\.launch_matrix|scripts/launch_matrix" | grep -v grep >/dev/null; then
  echo "[equi-dqn] matrix launcher still running, exit to avoid GPU contention"
  exit 1
fi

echo "[equi-dqn] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  OMP/MKL=$OMP_NUM_THREADS  isolate='${ISOLATE[*]}'"

run_cell() {
  local tag=$1
  local network=$2     # equi / cnn / drq
  local seed=$3

  local marker="$MARKER_DIR/${tag}.done"
  if [ -f "$marker" ]; then
    echo "[equi-dqn] skip $tag (marker exists)"
    return
  fi
  local log_dir="$MATRIX_DIR/$tag"
  mkdir -p "$log_dir"
  local log="$log_dir/launch.log"
  echo "[equi-dqn] $(date +%H:%M:%S) running $tag"

  "${ISOLATE[@]}" conda run -n "$CONDA_ENV" python -m scripts.train_dqn \
    --env-name EquiPickCube-v1 --env-backend maniskill \
    --network "$network" --num-envs 5 --seed "$seed" --total-steps 20000 \
    --output-dir outputs/matrix --run-name "$tag" \
    2>&1 | tee -a "$log"

  local rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then
    touch "$marker"
    echo "[equi-dqn] $(date +%H:%M:%S) DONE $tag"
  else
    echo "[equi-dqn] $(date +%H:%M:%S) FAILED $tag (exit=$rc, log=$log)"
  fi
}

# Seed-major ordering so a partial run still gives us seed-0 across all 3
# variants before starting seed 1. Better than variant-major if we time out.
for seed in 0 1; do
  run_cell maniskill_EquiPickCube-v1_dqn_equi_s${seed}    equi "$seed"
  run_cell maniskill_EquiPickCube-v1_dqn_cnn_s${seed}     cnn  "$seed"
  run_cell maniskill_EquiPickCube-v1_dqn_drq_cnn_s${seed} drq  "$seed"
done

echo "[equi-dqn] run complete at $(date +%H:%M:%S)"
