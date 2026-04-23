#!/bin/bash
# Seed-1 DQN BulletArm reruns on block_pulling + drawer_opening.
# 10 cells total: 5 network variants (equi, cnn, drq, rad, curl) × 2 tasks.
#
# Designed to run on helper01 shared with sakshi's diffusion-transformer job.
# Isolation pattern mirrors /tmp/cell1.sh:
#   - GPU 1 only (driver can't see GPU 0)
#   - Cores 24-31 only, SCHED_IDLE, idle I/O
#   - BLAS threads capped at 2
#
# Runtime estimate: ~15-20 min per BulletArm DQN cell at 20k steps.
# 10 cells serial ≈ 2.5-3.5 hours.
#
# Override conda env with:  CONDA_ENV=<name> ./scripts/run_bulletarm_dqn_s1.sh

set -u

CONDA_ENV=${CONDA_ENV:-equi_rl}
REPO=/home/wadewilliams/Dev/so2_equi_rl
OUT_DIR=$REPO/outputs
MARKER_DIR=$OUT_DIR/.markers_bulletarm_s1
mkdir -p "$MARKER_DIR"

cd "$REPO"

# ---- Zero-interference safeguards (helper01, shared with sakshi) --------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
ISOLATE=(taskset -c 24-31 chrt --idle 0 ionice -c 3)

# Safety: don't clash with a running matrix launcher.
if ps -eo pid,cmd | grep -E "python -m scripts\.launch_matrix|scripts/launch_matrix" | grep -v grep >/dev/null; then
  echo "[bulletarm-s1] matrix launcher still running, exit to avoid GPU contention"
  exit 1
fi

echo "[bulletarm-s1] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  OMP/MKL=$OMP_NUM_THREADS  env=$CONDA_ENV  isolate='${ISOLATE[*]}'"

run_cell() {
  local task=$1
  local network=$2
  local tag="${network}_dqn_${task}_s1"

  local marker="$MARKER_DIR/${tag}.done"
  if [ -f "$marker" ]; then
    echo "[bulletarm-s1] skip $tag (marker exists)"
    return
  fi
  local log_dir="$OUT_DIR/$tag"
  mkdir -p "$log_dir"
  local log="$log_dir/launch.log"
  echo "[bulletarm-s1] $(date +%H:%M:%S) running $tag"

  "${ISOLATE[@]}" conda run -n "$CONDA_ENV" python -m scripts.train_dqn \
    --network "$network" --env-name "close_loop_${task}" \
    --env-backend bulletarm --num-processes 5 --seed 1 --total-steps 20000 \
    --output-dir outputs --run-name "$tag" \
    2>&1 | tee -a "$log"

  local rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then
    touch "$marker"
    echo "[bulletarm-s1] $(date +%H:%M:%S) DONE $tag"
  else
    echo "[bulletarm-s1] $(date +%H:%M:%S) FAILED $tag (exit=$rc, log=$log)"
  fi
}

# Task-major ordering — finish all variants on one task before moving to
# the next. A partial run then gives complete coverage of one task rather
# than partial coverage of both.
for task in block_pulling drawer_opening; do
  for network in equi cnn drq rad curl; do
    run_cell "$task" "$network"
  done
done

echo "[bulletarm-s1] run complete at $(date +%H:%M:%S)"
