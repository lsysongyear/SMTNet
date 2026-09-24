#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J vad_pt_5t5v
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# Pretrain with per_sub_5t5v split: each subject gets 5 test + 5 val randomly.
# Each array task = one hyperparameter combination (--run-index).
# ============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
test -f speech_code/train_v1.py || { echo "Missing release training source" >&2; exit 1; }

RUN_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SPLIT_SEED="${SPLIT_SEED:-545}"
FOLD="${FOLD:-0}"
N_RUNS="${N_RUNS:-?}"
LOG_DIR="log/per_sub_5t5v/pretrain"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/fold${FOLD}_run${RUN_IDX}_${TIMESTAMP}.log"

echo "============================================"
echo " Pretrain (per_sub_5t5v) - Run $RUN_IDX / $N_RUNS"
echo " Fold: $FOLD  |  Seed: $SPLIT_SEED"
echo " Subjects: ${PRETRAIN_SUBJECTS:-ALL 25}"
echo " Log: $LOG_FILE"
echo "============================================"

CONFIG="${CONFIG_FILE:-configs/speech/my_run/config.yaml}"
SEARCH_SPACE="${SEARCH_SPACE_FILE:-configs/speech/my_run/search-space.yaml}"

CMD="$PYTHON speech_code/train_v1.py \
  --config=$CONFIG \
  --search-space=$SEARCH_SPACE \
  --holdout=False \
  --run-index $RUN_IDX \
  --split-strategy per_sub_5t5v \
  --fold $FOLD \
  --split-seed $SPLIT_SEED \
  --pretrain"

if [ -n "${PRETRAIN_SUBJECTS:-}" ]; then
    CMD="$CMD --pretrain-subjects $PRETRAIN_SUBJECTS"
fi

eval $CMD > "$LOG_FILE" 2>&1
echo "Done. Log: $LOG_FILE"
