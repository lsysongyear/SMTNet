#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J vad_spk_pretrain
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# Pretrain + Per-subject eval on SparKULee (random_trial split).
# ============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
test -f speech_code/train_v1.py || { echo "Missing release training source" >&2; exit 1; }

SPLIT_SEED="${SPLIT_SEED:-84}"
RUN_IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_RUNS="${N_RUNS:-?}"
MODEL_TAG="${MODEL_TAG:-model}"
LOG_DIR="log/per_sub/pretrain/${MODEL_TAG}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run${RUN_IDX}_${TIMESTAMP}.log"

# Read subjects from file if provided
PRETRAIN_SUBJECTS=""
if [ -n "${SUBJECTS_FILE:-}" ] && [ -f "$SUBJECTS_FILE" ]; then
    PRETRAIN_SUBJECTS=$(cat "$SUBJECTS_FILE")
fi

# Use snapshot configs if provided, else defaults
CONFIG="${CONFIG_FILE:?CONFIG_FILE is required}"
SEARCH_SPACE="${SEARCH_SPACE_FILE:?SEARCH_SPACE_FILE is required}"

echo "============================================"
echo " SparKULee VAD - Pretrain + Per-Subject Eval"
echo " Run:         $RUN_IDX / $N_RUNS"
echo " Subjects:    ${PRETRAIN_SUBJECTS:-ALL 85}"
echo " Config:      $CONFIG"
echo " Log:         $LOG_FILE"
echo "============================================"

CMD="$PYTHON speech_code/train_v1.py \
  --config=$CONFIG \
  --search-space=$SEARCH_SPACE \
  --holdout=False \
  --run-index $RUN_IDX \
  --split-strategy per_sub \
  --fold 0 \
  --split-seed $SPLIT_SEED \
  --pretrain"

if [ -n "$PRETRAIN_SUBJECTS" ]; then
    CMD="$CMD --pretrain-subjects $PRETRAIN_SUBJECTS"
fi

eval $CMD > "$LOG_FILE" 2>&1

echo "Job finished. Log: $LOG_FILE"
