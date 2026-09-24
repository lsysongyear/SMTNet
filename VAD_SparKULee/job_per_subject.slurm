#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPUA800
#SBATCH -J vad_spk_per_subj
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# Train VAD model on a single SparKULee subject from scratch.
# Each subject: 1 random test trial + 1 random val trial, rest train.
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUBJECT_ID=$(printf "%03d" $((TASK_ID + 1)))
SPLIT_SEED="${SPLIT_SEED:-42}"
LOG_DIR="log/per_sub/per_subject"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sub-${SUBJECT_ID}_${TIMESTAMP}.log"

echo "============================================"
echo " SparKULee VAD - Per-subject (per_sub)"
echo " Subject:     $SUBJECT_ID"
echo " Split seed:  $SPLIT_SEED"
echo " Log:         $LOG_FILE"
echo "============================================"

$PYTHON speech_code/train.py \
  --config=configs/speech/my_run/config.yaml \
  --search-space=configs/speech/my_run/search-space.yaml \
  --holdout=False \
  --run-index=0 \
  --split-strategy per_sub \
  --fold 0 \
  --split-seed "$SPLIT_SEED" \
  --single-subject-id "$SUBJECT_ID" \
  > "$LOG_FILE" 2>&1

echo "Job finished. Log: $LOG_FILE"
