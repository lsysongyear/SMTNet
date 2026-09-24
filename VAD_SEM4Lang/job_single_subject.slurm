#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPUA800
#SBATCH -J sem4lang_ss
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# SEM4Lang per-subject training (single_subject split, 10 test + 5 val).
# Each array task trains one subject (01-12).
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUBJECT_ID=$(printf "%02d" $((TASK_ID + 1)))
FOLD="${FOLD:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
LOG_DIR="log/single_subject"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sub-${SUBJECT_ID}_fold${FOLD}_${TIMESTAMP}.log"

echo "============================================"
echo " SEM4Lang Per-Subject Training"
echo " Subject:  $SUBJECT_ID"
echo " Fold:     $FOLD"
echo " Seed:     $SPLIT_SEED"
echo " Log:      $LOG_FILE"
echo "============================================"

$PYTHON speech_code/train.py \
  --config=configs/speech/my_run/config.yaml \
  --search-space=configs/speech/my_run/search-space.yaml \
  --holdout=False \
  --run-index=0 \
  --split-strategy single_subject \
  --fold "$FOLD" \
  --split-seed "$SPLIT_SEED" \
  --single-subject-id "$SUBJECT_ID" \
  > "$LOG_FILE" 2>&1

echo "Done. Log: $LOG_FILE"
