#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPUA800
#SBATCH -J vad_spk_cv
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# 5-fold cross-subject training for SparKULee VAD.
# 85 subjects are split into 5 folds; each array task uses 17 subjects as test.
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

RUN_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
LOG_DIR="log/cross_subject"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/fold${RUN_IDX}_${TIMESTAMP}.log"

echo "Dataset: SparKULee | Strategy: cross_subject | Fold: $RUN_IDX | Seed: $SPLIT_SEED"
echo "Log: $LOG_FILE"

$PYTHON speech_code/train.py \
  --config=configs/speech/my_run/config.yaml \
  --search-space=configs/speech/my_run/search-space.yaml \
  --holdout=False \
  --run-index=0 \
  --split-strategy cross_subject \
  --fold "$RUN_IDX" \
  --split-seed "$SPLIT_SEED" \
  > "$LOG_FILE" 2>&1

echo "Job finished. Log: $LOG_FILE"
