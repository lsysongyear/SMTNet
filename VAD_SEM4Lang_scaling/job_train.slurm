#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J sem4lang_vad
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# SEM4Lang MEG VAD training (per_sub_5t5v split, 5 test + 5 val).
# Each array task = one model/search-space configuration.
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

SPLIT_SEED="${SPLIT_SEED:-42}"
RUN_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LOG_DIR="log/train"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_LABEL="${SLURM_JOB_NAME:-sem4lang_vad}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="${LOG_DIR}/${JOB_LABEL}_${JOB_ID}_run${RUN_INDEX}_${TIMESTAMP}.log"

echo "============================================"
echo " SEM4Lang MEG VAD Training"
echo " Split strategy: per_sub_5t5v (5 val + 5 test)"
echo " Split seed: $SPLIT_SEED"
echo " Run index:  $RUN_INDEX"
echo " Log:        $LOG_FILE"
echo "============================================"

CONFIG="${CONFIG_FILE:-configs/speech/my_run/config.yaml}"
SEARCH_SPACE="${SEARCH_SPACE_FILE:-configs/speech/my_run/search-space.yaml}"

$PYTHON speech_code/train_v1.py \
  --config="$CONFIG" \
  --search-space="$SEARCH_SPACE" \
  --holdout=False \
  --run-index="$RUN_INDEX" \
  --split-strategy per_sub_5t5v \
  --fold 0 \
  --split-seed "$SPLIT_SEED" \
  --pretrain \
  > "$LOG_FILE" 2>&1

echo "Done. Log: $LOG_FILE"
