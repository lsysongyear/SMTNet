#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J vad_ft_5t5v
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# Fine-tune with per_sub_5t5v split: same test/val/train stories as pretrain.
# PRETRAINED_CKPT env var must point to the pretrained .ckpt file.
# Each array task fine-tunes one subject (01-25).
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

if [ -z "${CKPT_PATH_FILE:-}" ] || [ ! -f "$CKPT_PATH_FILE" ]; then
    echo "ERROR: CKPT_PATH_FILE is required"
    exit 1
fi
PRETRAINED_CKPT=$(cat "$CKPT_PATH_FILE")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUBJECT_ID=$(printf "%02d" $((TASK_ID + 1)))
FINETUNE_LR="${FINETUNE_LR:-0.0005}"
SPLIT_SEED="${SPLIT_SEED:-2026}"
FOLD="${FOLD:-0}"
LOG_DIR="log/per_sub_5t5v/finetune"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sub-${SUBJECT_ID}_${TIMESTAMP}.log"

echo "============================================"
echo " Fine-tune (per_sub_5t5v)"
echo " Subject: $SUBJECT_ID  |  Fold: $FOLD  |  Seed: $SPLIT_SEED"
echo " LR: $FINETUNE_LR"
echo " Pretrained: $PRETRAINED_CKPT"
echo " Log: $LOG_FILE"
echo "============================================"

$PYTHON speech_code/train.py \
  --config=configs/speech/my_run/config.yaml \
  --search-space=configs/speech/my_run/search-space.yaml \
  --holdout=False \
  --run-index=0 \
  --split-strategy per_sub_5t5v \
  --fold "$FOLD" \
  --split-seed "$SPLIT_SEED" \
  --single-subject-id "$SUBJECT_ID" \
  --pretrained-ckpt "$PRETRAINED_CKPT" \
  --finetune-lr "$FINETUNE_LR" \
  > "$LOG_FILE" 2>&1

echo "Done. Log: $LOG_FILE"
