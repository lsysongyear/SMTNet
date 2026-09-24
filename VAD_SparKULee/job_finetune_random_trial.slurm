#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPUA800
#SBATCH -J vad_spk_finetune
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# Fine-tune VAD model on a single SparKULee subject (per_sub split).
# Reads pretrained checkpoint path from CKPT_PATH_FILE.
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

if [ -z "$CKPT_PATH_FILE" ]; then
    echo "ERROR: CKPT_PATH_FILE env var is required."
    exit 1
fi
if [ ! -f "$CKPT_PATH_FILE" ]; then
    echo "ERROR: CKPT_PATH_FILE not found: $CKPT_PATH_FILE"
    exit 1
fi

PRETRAINED_CKPT=$(cat "$CKPT_PATH_FILE")
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Checkpoint file does not exist: $PRETRAINED_CKPT"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUBJECT_ID=$(printf "%03d" $((TASK_ID + 1)))     # 001 .. 085
FINETUNE_LR="${FINETUNE_LR:-0.0001}"
SPLIT_SEED="${SPLIT_SEED:-42}"
LOG_DIR="log/per_sub/finetune"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sub-${SUBJECT_ID}_${TIMESTAMP}.log"

echo "============================================"
echo " SparKULee VAD - Fine-tune (per_sub)"
echo " Subject:     $SUBJECT_ID"
echo " Split seed:  $SPLIT_SEED"
echo " Finetune LR: $FINETUNE_LR"
echo " Pretrained:  $PRETRAINED_CKPT"
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
  --pretrained-ckpt "$PRETRAINED_CKPT" \
  --finetune-lr "$FINETUNE_LR" \
  > "$LOG_FILE" 2>&1

echo "Job finished. Log: $LOG_FILE"
