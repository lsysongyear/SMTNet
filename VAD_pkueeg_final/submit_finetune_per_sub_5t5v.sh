#!/bin/bash
# ============================================================================
# Submit fine-tune jobs with per_sub_5t5v split.
#
# Usage:
#   # Auto-discover from pretrain output:
#   bash submit_finetune_per_sub_5t5v.sh --pretrained-run <run_name>
#
#   # Point directly to checkpoint:
#   bash submit_finetune_per_sub_5t5v.sh --pretrained-ckpt <path/to/checkpoint.ckpt>
#
#   # Subset of subjects, custom LR:
#   bash submit_finetune_per_sub_5t5v.sh --pretrained-run <name> --subjects 1-10 --lr 0.0001
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULT_ROOT="results/speech-detection/model:brain_magic_speech_v4-dataset:eeg_speech_v1/split:per_sub_5t5v"

PRETRAINED_CKPT="results/speech-detection/model:brain_magic_speech_v4-dataset:eeg_speech_v1/split:per_sub_5t5v/input_dim-57_attention_dim-128_output_dim-1_kernel_size-25_depthwise_kernel-30_num_blocks1-9_num_blocks2-15_dropout-0.1_n_subjects-25_weight-[1.0, 1.0]_loss_type-mse_lr-0.001/pretrain/run0:seed1_fold0/checkpoints/model-epoch=07-val_loss=0.1675.ckpt"
PRETRAINED_RUN=""
SPLIT_SEED="545"
FINETUNE_LR="0.0005"
SUBJECT_RANGE="1-25"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained-ckpt) PRETRAINED_CKPT="$2"; shift 2 ;;
        --pretrained-run)  PRETRAINED_RUN="$2"; shift 2 ;;
        --seed) SPLIT_SEED="$2"; shift 2 ;;
        --lr) FINETUNE_LR="$2"; shift 2 ;;
        --subjects) SUBJECT_RANGE="$2"; shift 2 ;;
        *) echo "Unknown: $1"
           echo "Usage: bash submit_finetune_per_sub_5t5v.sh [--pretrained-ckpt <path> | --pretrained-run <name>] [--subjects 1-25] [--lr 5e-4] [--seed 2026]"
           exit 1 ;;
    esac
done

# Resolve checkpoint
if [ -z "$PRETRAINED_CKPT" ] && [ -z "$PRETRAINED_RUN" ]; then
    echo "ERROR: Specify --pretrained-ckpt or --pretrained-run"
    exit 1
fi
if [ -z "$PRETRAINED_CKPT" ]; then
    CKPT_TXT="${RESULT_ROOT}/${PRETRAINED_RUN}/pretrain/best_ckpt_fold0.txt"
    if [ ! -f "$CKPT_TXT" ]; then
        echo "ERROR: $CKPT_TXT not found"
        exit 1
    fi
    PRETRAINED_CKPT=$(cat "$CKPT_TXT")
    echo "Auto-discovered: $PRETRAINED_CKPT"
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

# Convert 1-25 to 0-24 array
START=$(echo "$SUBJECT_RANGE" | cut -d'-' -f1)
END=$(echo "$SUBJECT_RANGE" | cut -d'-' -f2)
ARRAY_RANGE="$((START-1))-$((END-1))"
N=$((END - START + 1))

echo "============================================"
echo " Fine-tune (per_sub_5t5v)"
echo " Subjects:  $SUBJECT_RANGE ($N subjects)"
echo " Array:     $ARRAY_RANGE"
echo " Seed:      $SPLIT_SEED"
echo " LR:        $FINETUNE_LR"
echo " Pretrain:  $PRETRAINED_CKPT"
echo "============================================"

mkdir -p log/per_sub_5t5v/finetune

# Write checkpoint path to file to avoid shell escaping issues
CKPT_PATH_FILE="log/per_sub_5t5v/finetune/ckpt_path.txt"
echo "$PRETRAINED_CKPT" > "$CKPT_PATH_FILE"

sbatch \
    --export="ALL,SPLIT_SEED=$SPLIT_SEED,FINETUNE_LR=$FINETUNE_LR,CKPT_PATH_FILE=$CKPT_PATH_FILE" \
    --array="$ARRAY_RANGE" \
    job_finetune_per_sub_5t5v.slurm

echo "Submitted. Logs: log/per_sub_5t5v/finetune/"
