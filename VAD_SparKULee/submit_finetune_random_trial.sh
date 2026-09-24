#!/bin/bash
# ============================================================================
# Submit SparKULee fine-tune jobs with per_sub split.
#
# Usage:
#   bash submit_finetune_random_trial.sh --pretrained-run <run_name>
#   bash submit_finetune_random_trial.sh --pretrained-ckpt <path>
#   bash submit_finetune_random_trial.sh --pretrained-run <name> --subjects 1-3,5-16,18-24,26
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULT_ROOT="results/speech-detection/model:brain_magic_speech_v4-dataset:sparkulee_vad/split:per_sub"

# --- Defaults ---
PRETRAINED_CKPT="results/speech-detection/model:brain_magic_speech_v4-dataset:sparkulee_vad/split:per_sub/input_dim-64_attention_dim-128_output_dim-1_kernel_size-25_depthwise_kernel-30_num_blocks1-9_num_blocks2-15_dropout-0.1_n_subjects-85_weight-1.0_1.0_loss_type-mse_lr-0.001/pretrain/run0:seed1_fold0/checkpoints/model-epoch=04-val_loss=0.1534.ckpt"
PRETRAINED_RUN=""
SPLIT_SEED="42"
FINETUNE_LR="0.0005"
SUBJECT_RANGE="1-3,5-16,18-24,26-36"

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained-ckpt) PRETRAINED_CKPT="$2"; shift 2 ;;
        --pretrained-run)  PRETRAINED_RUN="$2"; shift 2 ;;
        --seed) SPLIT_SEED="$2"; shift 2 ;;
        --lr) FINETUNE_LR="$2"; shift 2 ;;
        --subjects) SUBJECT_RANGE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# --- Resolve pretrained checkpoint ---
if [ -z "$PRETRAINED_CKPT" ] && [ -z "$PRETRAINED_RUN" ]; then
    echo "ERROR: Specify --pretrained-ckpt or --pretrained-run"
    exit 1
fi

if [ -z "$PRETRAINED_CKPT" ]; then
    CKPT_TXT="${RESULT_ROOT}/${PRETRAINED_RUN}/pretrain/best_ckpt.txt"
    if [ ! -f "$CKPT_TXT" ]; then
        echo "ERROR: $CKPT_TXT not found"
        exit 1
    fi
    PRETRAINED_CKPT=$(cat "$CKPT_TXT")
    echo "Auto-discovered: $PRETRAINED_CKPT"
fi

if [[ "$PRETRAINED_CKPT" != /* ]]; then
    PRETRAINED_CKPT="$(pwd)/${PRETRAINED_CKPT}"
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

# --- Write checkpoint path to file ---
CKPT_PATH_FILE="log/per_sub/finetune/ckpt_path.txt"
mkdir -p "$(dirname "$CKPT_PATH_FILE")"
echo "$PRETRAINED_CKPT" > "$CKPT_PATH_FILE"

# --- Convert subject spec to SLURM array ---
ARRAY_RANGE=$($PYTHON -c "
spec = '$SUBJECT_RANGE'
ids = set()
for part in spec.split(','):
    part = part.strip()
    if '-' in part:
        lo, hi = part.split('-', 1)
        for i in range(int(lo), int(hi)+1):
            ids.add(i)
    else:
        ids.add(int(part))
ids = sorted(ids)
parts = []
i = 0
while i < len(ids):
    j = i
    while j+1 < len(ids) and ids[j+1] == ids[j] + 1:
        j += 1
    if i == j:
        parts.append(str(ids[i]-1))
    else:
        parts.append(f'{ids[i]-1}-{ids[j]-1}')
    i = j + 1
print(','.join(parts))
")

N_SUBJECTS=$($PYTHON -c "
spec = '$SUBJECT_RANGE'
ids = set()
for part in spec.split(','):
    part = part.strip()
    if '-' in part:
        lo, hi = part.split('-', 1)
        for i in range(int(lo), int(hi)+1):
            ids.add(i)
    else:
        ids.add(int(part))
print(len(ids))
")

echo "============================================"
echo " SparKULee VAD - Fine-tune (per_sub)"
echo " Subjects:       $SUBJECT_RANGE ($N_SUBJECTS subjects)"
echo " Array range:    $ARRAY_RANGE"
echo " Split seed:     $SPLIT_SEED"
echo " Finetune LR:    $FINETUNE_LR"
echo " Pretrained:     $PRETRAINED_CKPT"
echo "============================================"

sbatch \
    --export="ALL,SPLIT_SEED=$SPLIT_SEED,FINETUNE_LR=$FINETUNE_LR,CKPT_PATH_FILE=$CKPT_PATH_FILE" \
    --array="$ARRAY_RANGE" \
    job_finetune_random_trial.slurm

echo "Submitted. Logs: log/per_sub/finetune/"
