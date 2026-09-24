#!/bin/bash
#SBATCH --job-name=spk_model_seed
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=gpua800n24
#SBATCH --output=log/model_seed_sweep/%x_%A_%a.out
#SBATCH --error=log/model_seed_sweep/%x_%A_%a.err

set -euo pipefail

PYTHON=${PYTHON:-python3}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
TASKS_TSV=${TASKS_TSV:?TASKS_TSV is required}
SUBJECTS_FILE=${SUBJECTS_FILE:?SUBJECTS_FILE is required}
SPLIT_SEED=${SPLIT_SEED:?SPLIT_SEED is required}
MANIFEST=${MANIFEST:?MANIFEST is required}
RUNTIME_ROOT=${RUNTIME_ROOT:?RUNTIME_ROOT is required}
SUBJECTS=$(<"$SUBJECTS_FILE")

export PYTHONDONTWRITEBYTECODE=1
cd "$RUNTIME_ROOT"
"$PYTHON" model_seed_sweep.py verify-runtime --manifest "$MANIFEST"
line=$(sed -n "$((TASK_ID + 1))p" "$TASKS_TSV")
if [[ -z "$line" ]]; then
    echo "No model-seed row for task $TASK_ID" >&2
    exit 2
fi
IFS=$'\t' read -r index model_seed config search_space output_path <<< "$line"
if [[ "$index" != "$TASK_ID" ]]; then
    echo "Task index mismatch: expected $TASK_ID, got $index" >&2
    exit 2
fi

echo "model_seed=$model_seed split_seed=$SPLIT_SEED"
echo "output=$output_path"
"$PYTHON" speech_code/train_v1.py \
    --config="$config" \
    --search-space="$search_space" \
    --run-index 0 \
    --split-strategy per_sub \
    --fold 0 \
    --split-seed "$SPLIT_SEED" \
    --pretrain \
    --pretrain-subjects "$SUBJECTS"
