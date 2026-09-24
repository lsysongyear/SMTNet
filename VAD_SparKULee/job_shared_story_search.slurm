#!/bin/bash
#SBATCH --job-name=spk_split_search
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --exclude=gpua800n24
#SBATCH --output=log/split_search/%x_%A_%a.out
#SBATCH --error=log/split_search/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT must be the released VAD_SparKULee directory}
PYTHON=${PYTHON:-python3}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
CANDIDATES_TSV=${CANDIDATES_TSV:?CANDIDATES_TSV is required}
SUBJECTS_FILE=${SUBJECTS_FILE:?SUBJECTS_FILE is required}
SUBJECTS=$(<"$SUBJECTS_FILE")

cd "$PROJECT_ROOT"
line=$(sed -n "$((TASK_ID + 1))p" "$CANDIDATES_TSV")
if [[ -z "$line" ]]; then
    echo "No candidate row for task $TASK_ID" >&2
    exit 2
fi
IFS=$'\t' read -r candidate split_seed val_story test_story config search_space output_path <<< "$line"
if [[ "$candidate" != "$TASK_ID" ]]; then
    echo "Candidate index mismatch: expected $TASK_ID, got $candidate" >&2
    exit 2
fi

echo "candidate=$candidate split_seed=$split_seed val=$val_story test=$test_story"
echo "output=$output_path"

"$PYTHON" speech_code/train_v1.py \
    --config="$config" \
    --search-space="$search_space" \
    --run-index 0 \
    --split-strategy per_sub \
    --fold 0 \
    --split-seed "$split_seed" \
    --pretrain \
    --pretrain-subjects "$SUBJECTS"
