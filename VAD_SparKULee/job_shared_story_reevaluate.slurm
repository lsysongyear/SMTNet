#!/bin/bash
#SBATCH --job-name=spk_split_reeval
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --exclude=gpua800n24
#SBATCH --output=log/split_search/%x_%A_%a.out
#SBATCH --error=log/split_search/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT must be the released VAD_SparKULee directory}
PYTHON=${PYTHON:-python3}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MANIFEST=${MANIFEST:?MANIFEST is required}

cd "$PROJECT_ROOT"
"$PYTHON" reevaluate_shared_story_candidate.py \
    --manifest "$MANIFEST" \
    --candidate-index "$TASK_ID"
