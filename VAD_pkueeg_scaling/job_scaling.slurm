#!/bin/bash
#SBATCH --job-name=vad_scaling
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=gpua800n24
#SBATCH --output=log/scaling/%x_%A_%a.out
#SBATCH --error=log/scaling/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the release project directory}"
PYTHON="${PYTHON:-python3}"
SCALES=(0.05 0.10 0.20 0.50 0.80 1.00)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( TASK_ID < 0 || TASK_ID >= ${#SCALES[@]} )); then
    echo "Invalid array task index: $TASK_ID" >&2
    exit 2
fi

SCALE=${SCALES[$TASK_ID]}
SPLIT_SEED=${SPLIT_SEED:-545}
SUBSET_SEED=${SUBSET_SEED:-1001}
MODEL_SEED=${MODEL_SEED:-1}

cd "$PROJECT_ROOT"
[[ -f train_scaling.py ]] || { echo "train_scaling.py not found in $PROJECT_ROOT" >&2; exit 2; }
echo "scale=$SCALE split_seed=$SPLIT_SEED subset_seed=$SUBSET_SEED model_seed=$MODEL_SEED"

"$PYTHON" train_scaling.py \
    --config configs/scaling/config.yaml \
    --scale "$SCALE" \
    --split-seed "$SPLIT_SEED" \
    --subset-seed "$SUBSET_SEED" \
    --model-seed "$MODEL_SEED"
