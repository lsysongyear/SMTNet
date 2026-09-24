#!/bin/bash
#SBATCH --job-name=spk_split_aggregate
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=00:20:00
#SBATCH --output=log/split_search/%x_%j.out
#SBATCH --error=log/split_search/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT must be the released VAD_SparKULee directory}
PYTHON=${PYTHON:-python3}
MANIFEST=${MANIFEST:?MANIFEST is required}

cd "$PROJECT_ROOT"
"$PYTHON" search_shared_story_splits.py aggregate --manifest "$MANIFEST"
