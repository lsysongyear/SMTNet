#!/bin/bash
#SBATCH --job-name=spk_seed_aggregate
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=00:20:00
#SBATCH --output=log/model_seed_sweep/%x_%j.out
#SBATCH --error=log/model_seed_sweep/%x_%j.err

set -euo pipefail

PYTHON=${PYTHON:-python3}
MANIFEST=${MANIFEST:?MANIFEST is required}
RUNTIME_ROOT=${RUNTIME_ROOT:?RUNTIME_ROOT is required}

export PYTHONDONTWRITEBYTECODE=1
cd "$RUNTIME_ROOT"
"$PYTHON" model_seed_sweep.py verify-runtime --manifest "$MANIFEST"
"$PYTHON" model_seed_sweep.py aggregate --manifest "$MANIFEST"
