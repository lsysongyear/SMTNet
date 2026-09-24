#!/bin/bash
# ============================================================================
#SBATCH -o ./log/preprocess_meg_%j.out
#SBATCH -e ./log/preprocess_meg_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH --partition=GPUA800
#SBATCH --gres=gpu:1
#SBATCH -J sem4lang_prep
#SBATCH --mem=32G
#SBATCH --nodes=1
# ============================================================================
# Preprocess SEM4Lang MEG: 1000Hz → 100Hz downsampling, story segmentation
# ============================================================================

PYTHON="${PYTHON:-python3}"

cd "$(dirname "$0")/.."
mkdir -p log SEM4Lang/meg

echo "Starting MEG preprocessing (NF, 100Hz)..."
$PYTHON SEM4Lang/preprocess_meg.py
echo "Done."
