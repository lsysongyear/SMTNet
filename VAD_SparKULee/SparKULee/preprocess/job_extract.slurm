#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J spk_eeg_prep
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================

PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo " SparKULee EEG Preprocessing"
echo " 512Hz → 250Hz, 1-80Hz bandpass"
echo " Log: $LOG_FILE"
echo "============================================"

$PYTHON "$SCRIPT_DIR/extract_eeg_250hz_bp.py" > "$LOG_FILE" 2>&1

echo "Done. Log: $LOG_FILE"
