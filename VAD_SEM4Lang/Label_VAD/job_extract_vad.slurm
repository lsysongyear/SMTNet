#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUA800
#SBATCH -J sem_vad
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# Extract Silero VAD segments for SEM4Lang 60 audio stories
# ============================================================================

PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG_DIR="Label_VAD/log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/extract_vad_${TIMESTAMP}.log"

echo "============================================"
echo " SEM4Lang Silero VAD Extraction"
echo " 60 audio files"
echo " Log: $LOG_FILE"
echo "============================================"

$PYTHON Label_VAD/extract_vad_segments.py > "$LOG_FILE" 2>&1

echo "Done. Results: Label_VAD/output/segments_*.tsv"
