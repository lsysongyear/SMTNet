#!/bin/bash
# ============================================================================
# Submit SparKULee 5-fold cross-subject VAD training.
#
# Usage:
#   bash submit_cross_subject_5fold.sh
#   SPLIT_SEED=1 bash submit_cross_subject_5fold.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SPLIT_SEED="${SPLIT_SEED:-42}"

echo "============================================"
echo " SparKULee VAD - 5-fold cross-subject"
echo " Subjects:   85"
echo " Test/fold:  17 subjects"
echo " Array:      0-4"
echo " Split seed: $SPLIT_SEED"
echo "============================================"

sbatch --export=ALL,SPLIT_SEED="$SPLIT_SEED" --array=0-4 job_cross_subject_5fold.slurm

echo "Submitted. Logs: log/cross_subject/"
