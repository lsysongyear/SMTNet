#!/bin/bash
# ============================================================================
# Submit per-subject VAD training jobs (no pretrain, train from scratch).
#
# Usage:
#   bash submit_per_subject.sh                        # all 85 subjects
#   bash submit_per_subject.sh --subjects 1-10        # first 10 subjects
#   bash submit_per_subject.sh --subjects 1-26        # AB-only subjects
#   bash submit_per_subject.sh --seed 1               # custom seed
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SPLIT_SEED="42"
SUBJECT_RANGE="1-85"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SPLIT_SEED="$2"
            shift 2
            ;;
        --subjects)
            SUBJECT_RANGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash submit_per_subject.sh [--subjects 1-85] [--seed 42]"
            exit 1
            ;;
    esac
done

START=$(echo "$SUBJECT_RANGE" | cut -d'-' -f1)
END=$(echo "$SUBJECT_RANGE" | cut -d'-' -f2)
ARRAY_START=$((START - 1))
ARRAY_END=$((END - 1))
ARRAY_RANGE="${ARRAY_START}-${ARRAY_END}"
N_SUBJECTS=$((ARRAY_END - ARRAY_START + 1))

echo "============================================"
echo " SparKULee VAD - Per-Subject Training"
echo " Subjects:     $SUBJECT_RANGE ($N_SUBJECTS subjects)"
echo " Array range:  $ARRAY_RANGE"
echo " Split seed:   $SPLIT_SEED"
echo "============================================"

sbatch \
    --export=ALL,SPLIT_SEED="$SPLIT_SEED" \
    --array="$ARRAY_RANGE" \
    job_per_subject.slurm

echo "Submitted. Logs: log/per_sub/per_subject/"
echo ""
echo "After all jobs complete, aggregate results:"
echo "  python compute_per_subject_means.py"
