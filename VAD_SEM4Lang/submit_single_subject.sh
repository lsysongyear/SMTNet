#!/bin/bash
# ============================================================================
# submit_single_subject.sh - 提交 SEM4Lang 逐被试训练
# ============================================================================
# single_subject split: per-subject cross_trial, 10 test + 5 val + 45 train.
# 12 名被试各自独立训练。
#
# 用法:
#   bash submit_single_subject.sh                         # 全部 12 人, fold=0, seed=42
#   bash submit_single_subject.sh --seed 102              # custom seed
#   bash submit_single_subject.sh --subjects 1-6          # 部分被试
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SPLIT_SEED="42"
SUBJECT_RANGE="1-12"
FOLD="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SPLIT_SEED="$2"; shift 2 ;;
        --subjects) SUBJECT_RANGE="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

START=$(echo "$SUBJECT_RANGE" | cut -d'-' -f1)
END=$(echo "$SUBJECT_RANGE" | cut -d'-' -f2)
ARRAY_RANGE="$((START-1))-$((END-1))"
N=$((END - START + 1))

echo "============================================"
echo " SEM4Lang Per-Subject Training"
echo " Subjects:  $SUBJECT_RANGE ($N subjects)"
echo " Fold:      $FOLD"
echo " Seed:      $SPLIT_SEED"
echo " Split:     10 test + 5 val + 45 train"
echo "============================================"

mkdir -p log/single_subject

sbatch \
    --export="ALL,SPLIT_SEED=$SPLIT_SEED,FOLD=$FOLD" \
    --array="$ARRAY_RANGE" \
    job_single_subject.slurm

echo "Submitted. Logs: log/single_subject/"
