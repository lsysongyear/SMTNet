#!/bin/bash
# ============================================================================
# submit_eval_pretrain.sh - 提交预训练模型逐被试评估
# ============================================================================
# 用法:
#   bash submit_eval_pretrain.sh --subj 001                    # 单名被试
#   bash submit_eval_pretrain.sh --subj 001-003                # 连续被试（array）
#   bash submit_eval_pretrain.sh --subjects 1-3,5-16,18-24,26   # 指定范围（array）
#   bash submit_eval_pretrain.sh --subj 001 --seed 102         # 指定 seed
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SUBJ="001"
SUBJECTS="1-3,5-16,18-24,26"
PYTHON="${PYTHON:-python3}"
SEED=42
CKPT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --subj) SUBJ="$2"; SUBJECTS=""; shift 2 ;;
        --subjects) SUBJECTS="$2"; SUBJ=""; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "Pass an existing checkpoint with --ckpt for this historical evaluation script" >&2
    exit 1
fi

mkdir -p log/per_sub/eval_pretrain

if [ -n "$SUBJECTS" ]; then
    # --- 多被试 array 模式 ---
    # 将被试 spec 转为有序 ID 列表写入文件
    SUBJ_LIST=$($PYTHON -c "
spec = '$SUBJECTS'
ids = set()
for part in spec.split(','):
    part = part.strip()
    if '-' in part:
        lo, hi = part.split('-', 1)
        for i in range(int(lo), int(hi)+1): ids.add(i)
    else: ids.add(int(part))
ids = sorted(ids)
print(' '.join(str(i) for i in ids))
")
    SUBJ_ARRAY=($SUBJ_LIST)
    N=${#SUBJ_ARRAY[@]}

    SUBJ_LIST_FILE="log/per_sub/eval_pretrain/subj_list.txt"
    mkdir -p "$(dirname "$SUBJ_LIST_FILE")"
    echo "$SUBJ_LIST" > "$SUBJ_LIST_FILE"

    echo "Submitting array: $N subjects"

    EXPORT="ALL,SEED=$SEED,SUBJ_LIST_FILE=$SUBJ_LIST_FILE"
    [ -n "$CKPT" ] && EXPORT="$EXPORT,CKPT=$CKPT"

    sbatch --export="$EXPORT" --array="0-$((N-1))" job_eval_pretrain.slurm
else
    # --- 单被试模式 ---
    echo "Submitting single subject: sub-$SUBJ"

    EXPORT="ALL,SUBJ=$SUBJ,SEED=$SEED"
    [ -n "$CKPT" ] && EXPORT="$EXPORT,CKPT=$CKPT"

    sbatch --export="$EXPORT" job_eval_pretrain.slurm
fi

echo "Submitted. Logs: log/per_sub/eval_pretrain/"
