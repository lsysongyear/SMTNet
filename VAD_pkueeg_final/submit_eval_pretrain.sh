#!/bin/bash
# ============================================================================
# submit_eval_pretrain.sh - 提交预训练模型逐被试评估作业
# ============================================================================
# 使用与预训练相同的 per_sub_5t5v 划分，通过 --seed 控制随机划分。
# 每名被试: 随机 5 test + 5 val, 与预训练数据划分一致。
#
# 用法:
#   bash submit_eval_pretrain.sh                           # 默认 seed=545, 全部 25 人
#   bash submit_eval_pretrain.sh --seed 102                 # 其他 seed
#   bash submit_eval_pretrain.sh --subjects 1-10            # 部分被试
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEED=545
SUBJECTS="1-25"
CKPT_DIR=""  # auto-detect from results/ if empty (avoids --export path-escaping issues)

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " Pretrain Model Per-Subject Evaluation"
echo " Seed:      $SEED"
echo " Subjects:  $SUBJECTS"
echo " Ckpt Dir:  ${CKPT_DIR:-auto}"
echo " Split:     per_sub_5t5v (5 test + 5 val random)"
echo "============================================"
echo ""
echo "NOTE: seed must match the pretrain split seed"
echo "      for consistent test/val stories."
echo ""

EXPORT="ALL,SUBJECTS=$SUBJECTS,SEED=$SEED"
[ -n "$CKPT_DIR" ] && EXPORT="$EXPORT,CKPT_DIR=$CKPT_DIR"

sbatch --export="$EXPORT" job_eval_pretrain.slurm

echo "Submitted. Results: sub-XX/pretrain_eval/"
