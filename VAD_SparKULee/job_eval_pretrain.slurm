#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPUA800
#SBATCH -J vad_eval_pt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# 预训练模型逐被试评估（不微调，验证集阈值搜索 + 测试）
# ============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"

# 从被试列表文件中读取当前 task 对应的被试 ID
if [ -n "${SUBJ_LIST_FILE:-}" ] && [ -f "$SUBJ_LIST_FILE" ]; then
    SUBJ_LIST=($(cat "$SUBJ_LIST_FILE"))
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    SUBJ=$(printf "%03d" ${SUBJ_LIST[$TASK_ID]})
fi
SUBJ="${SUBJ:-001}"
SEED="${SEED:-42}"
LOG_DIR="log/per_sub/eval_pretrain"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sub-${SUBJ}_${TIMESTAMP}.log"

ARGS=(--subj "$SUBJ" --seed "$SEED")
[ -n "${CKPT:-}" ] && ARGS+=(--ckpt "$CKPT")

echo "============================================"
echo " Pretrain Model Evaluation"
echo " Subject: $SUBJ  |  Seed: $SEED"
echo " Log:     $LOG_FILE"
echo "============================================"

"$PYTHON" eval_pretrain_per_subject.py "${ARGS[@]}" > "$LOG_FILE" 2>&1

echo "Done."
