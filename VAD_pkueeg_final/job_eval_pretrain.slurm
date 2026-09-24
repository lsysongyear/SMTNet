#!/bin/bash
# ============================================================================
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J vad_eval_pt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n24
# ============================================================================
# 使用预训练模型逐被试评估（不微调，直接验证集阈值搜索 + 测试）
# ============================================================================

PYTHON=/gpfs/share/home/2201112028/miniconda3/envs/libribrain2/bin/python3

SUBJECTS="${SUBJECTS:-1-25}"
SEED="${SEED:-545}"
LOG_DIR="log/eval_pretrain"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

echo "============================================"
echo " Pretrain Model Per-Subject Evaluation"
echo " Subjects: $SUBJECTS"
echo " Seed:     $SEED"
echo " Log:      $LOG_FILE"
echo "============================================"

ARGS=(--subjects "$SUBJECTS" --seed "$SEED")
if [ -n "${CKPT_DIR:-}" ]; then
    ARGS+=(--ckpt-dir "$CKPT_DIR")
fi

"$PYTHON" eval_pretrain_per_subject.py "${ARGS[@]}" > "$LOG_FILE" 2>&1

echo "Done. Results in: results/.../split:per_sub_5t5v/.../sub-XX/pretrain_eval/"
