#!/bin/bash
# ============================================================================
# Submit MEG_NIPS pretrain (multi-model sweep, 5 split_seeds each).
#
# Usage:
#   bash submit.sh                                    # all models in sweep
#   bash submit.sh --models brain_magic_speech_v7,awavenet
#   bash submit.sh --models ""                        # config.yaml only
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import yaml' || { echo "PYTHON must provide PyYAML" >&2; exit 1; }
test -f speech_code/train.py || { echo "Run from the Code_Release/VAD_MEG_NIPS source tree" >&2; exit 1; }
MODELS="all"
SLURM_SCRIPT="job1.slurm"
SWEEP_FILE="configs/speech/my_run/model_sweep.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODELS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve model list
if [ -z "$MODELS" ]; then
    MODEL_LIST=("")
elif [ "$MODELS" = "all" ]; then
    MODEL_LIST=($($PYTHON -c "
import yaml
with open('$SWEEP_FILE') as f:
    data = yaml.safe_load(f)
print(' '.join(data.keys()))
"))
else
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
fi

mkdir -p log

for model in "${MODEL_LIST[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SNAPSHOT_DIR="log/snapshot_${TIMESTAMP}${model:+_${model}}"
    mkdir -p "$SNAPSHOT_DIR"

    if [ -z "$model" ]; then
        cp configs/speech/my_run/config.yaml "$SNAPSHOT_DIR/config.yaml"
        cp configs/speech/my_run/search-space.yaml "$SNAPSHOT_DIR/search-space.yaml"
        JOB_MODEL=$(grep -E '^  [a-z_]+:' configs/speech/my_run/config.yaml | head -1 | sed 's/.*  //;s/:.*//')
        JOB_PREFIX="meg_pt"
    else
        $PYTHON -c "
import yaml
with open('configs/speech/my_run/config.yaml') as f:
    cfg = yaml.safe_load(f)
with open('$SWEEP_FILE') as f:
    sweep = yaml.safe_load(f)
if '$model' not in sweep:
    raise ValueError(f'Unknown model: $model')
cfg['model'] = {'$model': sweep['$model']}
cfg['general']['model_name'] = '$model'
with open('$SNAPSHOT_DIR/config.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

# Search-space with 5 split_seeds
ss = {}
for pk, pv in sweep['$model'].items():
    ss[f'(\"model\", \"$model\", \"{pk}\")'] = [pv]
ss['(\"general\", \"seed\")'] = [42]
ss['(\"general\", \"split_seed\")'] = [42, 43, 44, 45, 46]
ss['(\"loss\", \"config\", \"weight\")'] = [[1.0, 1.0]]
ss['(\"general\", \"loss_type\")'] = ['mse']
ss['(\"optimizer\", \"config\", \"lr\")'] = [0.001]
with open('$SNAPSHOT_DIR/search-space.yaml', 'w') as f:
    yaml.dump(ss, f, sort_keys=False)
"
        JOB_MODEL="$model"
        JOB_PREFIX="meg_pt_${model}"
    fi

    N_RUNS=$($PYTHON -c "
import yaml, itertools
with open('$SNAPSHOT_DIR/search-space.yaml') as f:
    ss = yaml.safe_load(f)
parsed = {eval(k): v for k,v in ss.items()}
print(len(list(itertools.product(*parsed.values()))))
")
    ARRAY_MAX=$((N_RUNS - 1))

    echo "=== Model: ${JOB_MODEL} === (HPO: $N_RUNS)"

    JOB_ID=$(sbatch \
        --export="ALL,N_RUNS=$N_RUNS,CONFIG_FILE=$SNAPSHOT_DIR/config.yaml,SEARCH_SPACE_FILE=$SNAPSHOT_DIR/search-space.yaml" \
        --parsable \
        --job-name="${JOB_PREFIX}" \
        --array="0-$ARRAY_MAX" \
        "$SLURM_SCRIPT")
    echo "  Submitted: $JOB_ID"
    echo ""
done

echo "All done. Logs: log/"
