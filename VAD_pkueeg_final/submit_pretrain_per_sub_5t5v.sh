#!/bin/bash
# ============================================================================
# Submit pretrain with per_sub_5t5v split.
#
# Usage:
#   bash submit_pretrain_per_sub_5t5v.sh                                  # current config
#   bash submit_pretrain_per_sub_5t5v.sh --models all                     # all models in model_sweep.yaml
#   bash submit_pretrain_per_sub_5t5v.sh --models brain_magic_speech_v7,awavenet,cnn_lstm
#   bash submit_pretrain_per_sub_5t5v.sh --seeds 102-106 --subjects 1-15
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import yaml' || { echo "PYTHON must provide PyYAML" >&2; exit 1; }
test -f speech_code/train_v1.py || { echo "Run from the Code_Release/VAD_pkueeg_final source tree" >&2; exit 1; }
SEED=545
SUBJECTS="1-25"
MODELS="all"       # default: all models in model_sweep.yaml; set "" for config.yaml only
SLURM_SCRIPT="job_pretrain_per_sub_5t5v.slurm"
SWEEP_FILE="configs/speech/my_run/model_sweep.yaml"

SEEDS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; SEEDS+=("$2"); shift 2 ;;
        --seeds)
            if [[ "$2" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                for ((s=${BASH_REMATCH[1]}; s<=${BASH_REMATCH[2]}; s++)); do SEEDS+=("$s"); done
            else
                for s in $2; do SEEDS+=("$s"); done
            fi
            shift 2 ;;
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[ ${#SEEDS[@]} -eq 0 ] && SEEDS=("$SEED")

# Resolve model list
if [ -z "$MODELS" ]; then
    MODEL_LIST=("")
elif [ "$MODELS" = "all" ]; then
    # Read all model names from sweep file
    MODEL_LIST=($($PYTHON -c "
import yaml
with open('$SWEEP_FILE') as f:
    data = yaml.safe_load(f)
print(' '.join(data.keys()))
"))
else
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
fi

mkdir -p log/per_sub_5t5v/pretrain

for model in "${MODEL_LIST[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SNAPSHOT_DIR="log/per_sub_5t5v/pretrain/snapshot_${TIMESTAMP}${model:+_${model}}"
    mkdir -p "$SNAPSHOT_DIR"

    if [ -z "$model" ]; then
        cp configs/speech/my_run/config.yaml "$SNAPSHOT_DIR/config.yaml"
        cp configs/speech/my_run/search-space.yaml "$SNAPSHOT_DIR/search-space.yaml"
        JOB_MODEL=$(grep -E '^  [a-z_]+:' configs/speech/my_run/config.yaml | head -1 | sed 's/.*  //;s/:.*//')
        JOB_PREFIX="vad_pt"
    else
        # Generate config + search-space from sweep file
        $PYTHON -c "
import yaml
with open('configs/speech/my_run/config.yaml') as f:
    cfg = yaml.safe_load(f)
with open('$SWEEP_FILE') as f:
    sweep = yaml.safe_load(f)
if '$model' not in sweep:
    raise ValueError(f'Unknown model: $model. Available: {sorted(sweep.keys())}')

cfg['model'] = {'$model': sweep['$model']}
cfg['general']['model_name'] = '$model'
with open('$SNAPSHOT_DIR/config.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

ss = {}
for pk, pv in sweep['$model'].items():
    ss[f'(\"model\", \"$model\", \"{pk}\")'] = [pv]
ss['(\"general\", \"seed\")'] = [1]
ss['(\"general\", \"split_seed\")'] = [545]
ss['(\"loss\", \"config\", \"weight\")'] = [[1.0, 1.0]]
ss['(\"general\", \"loss_type\")'] = ['mse']
ss['(\"optimizer\", \"config\", \"lr\")'] = [0.001]
ss['(\"data\", \"datasets\", \"train\", 0, \"eeg_speech_v1\", \"tmax\")'] = [12.0]
with open('$SNAPSHOT_DIR/search-space.yaml', 'w') as f:
    yaml.dump(ss, f, sort_keys=False)
"
        JOB_MODEL="$model"
        JOB_PREFIX="vad_pt_${model}"
    fi

    # Count HPO runs
    N_RUNS=$($PYTHON -c "
import yaml, itertools
with open('$SNAPSHOT_DIR/search-space.yaml') as f:
    ss = yaml.safe_load(f)
parsed = {}
for k,v in ss.items(): parsed[eval(k)] = v
print(len(list(itertools.product(*parsed.values()))))
")
    ARRAY_MAX=$((N_RUNS - 1))

    echo "=== Model: ${JOB_MODEL} === (HPO: $N_RUNS)"

    for s in "${SEEDS[@]}"; do
        EXPORT="ALL,SPLIT_SEED=$s,N_RUNS=$N_RUNS,CONFIG_FILE=$SNAPSHOT_DIR/config.yaml,SEARCH_SPACE_FILE=$SNAPSHOT_DIR/search-space.yaml"
        [ -n "$SUBJECTS" ] && EXPORT="$EXPORT,PRETRAIN_SUBJECTS=$SUBJECTS"

        JOB_ID=$(sbatch \
            --export="$EXPORT" \
            --parsable \
            --job-name="${JOB_PREFIX}_s${s}" \
            --array="0-$ARRAY_MAX" \
            "$SLURM_SCRIPT")
        echo "  [seed=$s] $JOB_ID"
    done
    echo ""
done

echo "All done. Logs: log/per_sub_5t5v/pretrain/"
