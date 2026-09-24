#!/bin/bash
# Submit paper models with the per_sub_5t5v split.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import yaml' || { echo "PYTHON must provide PyYAML" >&2; exit 1; }
test -f speech_code/train_v1.py || { echo "Run from the Code_Release/VAD_SEM4Lang source tree" >&2; exit 1; }
DEFAULT_SEEDS="5"
MODELS="all"
OUTPUT_PATH=""
SLURM_SCRIPT="job_train.slurm"
SWEEP_FILE="configs/speech/my_run/model_sweep.yaml"

SEEDS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            if [[ "$2" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                for ((s=${BASH_REMATCH[1]}; s<=${BASH_REMATCH[2]}; s++)); do SEEDS+=("$s"); done
            else
                for s in $2; do SEEDS+=("$s"); done
            fi
            shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ ${#SEEDS[@]} -eq 0 ]; then
    if [[ "$DEFAULT_SEEDS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        for ((s=${BASH_REMATCH[1]}; s<=${BASH_REMATCH[2]}; s++)); do SEEDS+=("$s"); done
    else
        for s in $DEFAULT_SEEDS; do SEEDS+=("$s"); done
    fi
fi

if [ -n "$OUTPUT_PATH" ]; then
    case "$OUTPUT_PATH" in
        /*|..|../*|*/..|*/../*) echo "Output path must remain within this release project" >&2; exit 1 ;;
    esac
fi

if [ "$MODELS" = "all" ]; then
    MODEL_LIST=($($PYTHON -c "
import yaml
with open('$SWEEP_FILE') as f:
    data = yaml.safe_load(f)
print(' '.join(data.keys()))
"))
else
    [ -n "$MODELS" ] || { echo "--models must name a model or use all" >&2; exit 2; }
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
fi

mkdir -p log/train

for model in "${MODEL_LIST[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SNAPSHOT_DIR="log/train/snapshot_${TIMESTAMP}_${model}"
    mkdir -p "$SNAPSHOT_DIR"

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

ss = {}
for pk, pv in sweep['$model'].items():
    ss[f'(\"model\", \"$model\", \"{pk}\")'] = [pv]
ss['(\"general\", \"seed\")'] = [1]
ss['(\"general\", \"split_seed\")'] = [0]
ss['(\"loss\", \"config\", \"weight\")'] = [[1.0, 1.0]]
ss['(\"general\", \"loss_type\")'] = ['mse']
ss['(\"optimizer\", \"config\", \"lr\")'] = [0.001]
with open('$SNAPSHOT_DIR/search-space.yaml', 'w') as f:
    yaml.dump(ss, f, sort_keys=False)
"
    JOB_MODEL="$model"
    JOB_PREFIX="s4l_pt_${model}"

    if [ -n "$OUTPUT_PATH" ]; then
        $PYTHON -c "
import yaml
path = '$SNAPSHOT_DIR/config.yaml'
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg['general']['output_path'] = '$OUTPUT_PATH'
with open(path, 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)
"
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

    echo "=== Model: ${JOB_MODEL} === (per_sub_5t5v, HPO runs: $N_RUNS)"
    for s in "${SEEDS[@]}"; do
        echo "  [split_seed=$s]"
        JOB_ID=$(sbatch \
            --export="ALL,SPLIT_SEED=$s,CONFIG_FILE=$SNAPSHOT_DIR/config.yaml,SEARCH_SPACE_FILE=$SNAPSHOT_DIR/search-space.yaml" \
            --parsable \
            --job-name="${JOB_PREFIX}_s${s}" \
            --array="0-$ARRAY_MAX" \
            "$SLURM_SCRIPT")
        echo "    Submitted: $JOB_ID"
    done
    echo ""
done

echo "All done. Logs: log/train/"
