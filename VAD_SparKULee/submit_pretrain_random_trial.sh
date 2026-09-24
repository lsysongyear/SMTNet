#!/bin/bash
# ============================================================================
# Submit SparKULee pretrain + per-subject eval (multi-model sweep support).
#
# Usage:
#   bash submit_pretrain_random_trial.sh                           # all models in sweep
#   bash submit_pretrain_random_trial.sh --models brain_magic_speech_v7,awavenet
#   bash submit_pretrain_random_trial.sh --models ""                # config.yaml only
#   bash submit_pretrain_random_trial.sh --subjects 1-10 --seed 42
#   bash submit_pretrain_random_trial.sh --output-path results_candidate18_seed1
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import yaml' || { echo "PYTHON must provide PyYAML" >&2; exit 1; }
test -f speech_code/train_v1.py || { echo "Run from the Code_Release/VAD_SparKULee source tree" >&2; exit 1; }
SPLIT_SEED="84"
SUBJECTS="1-3,5-16,18-24,26"
MODELS="all"
OUTPUT_PATH=""
SLURM_SCRIPT="job_pretrain_random_trial.slurm"
SWEEP_FILE="configs/speech/my_run/model_sweep.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SPLIT_SEED="$2"; shift 2 ;;
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_PATH" ]; then
    if [ "$SPLIT_SEED" = "84" ]; then
        OUTPUT_PATH="results_candidate18_seed1"
    else
        OUTPUT_PATH="results_shared_story_seed${SPLIT_SEED}"
    fi
fi
case "$OUTPUT_PATH" in
    /*|..|../*|*/..|*/../*) echo "Output path must remain within this release project" >&2; exit 1 ;;
esac

# Verify the paper's common-story split against the mounted preprocessed files.
if [ "$SPLIT_SEED" = "84" ] && [ "$SUBJECTS" = "1-3,5-16,18-24,26" ]; then
    "$PYTHON" - <<'PY'
import random
from pathlib import Path
import yaml

with open("configs/speech/my_run/config.yaml") as handle:
    config = yaml.safe_load(handle)
dataset = config["data"]["datasets"]["train"][0]["sparkulee_vad"]
data_dir = Path(dataset["data_path"])
if not data_dir.is_dir():
    raise SystemExit(f"Missing SParKULee preprocessed data: {data_dir}")
expected_subjects = {f"{number:03d}" for number in (
    1, 2, 3, *range(5, 17), *range(18, 25), 26
)}
stories_by_subject = {subject: set() for subject in expected_subjects}
for file in data_dir.glob("*.npy"):
    parts = file.stem.split("_")
    if len(parts) < 4 or parts[-2] != dataset["preproc"] or parts[-1] != dataset["eeg_fs_tag"]:
        continue
    subject = parts[0].removeprefix("sub-")
    if subject not in stories_by_subject:
        continue
    audio_key = "_".join(parts[1:-2])
    if audio_key.startswith("audio-"):
        stories_by_subject[subject].add(audio_key.removeprefix("audio-"))
if any(len(stories) < 3 for stories in stories_by_subject.values()):
    raise SystemExit("A paper subject is missing at least three preprocessed stories")
common_stories = sorted(set.intersection(*stories_by_subject.values()))
random.Random(84).shuffle(common_stories)
actual = (common_stories[1], common_stories[0]) if len(common_stories) >= 2 else None
expected = ("audiobook-5-1", "audiobook-5-3")
if actual != expected:
    raise SystemExit(f"Candidate 18 split mismatch: expected val/test {expected}, got {actual}")
print(f"Verified candidate 18 shared-story split: val={actual[0]}, test={actual[1]}")
PY
fi

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

mkdir -p log/per_sub/pretrain

# Write subjects to file (avoids --export issues with commas)
SUBJECTS_FILE="log/per_sub/pretrain/subjects.txt"
mkdir -p "$(dirname "$SUBJECTS_FILE")"
echo "$SUBJECTS" > "$SUBJECTS_FILE"

for model in "${MODEL_LIST[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SNAPSHOT_DIR="log/per_sub/pretrain/snapshot_${TIMESTAMP}${model:+_${model}}"
    mkdir -p "$SNAPSHOT_DIR"

    if [ -z "$model" ]; then
        cp configs/speech/my_run/config.yaml "$SNAPSHOT_DIR/config.yaml"
        cp configs/speech/my_run/search-space.yaml "$SNAPSHOT_DIR/search-space.yaml"
        JOB_MODEL=$(grep -E '^  [a-z_]+:' configs/speech/my_run/config.yaml | head -1 | sed 's/.*  //;s/:.*//')
        JOB_PREFIX="spk_pt"
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

ss = {}
for pk, pv in sweep['$model'].items():
    ss[f'(\"model\", \"$model\", \"{pk}\")'] = [pv]
ss['(\"general\", \"seed\")'] = [1]
ss['(\"loss\", \"config\", \"weight\")'] = [[1.0, 1.0]]
ss['(\"general\", \"loss_type\")'] = ['mse']
ss['(\"optimizer\", \"config\", \"lr\")'] = [0.001]
with open('$SNAPSHOT_DIR/search-space.yaml', 'w') as f:
    yaml.dump(ss, f, sort_keys=False)
"
        JOB_MODEL="$model"
        JOB_PREFIX="spk_pt_${model}"
    fi

    $PYTHON -c "
import yaml
path = '$SNAPSHOT_DIR/config.yaml'
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg['general']['output_path'] = '$OUTPUT_PATH'
cfg['general']['split_protocol'] = 'global_shared_story'
with open(path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
"

    N_RUNS=$($PYTHON -c "
import yaml, itertools
with open('$SNAPSHOT_DIR/search-space.yaml') as f:
    ss = yaml.safe_load(f)
parsed = {eval(k): v for k,v in ss.items()}
print(len(list(itertools.product(*parsed.values()))))
")
    ARRAY_MAX=$((N_RUNS - 1))

    echo "=== Model: ${JOB_MODEL} === (HPO: $N_RUNS)"
    echo "  Snapshot: $SNAPSHOT_DIR"

    JOB_ID=$(sbatch \
        --export="ALL,SPLIT_SEED=$SPLIT_SEED,N_RUNS=$N_RUNS,MODEL_TAG=$JOB_MODEL,SUBJECTS_FILE=$SUBJECTS_FILE,CONFIG_FILE=$SNAPSHOT_DIR/config.yaml,SEARCH_SPACE_FILE=$SNAPSHOT_DIR/search-space.yaml" \
        --parsable \
        --job-name="${JOB_PREFIX}" \
        --array="0-$ARRAY_MAX" \
        "$SLURM_SCRIPT")
    echo "  Submitted: $JOB_ID"
    echo ""
done

echo "All done. Logs: log/per_sub/pretrain/"
echo "Results: $OUTPUT_PATH"
