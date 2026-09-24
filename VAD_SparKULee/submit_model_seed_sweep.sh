#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-python3}
export PYTHON
BASE_SNAPSHOT=${BASE_SNAPSHOT:-}
SUBJECTS="1-3,5-16,18-24,26"
SEEDS="1-20"
SPLIT_SEED=84
MAX_CONCURRENT=4
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="candidate18_run_${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-snapshot) BASE_SNAPSHOT="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrent must be a positive integer" >&2
    exit 2
fi
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Invalid run name: $RUN_NAME" >&2
    exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 2
fi
if [[ -z "$BASE_SNAPSHOT" ]]; then
    echo "Provide --base-snapshot or BASE_SNAPSHOT with the candidate_18 directory produced by submit_shared_story_search.sh" >&2
    exit 2
fi
if [[ ! -f "$BASE_SNAPSHOT/config.yaml" || ! -f "$BASE_SNAPSHOT/search-space.yaml" ]]; then
    echo "Base snapshot must contain config.yaml and search-space.yaml: $BASE_SNAPSHOT" >&2
    exit 2
fi

mkdir -p log/model_seed_sweep results_model_seed_sweep
SNAPSHOT_DIR="$SCRIPT_DIR/log/model_seed_sweep/snapshot_${RUN_NAME}"
OUTPUT_ROOT="$SCRIPT_DIR/results_model_seed_sweep/$RUN_NAME"

"$PYTHON" model_seed_sweep.py prepare \
    --base-config "$BASE_SNAPSHOT/config.yaml" \
    --base-search-space "$BASE_SNAPSHOT/search-space.yaml" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --seeds "$SEEDS" \
    --subjects "$SUBJECTS" \
    --split-seed "$SPLIT_SEED" \
    --validation-story audiobook-5-1 \
    --test-story audiobook-5-3

TASKS_TSV="$SNAPSHOT_DIR/tasks.tsv"
MANIFEST="$SNAPSHOT_DIR/manifest.json"
SUBJECTS_FILE="$SNAPSHOT_DIR/subjects.txt"
RUNTIME_ROOT="$SNAPSHOT_DIR/runtime"
N_TASKS=$(wc -l < "$TASKS_TSV")
if [[ "$N_TASKS" -ne 20 ]]; then
    echo "Expected exactly 20 model seeds, found $N_TASKS" >&2
    exit 2
fi

ARRAY_JOB_ID=$(sbatch \
    --parsable \
    --array="0-$((N_TASKS - 1))%${MAX_CONCURRENT}" \
    --export="ALL,TASKS_TSV=$TASKS_TSV,SUBJECTS_FILE=$SUBJECTS_FILE,SPLIT_SEED=$SPLIT_SEED,MANIFEST=$MANIFEST,RUNTIME_ROOT=$RUNTIME_ROOT" \
    job_model_seed_sweep.slurm)
ARRAY_JOB_ID=${ARRAY_JOB_ID%%;*}

AGGREGATE_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --export="ALL,MANIFEST=$MANIFEST,RUNTIME_ROOT=$RUNTIME_ROOT" \
    job_model_seed_aggregate.slurm)
AGGREGATE_JOB_ID=${AGGREGATE_JOB_ID%%;*}

cat <<EOF
Submitted model-seed sweep: $ARRAY_JOB_ID
Submitted dependent aggregation: $AGGREGATE_JOB_ID
Seeds: $SEEDS
Fixed split: seed=$SPLIT_SEED, val=audiobook-5-1, test=audiobook-5-3
Manifest: $MANIFEST
Results: $OUTPUT_ROOT
EOF
