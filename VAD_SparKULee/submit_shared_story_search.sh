#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-python3}
export PYTHON
SUBJECTS="1-3,5-16,18-24,26"
MAX_CONCURRENT=4
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="run_${TIMESTAMP}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subjects) SUBJECTS="$2"; shift 2 ;;
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
    echo "--run-name may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 2
fi

mkdir -p log/split_search results_shared_story_search
SNAPSHOT_DIR="$SCRIPT_DIR/log/split_search/snapshot_${RUN_NAME}"
OUTPUT_ROOT="$SCRIPT_DIR/results_shared_story_search/$RUN_NAME"

"$PYTHON" search_shared_story_splits.py prepare \
    --base-config "$SCRIPT_DIR/configs/speech/my_run/config.yaml" \
    --model-sweep "$SCRIPT_DIR/configs/speech/my_run/model_sweep.yaml" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --subjects "$SUBJECTS"

CANDIDATES_TSV="$SNAPSHOT_DIR/candidates.tsv"
MANIFEST="$SNAPSHOT_DIR/manifest.json"
SUBJECTS_FILE="$SNAPSHOT_DIR/subjects.txt"
printf '%s\n' "$SUBJECTS" > "$SUBJECTS_FILE"
N_CANDIDATES=$(wc -l < "$CANDIDATES_TSV")
if [[ "$N_CANDIDATES" -ne 30 ]]; then
    echo "Expected 30 ordered pairs from six common stories, found $N_CANDIDATES" >&2
    exit 2
fi

JOB_ID=$(sbatch \
    --parsable \
    --array="0-$((N_CANDIDATES - 1))%${MAX_CONCURRENT}" \
    --export="ALL,PROJECT_ROOT=$SCRIPT_DIR,CANDIDATES_TSV=$CANDIDATES_TSV,SUBJECTS_FILE=$SUBJECTS_FILE" \
    job_shared_story_search.slurm)

AGGREGATE_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:${JOB_ID}" \
    --export="ALL,PROJECT_ROOT=$SCRIPT_DIR,MANIFEST=$MANIFEST" \
    job_shared_story_aggregate.slurm)

cat <<EOF
Submitted shared-story search: $JOB_ID
Submitted dependent aggregation: $AGGREGATE_JOB_ID
Candidates: $N_CANDIDATES (ordered validation/test pairs)
Concurrency: $MAX_CONCURRENT
Manifest: $MANIFEST
Results: $OUTPUT_ROOT
EOF
