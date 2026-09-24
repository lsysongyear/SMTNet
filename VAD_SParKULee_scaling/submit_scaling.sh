#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

SPLIT_SEED=84
SUBSET_SEED=1001
MODEL_SEED=1
SPLIT_PROTOCOL=shared_story
DRY_RUN=false
RERUN_COMPLETED=false
SCALE_TAGS=(005 010 020 050 080 100)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split-seed)
            SPLIT_SEED="$2"
            shift 2
            ;;
        --subset-seed)
            SUBSET_SEED="$2"
            shift 2
            ;;
        --model-seed)
            MODEL_SEED="$2"
            shift 2
            ;;
        --split-protocol)
            SPLIT_PROTOCOL="$2"
            shift 2
            ;;
        --rerun-completed)
            RERUN_COMPLETED=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p log/scaling
if [[ "$SPLIT_PROTOCOL" != "shared_story" ]]; then
    echo "split protocol must be shared_story" >&2
    exit 2
fi
EXPORTS="ALL,SPLIT_SEED=$SPLIT_SEED,SUBSET_SEED=$SUBSET_SEED,MODEL_SEED=$MODEL_SEED,SPLIT_PROTOCOL=$SPLIT_PROTOCOL"
missing_tasks=()
skipped=0
for task_id in "${!SCALE_TAGS[@]}"; do
    scale_tag=${SCALE_TAGS[$task_id]}
    scale_dir="results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only/protocol_${SPLIT_PROTOCOL}/split_seed_${SPLIT_SEED}/subset_seed_${SUBSET_SEED}/model_seed_${MODEL_SEED}/scale_${scale_tag}"
    completed=false
    if [[ "$RERUN_COMPLETED" == false ]]; then
        for status_file in "$scale_dir"/run_*/status.json; do
            if [[ -f "$status_file" ]] && grep -q '"status": "completed"' "$status_file"; then
                completed=true
                break
            fi
        done
    fi
    if [[ "$completed" == true ]]; then
        ((skipped += 1))
    else
        missing_tasks+=("$task_id")
    fi
done

echo "SParKULee scaling experiment"
echo "  scales:      5, 10, 20, 50, 80, 100 percent"
echo "  split seed:  $SPLIT_SEED"
echo "  subset seed: $SUBSET_SEED (compatibility only; no sampling effect)"
echo "  model seed:  $MODEL_SEED"
echo "  protocol:    $SPLIT_PROTOCOL"
echo "  scale unit:  per-subject chronological training recording duration"
echo "  completed tasks skipped: $skipped"

if (( ${#missing_tasks[@]} == 0 )); then
    echo "All six paper scales are already completed. Nothing to submit."
    exit 0
fi

array_spec=$(IFS=,; echo "${missing_tasks[*]}")

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: sbatch --array=$array_spec --export=$EXPORTS job_scaling.slurm"
    exit 0
fi

JOB_ID=$(sbatch \
    --parsable \
    --array="$array_spec" \
    --export="$EXPORTS" \
    job_scaling.slurm)
echo "Submitted array job: $JOB_ID (tasks $array_spec)"
