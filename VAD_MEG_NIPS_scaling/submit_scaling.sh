#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

SPLIT_SEEDS_CSV=42,43,44,45,46
SUBSET_SEED=1001
MODEL_SEED=42
DRY_RUN=false
RERUN_COMPLETED=false
SCALE_TAGS=(005 010 020 050 080 100)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split-seed)
            SPLIT_SEEDS_CSV="$2"
            shift 2
            ;;
        --split-seeds)
            SPLIT_SEEDS_CSV="$2"
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

IFS=',' read -r -a SPLIT_SEEDS <<< "$SPLIT_SEEDS_CSV"
mkdir -p log/scaling

echo "MEG NIPS scaling experiment"
echo "  scales:       5, 10, 20, 50, 80, 100 percent"
echo "  split seeds:  ${SPLIT_SEEDS[*]}"
echo "  subset seed:  $SUBSET_SEED (compatibility only; no sampling effect)"
echo "  model seed:   $MODEL_SEED"
echo "  scale unit:   chronological training recording duration"

submitted=0
skipped=0
for split_seed in "${SPLIT_SEEDS[@]}"; do
    if [[ ! "$split_seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid split seed: $split_seed" >&2
        exit 2
    fi

    missing_tasks=()
    for task_id in "${!SCALE_TAGS[@]}"; do
        scale_tag=${SCALE_TAGS[$task_id]}
        scale_dir="results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only/split_seed_${split_seed}/subset_seed_${SUBSET_SEED}/model_seed_${MODEL_SEED}/scale_${scale_tag}"
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

    if (( ${#missing_tasks[@]} == 0 )); then
        echo "  split_seed=$split_seed: all six paper scales already completed; skipped"
        continue
    fi

    array_spec=$(IFS=,; echo "${missing_tasks[*]}")
    exports="ALL,SPLIT_SEED=$split_seed,SUBSET_SEED=$SUBSET_SEED,MODEL_SEED=$MODEL_SEED"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  split_seed=$split_seed: DRY RUN sbatch --array=$array_spec --export=$exports job_scaling.slurm"
        continue
    fi

    job_id=$(sbatch \
        --parsable \
        --array="$array_spec" \
        --export="$exports" \
        --job-name="meg_dur_s${split_seed}" \
        job_scaling.slurm)
    echo "  split_seed=$split_seed: submitted array job $job_id (tasks $array_spec)"
    ((submitted += ${#missing_tasks[@]}))
done

echo "Tasks submitted: $submitted; completed tasks skipped: $skipped"
