#!/usr/bin/env bash
# Submit the exact model families and seeds reported in the paper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATASET="all"
EXPERIMENT="all"
SUBMIT=false

usage() {
    echo "Usage: bash reproduce_paper.sh --dataset pku|spar|sem|libri|all --experiment tables|scaling|all [--submit]"
    echo "Without --submit, only prints the Slurm submission commands."
}

while (($#)); do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --submit) SUBMIT=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$DATASET" in pku|spar|sem|libri|all) ;; *) usage >&2; exit 2 ;; esac
case "$EXPERIMENT" in tables|scaling|all) ;; *) usage >&2; exit 2 ;; esac

table_models="brain_magic_speech_v7,cnn_lstm,dilated_conv,eeg_conformer,awavenet,pnpl_cnn_tcn,brain_magic_no_subject_attn,brain_magic_no_short_conv,brain_magic_no_feature_encoder"
libri_models="brain_magic_speech_v7,cnn_lstm,dilated_conv,eeg_conformer,awavenet,pnpl_cnn_tcn,brain_magic_no_short_conv,brain_magic_no_feature_encoder"

run() {
    local project="$1"
    shift
    printf 'cd %q &&' "$ROOT/$project"
    printf ' %q' "$@"
    printf '\n'
    if [[ "$SUBMIT" == true ]]; then
        (cd "$ROOT/$project" && "$@")
    fi
}

if [[ "$DATASET" == pku || "$DATASET" == all ]]; then
    if [[ "$EXPERIMENT" == tables || "$EXPERIMENT" == all ]]; then
        run VAD_pkueeg_final bash submit_pretrain_per_sub_5t5v.sh --models "$table_models" --seed 545
    fi
    if [[ "$EXPERIMENT" == scaling || "$EXPERIMENT" == all ]]; then
        run VAD_pkueeg_scaling bash submit_scaling.sh --split-seed 545 --model-seed 1
    fi
fi
if [[ "$DATASET" == spar || "$DATASET" == all ]]; then
    if [[ "$EXPERIMENT" == tables || "$EXPERIMENT" == all ]]; then
        run VAD_SparKULee bash submit_pretrain_random_trial.sh --models "$table_models" --seed 84 --output-path results_paper_candidate18
    fi
    if [[ "$EXPERIMENT" == scaling || "$EXPERIMENT" == all ]]; then
        run VAD_SParKULee_scaling bash submit_scaling.sh --split-seed 84 --model-seed 1
    fi
fi
if [[ "$DATASET" == sem || "$DATASET" == all ]]; then
    if [[ "$EXPERIMENT" == tables || "$EXPERIMENT" == all ]]; then
        run VAD_SEM4Lang bash submit_train.sh --models "$table_models" --seeds 5 --output-path results_paper_split5
    fi
    if [[ "$EXPERIMENT" == scaling || "$EXPERIMENT" == all ]]; then
        run VAD_SEM4Lang_scaling bash submit_scaling.sh --split-seed 5 --model-seed 1
    fi
fi
if [[ "$DATASET" == libri || "$DATASET" == all ]]; then
    if [[ "$EXPERIMENT" == tables || "$EXPERIMENT" == all ]]; then
        run VAD_MEG_NIPS bash submit.sh --models "$libri_models"
    fi
    if [[ "$EXPERIMENT" == scaling || "$EXPERIMENT" == all ]]; then
        run VAD_MEG_NIPS_scaling bash submit_scaling.sh --split-seeds 42,43,44,45,46 --model-seed 42
    fi
fi
