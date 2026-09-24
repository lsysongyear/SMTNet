#!/bin/bash
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 48:00:00
#SBATCH --partition=GPUA800
#SBATCH -J meg_nips_vad
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpua800n22,gpua800n01,gpua800n25,gpua800n13

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON="${PYTHON:-python3}"
test -f speech_code/train.py || { echo "Missing release training source" >&2; exit 1; }
CONFIG="${CONFIG_FILE:?CONFIG_FILE is required}"
SEARCH_SPACE="${SEARCH_SPACE_FILE:?SEARCH_SPACE_FILE is required}"

$PYTHON speech_code/train.py --config="$CONFIG" --search-space="$SEARCH_SPACE" --run-index=$RUN_INDEX
