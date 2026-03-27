#!/usr/bin/env bash
# Run a single evaluation inside Docker
#
# Usage:
#   bash docker/eval_single.sh <task> <model> <data_dir> [output_dir] [extra_args...]
#
# Examples:
#   # TSE evaluation
#   bash docker/eval_single.sh tse orange_pi /path/to/data
#   bash docker/eval_single.sh tse orange_pi_5out /path/to/data ./results
#
#   # SED evaluation (single condition)
#   bash docker/eval_single.sh sed finetuned_ast /path/to/data ./results \
#       --num_fg_min 1 --num_fg_max 1 --num_bg_min 1 --num_bg_max 1
set -euo pipefail

TASK="${1:?Usage: bash docker/eval_single.sh <tse|sed> <model> <data_dir> [output_dir] [extra_args...]}"
MODEL="${2:?Specify model name}"
DATA_DIR="${3:?Specify data directory}"
OUTPUT_DIR="${4:-./eval_results/${TASK}/${MODEL}}"
shift 4 2>/dev/null || shift 3
EXTRA_ARGS="$*"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

IMAGE="fine-grained-soundscape"

if [ "$TASK" = "tse" ]; then
    docker run --rm --gpus all \
        --shm-size=2g \
        -v "${DATA_DIR}:/data:ro" \
        -v "${OUTPUT_DIR}:/output" \
        -w /app \
        "${IMAGE}" python -m src.tse.eval \
        --pretrained ooshyun/fine_grained_soundscape_control \
        --model "${MODEL}" \
        --data_dir /data \
        --output_dir /output \
        ${EXTRA_ARGS}
elif [ "$TASK" = "sed" ]; then
    docker run --rm --gpus all \
        --shm-size=2g \
        -v "${DATA_DIR}:/data:ro" \
        -v "${OUTPUT_DIR}:/output" \
        -w /app \
        "${IMAGE}" python -m src.sed.eval \
        --pretrained ooshyun/sound_event_detection \
        --model "${MODEL}" \
        --dataset misophonia \
        --root_dataset_dir /data \
        --sr 16000 --duration 5 --samples 2000 \
        --num_noise_min 1 --num_noise_max 1 \
        --find_thresholds --val_samples 2000 \
        --output_dir /output \
        ${EXTRA_ARGS}
else
    echo "Unknown task: ${TASK} (use tse or sed)"
    exit 1
fi
