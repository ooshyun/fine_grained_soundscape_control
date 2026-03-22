#!/usr/bin/env bash
# Run training inside Docker
#
# Usage:
#   bash docker/train_single.sh <tse|sed> <config> <data_dir> [output_dir]
#
# Examples:
#   bash docker/train_single.sh tse configs/tse/orange_pi.yaml /path/to/data
#   bash docker/train_single.sh sed configs/sed/finetuned_ast.yaml /path/to/data ./runs
set -euo pipefail

TASK="${1:?Usage: bash docker/train_single.sh <tse|sed> <config> <data_dir> [output_dir]}"
CONFIG="${2:?Specify config file}"
DATA_DIR="${3:?Specify data directory}"
OUTPUT_DIR="${4:-./runs}"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

IMAGE="fine-grained-soundscape"

docker run --rm --gpus all \
    -v "${DATA_DIR}:/data:ro" \
    -v "${OUTPUT_DIR}:/output" \
    "${IMAGE}" "src.${TASK}.train" \
    --config "${CONFIG}" \
    --data_dir /data \
    --output_dir /output
