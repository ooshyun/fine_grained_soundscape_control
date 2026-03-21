#!/usr/bin/env bash
# Quick train sanity check — runs a few steps to verify loss decreases.
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_train_test.sh <data_dir> [config]}"
CONFIG="${2:-configs/tse/orange_pi.yaml}"

echo "Train Sanity Check"
echo "Data: ${DATA_DIR}"
echo "Config: ${CONFIG}"
echo ""

python -m src.tse.train \
    --config "${CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --max_steps 50 \
    --output_dir runs/train_test
