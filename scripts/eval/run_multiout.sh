#!/usr/bin/env bash
# Table 2: Multi-output TSE (Orange Pi, 5-out vs 20-out)
# Reproduces paper Table 2 metrics
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_multiout.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-eval_results/table2_multiout}"

echo "Table 2: Multi-output TSE"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

echo "--- Orange Pi 5-out ---"
python -m src.tse.eval \
    --pretrained ooshyun/fine_grained_soundscape_control \
    --model orange_pi_5out \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_5out"
echo ""

echo "--- Orange Pi 20-out ---"
python -m src.tse.eval \
    --pretrained ooshyun/fine_grained_soundscape_control \
    --model orange_pi_20out \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_20out"
echo ""

echo "✓ Table 2 results saved to ${OUTPUT_DIR}/"
