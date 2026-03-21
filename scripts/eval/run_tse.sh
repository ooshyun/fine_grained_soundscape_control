#!/usr/bin/env bash
# Table 1: TSE Model Comparison (Orange Pi, Raspberry Pi, NeuralAids)
# Reproduces paper Table 1 metrics (SNRi, SI-SDRi)
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_tse.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-eval_results/table1_tse}"

echo "Table 1: TSE Model Comparison"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

for model in orange_pi raspberry_pi neuralaid; do
    echo "--- ${model} ---"
    python -m src.tse.eval \
        --pretrained ooshyun/semantic_listening \
        --model "${model}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/${model}"
    echo ""
done

echo "✓ Table 1 results saved to ${OUTPUT_DIR}/"
