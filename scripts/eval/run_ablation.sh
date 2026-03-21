#!/usr/bin/env bash
# Table 3: FiLM Ablation (Orange Pi + NeuralAids × first/all/all-except-first)
# Reproduces paper Table 3 metrics
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_ablation.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-eval_results/table3_ablation}"

echo "Table 3: FiLM Ablation"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

echo "=== Orange Pi ==="
for variant in film_first film_all film_all_except_first; do
    echo "--- orange_pi_${variant} ---"
    python -m src.tse.eval \
        --pretrained ooshyun/semantic_listening \
        --model "orange_pi_${variant}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/orange_pi_${variant}"
    echo ""
done

echo "=== NeuralAids ==="
for variant in film_first film_all film_all_except_first; do
    echo "--- neuralaid_${variant} ---"
    python -m src.tse.eval \
        --pretrained ooshyun/semantic_listening \
        --model "neuralaid_${variant}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/neuralaid_${variant}"
    echo ""
done

echo "✓ Table 3 results saved to ${OUTPUT_DIR}/"
