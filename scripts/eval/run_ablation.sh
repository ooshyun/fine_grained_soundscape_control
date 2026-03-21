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

# Orange Pi variants
echo "=== Orange Pi ==="

echo "--- FiLM: first ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_first_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_film_first"
echo ""

echo "--- FiLM: all ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_film_all"
echo ""

echo "--- FiLM: all-except-first ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_film_all_except_first"
echo ""

# NeuralAids variants
echo "=== NeuralAids ==="

echo "--- FiLM: first ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfmlpnet_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_first_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/neuralaid_film_first"
echo ""

echo "--- FiLM: all ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfmlpnet_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_layers_6_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/neuralaid_film_all"
echo ""

echo "--- FiLM: all-except-first ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfmlpnet_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/neuralaid_film_all_except_first"
echo ""

echo "✓ Table 3 results saved to ${OUTPUT_DIR}/"
