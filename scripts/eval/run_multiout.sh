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

# 5ch/5spk/5out (Orange Pi)
echo "--- Orange Pi 5-out ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_5out"
echo ""

# 20ch/5spk/20out (Orange Pi)
echo "--- Orange Pi 20-out ---"
python -m src.tse.eval \
    --pretrained ooshyun/semantic_listening \
    --model tfgridnet_large_snr_ctl_v2_20ch_5spk_20out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/orange_pi_20out"
echo ""

echo "✓ Table 2 results saved to ${OUTPUT_DIR}/"
