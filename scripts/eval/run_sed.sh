#!/usr/bin/env bash
# Table 4, Figure 4: SED Evaluation (Fine-tuned AST)
# Reproduces paper Table 4 metrics (mAP, F1)
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_sed.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-eval_results/table4_sed}"

echo "Table 4: SED Evaluation"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python -m src.sed.eval \
    --pretrained ooshyun/sound_event_detection \
    --model finetuned_ast \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "✓ Table 4 results saved to ${OUTPUT_DIR}/"
