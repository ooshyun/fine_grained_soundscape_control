#!/usr/bin/env bash
# Figure 4: SED Baseline Comparison
#
# Compares fine-tuned AST vs pretrained AST (AudioSet 527-class baseline)
# Single condition: tgt=1, bg=1-1 (simplest case for baseline comparison)
#
# Usage:
#   bash scripts/eval/run_sed_baselines.sh <data_dir> [output_dir]
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_sed_baselines.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-eval_results/fig4_sed_baselines}"

COMMON_ARGS="--dataset misophonia --root_dataset_dir ${DATA_DIR} \
    --sr 16000 --duration 5 --samples 2000 \
    --num_fg_min 1 --num_fg_max 1 \
    --num_bg_min 1 --num_bg_max 1 \
    --num_noise_min 1 --num_noise_max 1"

echo "============================================================"
echo "  Figure 4: SED Baseline Comparison"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Fine-tuned AST (our model)
echo "--- Fine-tuned AST ---"
python -m src.sed.eval \
    --pretrained ooshyun/sound_event_detection \
    --model finetuned_ast \
    ${COMMON_ARGS} \
    --find_thresholds --val_samples 2000 \
    --output_dir "${OUTPUT_DIR}/finetuned_ast"
echo ""

# Pretrained AST baseline (AudioSet 527 classes, no fine-tuning)
echo "--- Pretrained AST (baseline) ---"
python -m src.sed.eval \
    --pretrained ooshyun/sound_event_detection \
    --model ast_pretrained \
    ${COMMON_ARGS} \
    --find_thresholds --val_samples 2000 \
    --output_dir "${OUTPUT_DIR}/ast_pretrained"
echo ""

echo "✓ Baseline comparison saved to ${OUTPUT_DIR}/"
