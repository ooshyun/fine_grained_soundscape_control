#!/usr/bin/env bash
# Table 4, Figure 4: SED Evaluation (Fine-tuned AST)
#
# Reproduces paper Table 4 metrics: Accuracy, Precision, Recall, F1
# Method: per-class F1-maximizing thresholds found on val set → applied to test set
#
# 7 conditions from paper §4.2:
#   tgt={1,2,3,4,5} × bg=1-1, noise=1-1
#   tgt={1,5}       × bg=1-3, noise=1-1
#
# Usage:
#   bash scripts/eval/run_sed.sh <data_dir> [output_dir] [--thresholds <path>]
#
# Examples:
#   # Full eval with val→threshold→test (paper method)
#   bash scripts/eval/run_sed.sh /scr
#
#   # Use pre-computed thresholds
#   bash scripts/eval/run_sed.sh /scr eval_results/table4_sed --thresholds configs/sed/optimal_thresholds.json
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/run_sed.sh <data_dir> [output_dir] [--thresholds <path>]}"
OUTPUT_DIR="${2:-eval_results/table4_sed}"
shift 2 2>/dev/null || shift 1

# Parse optional --thresholds flag
THRESHOLD_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds) THRESHOLD_ARG="--thresholds $2"; shift 2 ;;
        *) shift ;;
    esac
done

COMMON_ARGS="--pretrained ooshyun/sound_event_detection --model finetuned_ast \
    --dataset misophonia --root_dataset_dir ${DATA_DIR} \
    --sr 16000 --duration 5 --samples 2000 \
    --num_noise_min 1 --num_noise_max 1"

# If no pre-computed thresholds, use --find_thresholds (val→threshold→test)
if [[ -z "$THRESHOLD_ARG" ]]; then
    THRESHOLD_ARG="--find_thresholds --val_samples 2000"
fi

echo "============================================================"
echo "  Table 4: SED Evaluation (Fine-tuned AST)"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Condition 1-5: tgt={1..5}, bg=1-1
for tgt in 1 2 3 4 5; do
    echo "--- tgt=${tgt}, bg=1-1 ---"
    python -m src.sed.eval ${COMMON_ARGS} \
        --num_fg_min ${tgt} --num_fg_max ${tgt} \
        --num_bg_min 1 --num_bg_max 1 \
        ${THRESHOLD_ARG} \
        --output_dir "${OUTPUT_DIR}/tgt${tgt}_bg1-1"
    echo ""
done

# Condition 6: tgt=1, bg=1-3
echo "--- tgt=1, bg=1-3 ---"
python -m src.sed.eval ${COMMON_ARGS} \
    --num_fg_min 1 --num_fg_max 1 \
    --num_bg_min 1 --num_bg_max 3 \
    ${THRESHOLD_ARG} \
    --output_dir "${OUTPUT_DIR}/tgt1_bg1-3"
echo ""

# Condition 7: tgt=5, bg=1-3
echo "--- tgt=5, bg=1-3 ---"
python -m src.sed.eval ${COMMON_ARGS} \
    --num_fg_min 5 --num_fg_max 5 \
    --num_bg_min 1 --num_bg_max 3 \
    ${THRESHOLD_ARG} \
    --output_dir "${OUTPUT_DIR}/tgt5_bg1-3"
echo ""

echo "✓ Table 4 results saved to ${OUTPUT_DIR}/"
echo ""
echo "Results per condition:"
for d in "${OUTPUT_DIR}"/*/; do
    if [[ -f "${d}/metrics.json" ]]; then
        echo "  $(basename $d): $(cat ${d}/metrics.json)"
    fi
done
