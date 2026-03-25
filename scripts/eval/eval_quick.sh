#!/usr/bin/env bash
# Quick Evaluation — subset of paper results (~1.5 hours on single GPU)
#
# TSE: Table 3 FiLM=All variants (orange_pi, neuralaid) × 2000 samples
# SED: Best case (tgt=1) + Worst case (tgt=5) × 2000 samples
#
# Usage:
#   bash scripts/eval/eval_quick.sh <data_dir> [output_dir]
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/eval/eval_quick.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-eval_results/quick}"

echo "============================================================"
echo "  Quick Evaluation (subset of paper Tables 1, 3, 4)"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Started: $(date)"
echo "============================================================"

# ============================================================
# TSE: FiLM=All (3 models from Table 3, also covers Table 1)
# ============================================================
echo ""
echo "========== TSE: FiLM=All Models (Table 1 & 3) =========="
for model in orange_pi neuralaid; do
    model_name="${model}_film_all"
    echo ""
    echo "--- ${model_name} --- $(date)"
    python -m src.tse.eval \
        --pretrained ooshyun/fine_grained_soundscape_control \
        --model "${model_name}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/tse/${model_name}"
done

# ============================================================
# SED: Best case + Worst case (Table 4)
# ============================================================
echo ""
echo "========== SED: Best & Worst Cases (Table 4) =========="
SED_COMMON="--pretrained ooshyun/sound_event_detection --model finetuned_ast \
    --dataset misophonia --root_dataset_dir ${DATA_DIR} \
    --sr 16000 --duration 5 --samples 2000 \
    --num_noise_min 1 --num_noise_max 1 \
    --find_thresholds --val_samples 2000"

echo ""
echo "--- Best case: tgt=1 --- $(date)"
python -m src.sed.eval ${SED_COMMON} \
    --num_fg_min 1 --num_fg_max 1 \
    --num_bg_min 1 --num_bg_max 1 \
    --output_dir "${OUTPUT_DIR}/sed/tgt1"

echo ""
echo "--- Worst case: tgt=5 --- $(date)"
python -m src.sed.eval ${SED_COMMON} \
    --num_fg_min 5 --num_fg_max 5 \
    --num_bg_min 1 --num_bg_max 1 \
    --output_dir "${OUTPUT_DIR}/sed/tgt5"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  RESULTS — $(date)"
echo "============================================================"

echo ""
echo "=== TSE (Table 1 & 3: FiLM=All) ==="
echo "  Paper values: OrangePi SNRi=12.26, NeuralAids SNRi=10.50"
for model in orange_pi neuralaid; do
    f="${OUTPUT_DIR}/tse/${model}_film_all/metrics_total_averages.json"
    [ -f "$f" ] && python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${model}_film_all: SNRi={d[\"snr_i\"]:.2f}, SI-SDRi={d[\"si_sdr_i\"]:.2f}')
" || echo "  ${model}_film_all: N/A"
done

echo ""
echo "=== SED (Table 4) ==="
echo "  Paper values: tgt=1 Acc=0.992/F1=0.921, tgt=5 Acc=0.923/F1=0.838"
for d in "${OUTPUT_DIR}"/sed/*/; do
    name=$(basename "$d")
    f="${d}/metrics.json"
    [ -f "$f" ] && python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${name}: Acc={d[\"accuracy\"]:.3f}, Prec={d[\"precision\"]:.3f}, Rec={d[\"recall\"]:.3f}, F1={d[\"f1\"]:.3f}')
" || echo "  ${name}: N/A"
done

echo ""
echo "Done! $(date)"
