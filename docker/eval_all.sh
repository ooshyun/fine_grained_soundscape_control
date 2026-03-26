#!/usr/bin/env bash
# Run all paper evaluations (Tables 1-4) inside Docker
#
# Usage:
#   bash docker/eval_all.sh /path/to/data [/path/to/output]
#
# Requirements:
#   - Docker with NVIDIA GPU support (nvidia-container-toolkit)
#   - Dataset at DATA_DIR (BinauralCuratedDataset extracted)
#   - Run docker/build.sh first
#
# Output structure:
#   <output_dir>/
#   ├── table1/  (TSE: orange_pi, raspberry_pi, neuralaid, waveformer)
#   ├── table2/  (Multi-output: 5out, 20out)
#   ├── table3/  (FiLM ablation: 6 variants)
#   └── table4/  (SED: 7 conditions)
set -euo pipefail

DATA_DIR="${1:?Usage: bash docker/eval_all.sh <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-./eval_results}"
IMAGE="fine-grained-soundscape"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

DOCKER_RUN="docker run --rm --gpus all \
    -v ${DATA_DIR}:/data:ro \
    -v ${OUTPUT_DIR}:/output \
    ${IMAGE}"

echo "============================================================"
echo "  Paper Evaluation (Docker)"
echo "  Image: ${IMAGE}"
echo "  Data:  ${DATA_DIR} -> /data"
echo "  Output: ${OUTPUT_DIR} -> /output"
echo "  Started: $(date)"
echo "============================================================"

# ============================================================
# Table 1: TSE Model Comparison (4 models)
# ============================================================
echo ""
echo "========== TABLE 1: TSE Model Comparison =========="
for model in orange_pi raspberry_pi neuralaid waveformer; do
    echo ""
    echo "--- ${model} ---"
    ${DOCKER_RUN} src.tse.eval \
        --pretrained ooshyun/fine_grained_soundscape_control \
        --model "${model}" \
        --data_dir /data \
        --output_dir "/output/table1/${model}"
done

# ============================================================
# Table 2: Multi-output TSE (2 models)
# ============================================================
echo ""
echo "========== TABLE 2: Multi-output TSE =========="
for model in orange_pi_5out orange_pi_20out; do
    echo ""
    echo "--- ${model} ---"
    ${DOCKER_RUN} src.tse.eval \
        --pretrained ooshyun/fine_grained_soundscape_control \
        --model "${model}" \
        --data_dir /data \
        --output_dir "/output/table2/${model}"
done

# ============================================================
# Table 3: FiLM Ablation (6 models)
# ============================================================
echo ""
echo "========== TABLE 3: FiLM Ablation =========="
for arch in orange_pi neuralaid; do
    for variant in film_first film_all film_all_except_first; do
        model="${arch}_${variant}"
        echo ""
        echo "--- ${model} ---"
        ${DOCKER_RUN} src.tse.eval \
            --pretrained ooshyun/fine_grained_soundscape_control \
            --model "${model}" \
            --data_dir /data \
            --output_dir "/output/table3/${model}"
    done
done

# ============================================================
# Table 4: SED (7 conditions)
# ============================================================
echo ""
echo "========== TABLE 4: SED Evaluation =========="
SED_COMMON="src.sed.eval \
    --pretrained ooshyun/sound_event_detection --model finetuned_ast \
    --dataset misophonia --root_dataset_dir /data \
    --sr 16000 --duration 5 --samples 2000 \
    --num_noise_min 1 --num_noise_max 1 \
    --find_thresholds --val_samples 2000"

for tgt in 1 2 3 4 5; do
    echo ""
    echo "--- tgt=${tgt}, bg=1-2 ---"
    ${DOCKER_RUN} ${SED_COMMON} \
        --num_fg_min ${tgt} --num_fg_max ${tgt} \
        --num_bg_min 1 --num_bg_max 2 \
        --output_dir "/output/table4/tgt${tgt}_bg1-2"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  SUMMARY — $(date)"
echo "============================================================"

echo ""
echo "=== Table 1: TSE ==="
for model in orange_pi raspberry_pi neuralaid waveformer; do
    f="${OUTPUT_DIR}/table1/${model}/metrics_total_averages.json"
    [ -f "$f" ] && python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${model}: SNRi={d[\"snr_i\"]:.2f}, SI-SDRi={d[\"si_sdr_i\"]:.2f}')
" || echo "  ${model}: N/A"
done

echo ""
echo "=== Table 2: Multi-output ==="
for model in orange_pi_5out orange_pi_20out; do
    f="${OUTPUT_DIR}/table2/${model}/metrics_total_averages.json"
    [ -f "$f" ] && python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${model}: SNRi={d[\"snr_i\"]:.2f}, SI-SDRi={d[\"si_sdr_i\"]:.2f}')
" || echo "  ${model}: N/A"
done

echo ""
echo "=== Table 3: FiLM Ablation ==="
for arch in orange_pi neuralaid; do
    for variant in film_first film_all film_all_except_first; do
        model="${arch}_${variant}"
        f="${OUTPUT_DIR}/table3/${model}/metrics_total_averages.json"
        [ -f "$f" ] && python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${model}: SNRi={d[\"snr_i\"]:.2f}, SI-SDRi={d[\"si_sdr_i\"]:.2f}')
" || echo "  ${model}: N/A"
    done
done

echo ""
echo "=== Table 4: SED ==="
for d in "${OUTPUT_DIR}"/table4/*/; do
    cond=$(basename "$d")
    f="${d}/metrics.json"
    [ -f "$f" ] && echo "  ${cond}: $(cat $f)" || echo "  ${cond}: N/A"
done

echo ""
echo "Done!"
