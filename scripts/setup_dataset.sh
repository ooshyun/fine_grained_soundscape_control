#!/usr/bin/env bash
# =================================================================
# Dataset Setup Script
# =================================================================
#
# Downloads and prepares BinauralCuratedDataset for training and eval.
#
# The public tar does NOT include noise_scaper_fmt/ (TAU noise symlinks),
# so this script builds it after extraction.
#
# Usage:
#   # Full setup (download + extract + build noise)
#   bash scripts/setup_dataset.sh --output_dir /path/to/output
#
#   # If you already have the tar extracted:
#   bash scripts/setup_dataset.sh --output_dir /path/to/output --skip_download
#
#   # If TAU raw data is at a separate path:
#   bash scripts/setup_dataset.sh --output_dir /path/to/output \
#       --tau_raw_dir /path/to/TAU-2019
#
# After setup, the output directory will contain:
#   output_dir/
#     BinauralCuratedDataset/
#       scaper_fmt/{train,val,test}/{class}/     (fg audio symlinks)
#       bg_scaper_fmt/{train,val,test}/{class}/  (bg audio symlinks)
#       noise_scaper_fmt/{train,val,test}/{scene}/ (TAU noise symlinks)
#       hrtf/CIPIC/{*.sofa, {train,val,test}_hrtf.txt}
#       FSD50K/, ESC-50/, musdb18/, disco_noises/, TAU-acoustic-sounds/
#       start_times.csv
#
# For training:
#   python -m src.tse.train --config configs/tse/orange_pi.yaml \
#       --data_dir /path/to/output
#
# For eval:
#   python -m src.tse.eval --pretrained ooshyun/semantic_listening \
#       --model orange_pi --data_dir /path/to/output
# =================================================================

set -euo pipefail

TAR_URL="https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar"
TAR_NAME="BinauralCuratedDataset.tar"
DATASET_DIR="BinauralCuratedDataset"

# ---- Parse arguments ----
OUTPUT_DIR=""
TAU_RAW_DIR=""
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tau_raw_dir) TAU_RAW_DIR="$2"; shift 2 ;;
        --skip_download) SKIP_DOWNLOAD=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash scripts/setup_dataset.sh --output_dir /path/to/output"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${OUTPUT_DIR}/${DATASET_DIR}"

echo "============================================================"
echo "  Dataset Setup"
echo "  Output: ${OUTPUT_DIR}"
echo "  Dataset: ${DATA_DIR}"
echo "============================================================"

# ---- Step 1: Download + Extract ----
if [[ "$SKIP_DOWNLOAD" == false ]]; then
    mkdir -p "$OUTPUT_DIR"
    cd "$OUTPUT_DIR"

    if [[ -d "$DATASET_DIR" ]]; then
        echo "[Step 1] Dataset already extracted, skipping download."
    elif [[ -f "$TAR_NAME" ]]; then
        echo "[Step 1] Tar exists, extracting..."
        tar xf "$TAR_NAME"
        echo "  ✓ Extracted"
    else
        echo "[Step 1] Downloading ${TAR_URL}..."
        wget -q --show-progress "$TAR_URL"
        echo "  ✓ Downloaded"
        echo "  Extracting..."
        tar xf "$TAR_NAME"
        echo "  ✓ Extracted"
    fi
else
    echo "[Step 1] Skipped (--skip_download)"
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: ${DATA_DIR} not found after extraction."
    exit 1
fi

# ---- Step 2: Build noise_scaper_fmt ----
if [[ -d "${DATA_DIR}/noise_scaper_fmt/train" ]]; then
    echo "[Step 2] noise_scaper_fmt already exists, skipping."
else
    echo "[Step 2] Building noise_scaper_fmt..."

    TAU_ARG=""
    if [[ -n "$TAU_RAW_DIR" ]]; then
        TAU_ARG="--tau_raw_dir ${TAU_RAW_DIR}"
    fi

    cd "$REPO_DIR"
    python scripts/build_noise_scaper_fmt.py \
        --data_dir "$DATA_DIR" \
        $TAU_ARG

    echo "  ✓ noise_scaper_fmt built"
fi

# ---- Step 3: Verify ----
echo ""
echo "============================================================"
echo "  Verification"
echo "============================================================"

verify() {
    local path="$1"
    local desc="$2"
    if [[ -e "$path" ]]; then
        echo "  ✓ ${desc}"
    else
        echo "  ✗ ${desc} — MISSING: ${path}"
    fi
}

verify "${DATA_DIR}/scaper_fmt/test" "scaper_fmt/test (fg)"
verify "${DATA_DIR}/bg_scaper_fmt/test" "bg_scaper_fmt/test (bg)"
verify "${DATA_DIR}/noise_scaper_fmt/test" "noise_scaper_fmt/test (noise)"
verify "${DATA_DIR}/hrtf/CIPIC/test_hrtf.txt" "hrtf/CIPIC/test_hrtf.txt"
verify "${DATA_DIR}/start_times.csv" "start_times.csv"

FG_CLASSES=$(ls "${DATA_DIR}/scaper_fmt/test/" 2>/dev/null | wc -l)
echo ""
echo "  FG test classes: ${FG_CLASSES}"
echo ""

if [[ "$FG_CLASSES" -ge 20 ]]; then
    echo "✓ Dataset ready!"
    echo ""
    echo "Train:"
    echo "  python -m src.tse.train --config configs/tse/orange_pi.yaml \\"
    echo "      --data_dir ${OUTPUT_DIR}"
    echo ""
    echo "Eval:"
    echo "  python -m src.tse.eval --pretrained ooshyun/semantic_listening \\"
    echo "      --model orange_pi --data_dir ${OUTPUT_DIR}"
else
    echo "⚠ Dataset incomplete (expected ≥20 FG classes, got ${FG_CLASSES})"
fi
