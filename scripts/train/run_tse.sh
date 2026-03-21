#!/usr/bin/env bash
# Train TSE model
#
# Usage:
#   bash scripts/train/run_tse.sh <data_dir> [config] [extra args...]
#
# Examples:
#   bash scripts/train/run_tse.sh /path/to/output                        # Orange Pi (default)
#   bash scripts/train/run_tse.sh /path/to/output orange_pi              # explicit config
#   bash scripts/train/run_tse.sh /path/to/output raspberry_pi
#   bash scripts/train/run_tse.sh /path/to/output neuralaid
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/train/run_tse.sh <data_dir> [config] [extra args...]}"
CONFIG="${2:-orange_pi}"
shift 2 2>/dev/null || shift 1

echo "Train TSE: ${CONFIG}"
echo "Data: ${DATA_DIR}"
echo ""

python -m src.tse.train \
    --config "configs/tse/${CONFIG}.yaml" \
    --data_dir "${DATA_DIR}" \
    "$@"
