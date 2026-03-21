#!/usr/bin/env bash
# Train SED model (Fine-tuned AST)
#
# Usage:
#   bash scripts/train/run_sed.sh <data_dir> [config] [extra args...]
#
# Examples:
#   bash scripts/train/run_sed.sh /path/to/output                        # AST finetune (default)
#   bash scripts/train/run_sed.sh /path/to/output ast_finetune
set -euo pipefail

DATA_DIR="${1:?Usage: bash scripts/train/run_sed.sh <data_dir> [config] [extra args...]}"
CONFIG="${2:-ast_finetune}"
shift 2 2>/dev/null || shift 1

echo "Train SED: ${CONFIG}"
echo "Data: ${DATA_DIR}"
echo ""

python -m src.sed.train \
    --config "configs/sed/${CONFIG}.yaml" \
    --data_dir "${DATA_DIR}" \
    "$@"
