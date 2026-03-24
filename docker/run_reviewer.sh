#!/usr/bin/env bash
# Launch isolated Docker environment for artifact evaluation reviewers.
#
# Provides JupyterLab with:
#   - Dataset mounted read-only at /data
#   - Output directory at /output (writable, persisted to host)
#   - GPU access
#   - No access to host filesystem beyond /data and /output
#
# Usage:
#   bash docker/run_reviewer.sh <data_dir> [port] [token]
#
# Example:
#   bash docker/run_reviewer.sh /mnt/sda1/tmp 8888 mobisys2026ae
set -euo pipefail

DATA_DIR="${1:?Usage: bash docker/run_reviewer.sh <data_dir> [port] [token]}"
PORT="${2:-8888}"
TOKEN="${3:-mobisys2026ae}"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
OUTPUT_DIR="$(pwd)/reviewer_output"
mkdir -p "$OUTPUT_DIR"

IMAGE="fine-grained-soundscape"
CONTAINER_NAME="reviewer-eval"

echo "============================================================"
echo "  Artifact Evaluation — Reviewer Environment"
echo "============================================================"
echo "  Image:   ${IMAGE}"
echo "  Data:    ${DATA_DIR} -> /data (read-only)"
echo "  Output:  ${OUTPUT_DIR} -> /output"
echo "  Port:    ${PORT}"
echo "  Token:   ${TOKEN}"
echo ""
echo "  Access URL: http://localhost:${PORT}/lab?token=${TOKEN}"
echo "============================================================"

# Stop existing container if any
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${PORT}:8888" \
    -v "${DATA_DIR}:/data:ro" \
    -v "${OUTPUT_DIR}:/output" \
    -w /app \
    "${IMAGE}" \
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.token="${TOKEN}" \
        --ServerApp.root_dir=/app \
        --ServerApp.terminals_enabled=True
