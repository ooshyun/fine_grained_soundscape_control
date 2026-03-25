#!/usr/bin/env bash
# Launch isolated Docker environment for artifact evaluation reviewers.
#
# Provides JupyterLab with:
#   - Dataset mounted read-only at /data
#   - Pre-cached models at /root/.cache/huggingface (read-only)
#   - Output directory at /output (writable, persisted to host)
#   - GPU access
#   - No access to host filesystem, host network, or internet
#
# Prerequisites:
#   1. Build image: bash docker/build.sh
#   2. Pre-cache models: bash docker/cache_models.sh <data_dir>
#
# Usage:
#   bash docker/run_reviewer.sh <data_dir> [port] [token] [model_cache_dir]
#
# Example:
#   bash docker/run_reviewer.sh /mnt/sda1/tmp 8888 mobisys2026ae ./model_cache
set -euo pipefail

DATA_DIR="${1:?Usage: bash docker/run_reviewer.sh <data_dir> [port] [token] [model_cache_dir]}"
PORT="${2:-8888}"
TOKEN="${3:-mobisys2026ae}"
MODEL_CACHE="${4:-./model_cache}"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
OUTPUT_DIR="$(pwd)/reviewer_output"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "$MODEL_CACHE" ]; then
    echo "ERROR: Model cache not found at ${MODEL_CACHE}"
    echo "Run: bash docker/cache_models.sh ${DATA_DIR}"
    exit 1
fi
MODEL_CACHE="$(cd "$MODEL_CACHE" && pwd)"

IMAGE="fine-grained-soundscape"
CONTAINER_NAME="reviewer-eval"

echo "============================================================"
echo "  Artifact Evaluation — Reviewer Environment"
echo "============================================================"
echo "  Image:    ${IMAGE}"
echo "  Data:     ${DATA_DIR} -> /data (read-only)"
echo "  Models:   ${MODEL_CACHE} -> /root/.cache/huggingface (read-only)"
echo "  Output:   ${OUTPUT_DIR} -> /output"
echo "  Port:     ${PORT}"
echo "  Token:    ${TOKEN}"
echo ""
echo "  Security: host network blocked, internet blocked"
echo "============================================================"

# Stop existing container if any
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Start container
docker run --rm -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${PORT}:8888" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "${DATA_DIR}:/data:ro" \
    -v "${MODEL_CACHE}:/root/.cache/huggingface:ro" \
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

# Wait for container to start
sleep 3

# Apply network isolation: block host access + internet from container
CONTAINER_IP=$(docker inspect "${CONTAINER_NAME}" --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
echo ""
echo "  Container IP: ${CONTAINER_IP}"

# Block new connections from container to host gateway
sudo iptables -I INPUT -s "${CONTAINER_IP}" -i docker0 -m conntrack --ctstate NEW -j DROP
# Block internet access
sudo iptables -I FORWARD -s "${CONTAINER_IP}" ! -d 172.17.0.0/16 -j DROP

echo "  Network isolation applied"
echo ""
echo "  Access URL: http://localhost:${PORT}/lab?token=${TOKEN}"
echo ""
echo "  To stop: docker stop ${CONTAINER_NAME}"
echo "  Cleanup: sudo iptables -D INPUT -s ${CONTAINER_IP} -i docker0 -m conntrack --ctstate NEW -j DROP"
echo "           sudo iptables -D FORWARD -s ${CONTAINER_IP} ! -d 172.17.0.0/16 -j DROP"

# Follow logs
docker logs -f "${CONTAINER_NAME}"
