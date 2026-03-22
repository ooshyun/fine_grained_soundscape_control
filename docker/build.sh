#!/usr/bin/env bash
# Build Docker image for fine-grained soundscape control
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="fine-grained-soundscape"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${REPO_DIR}"
echo "Done: ${IMAGE_NAME}"
