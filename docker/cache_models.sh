#!/usr/bin/env bash
# Pre-download all models needed for evaluation into a local cache directory.
# This cache is mounted read-only into the reviewer container.
#
# Usage:
#   bash docker/cache_models.sh [output_dir]
#
# Example:
#   bash docker/cache_models.sh ./model_cache
set -euo pipefail

OUTPUT_DIR="${1:-./model_cache}"
IMAGE="fine-grained-soundscape"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "Caching models to ${OUTPUT_DIR}..."

docker run --rm \
    -v "${OUTPUT_DIR}:/cache" \
    "${IMAGE}" \
    python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HOME'] = '/cache'

print('Downloading TSE models...')
snapshot_download('ooshyun/fine_grained_soundscape_control', cache_dir='/cache')

print('Downloading SED models...')
snapshot_download('ooshyun/sound_event_detection', cache_dir='/cache')

print('Downloading AST pretrained...')
from transformers import AutoFeatureExtractor, ASTForAudioClassification
AutoFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593', cache_dir='/cache')
ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593', cache_dir='/cache')

print('All models cached!')
"

echo "Done! Cache size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
