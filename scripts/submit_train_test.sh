#!/usr/bin/env bash
# submit_train_test.sh — Submit a pipeline test job (train 20 epochs + eval)
#
# Usage (from local machine):
#   ./scripts/submit_train_test.sh
#
# Runs on: klone-login → sbatch on gpu-a40 (1 GPU)
set -euo pipefail

REMOTE_HOST="klone-login"
REMOTE_WORKSPACE="/mmfs1/home/shoh10/shoh10/workspace/project/sementic_listening/fine_grained_soundscape_control_for_augmented_hearing"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="configs/tse/test_pipeline.yaml"
JOB_NAME="tse_pipeline_test"
PARTITION="gpu-l40s"
ACCOUNT="intelligentsystems"
GPUS="1"
CPUS="8"
MEM="32G"
WALLTIME="04:00:00"

LOG_DIR="/gscratch/scrubbed/shoh10/cache/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " TSE Pipeline Test — Train + Eval"
echo "============================================"
echo "  Config    : ${CONFIG}"
echo "  Partition : ${PARTITION}"
echo "  GPUs      : ${GPUS}"
echo "  Walltime  : ${WALLTIME}"
echo "============================================"

# ── Sync local code to remote ──
echo ""
echo "Syncing code to ${REMOTE_HOST}:${REMOTE_WORKSPACE}..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='runs/' \
    --exclude='eval_results/' \
    --exclude='*.pyc' \
    "${LOCAL_REPO}/" \
    "${REMOTE_HOST}:${REMOTE_WORKSPACE}/"
echo "Sync complete."

# ── Build the sbatch script on the remote host ──
echo ""
echo "Submitting SLURM job..."

ssh "${REMOTE_HOST}" bash <<REMOTE_EOF
set -euo pipefail
mkdir -p ${LOG_DIR}

cat > /tmp/tse_pipeline_test.sh <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

set -euo pipefail

echo "============================================"
echo " TSE Pipeline Test Job"
echo "============================================"
echo "  Host      : \$(hostname)"
echo "  Date      : \$(date)"
echo "  Job ID    : \${SLURM_JOB_ID}"
echo "  Partition : \${SLURM_JOB_PARTITION}"
echo "  GPUs      : \${SLURM_GPUS}"
echo "============================================"

# ── Load CUDA ──
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
NVCC_PATH=\$(which nvcc 2>/dev/null || true)
if [[ -n "\$NVCC_PATH" ]]; then
    CUDA_BASE=\$(dirname "\$(dirname "\$NVCC_PATH")")
    export CUDA_HOME="\$CUDA_BASE"
    export LD_LIBRARY_PATH="\${CUDA_BASE}/lib64:\${LD_LIBRARY_PATH:-}"
    echo "  CUDA_HOME : \$CUDA_HOME"
fi

# ── Environment ──
export UV_CACHE_DIR="/gscratch/scrubbed/shoh10/cache/uv-cache"
export HF_HOME="/gscratch/scrubbed/shoh10/cache/huggingface"
export TORCH_HOME="/gscratch/scrubbed/shoh10/cache/torch"
export XDG_CACHE_HOME="/gscratch/scrubbed/shoh10/cache"
export UV_PYTHON_INSTALL_DIR="/gscratch/scrubbed/shoh10/cache/uv-python/sementic-listening"
export PYTHONUNBUFFERED=1

# ── Navigate to workspace ──
cd "${REMOTE_WORKSPACE}"

# ── Setup Python ──
if [[ ! -d ".venv" ]]; then
    echo "Creating venv..."
    uv python install 3.11
    uv venv --python 3.11
    uv pip install -r requirements.txt
else
    echo "Existing .venv found, verifying..."
    if ! uv run python -c "import torch; print('OK:', torch.__version__)" 2>/dev/null; then
        echo "Python broken, reinstalling..."
        rm -rf .venv
        uv python install 3.11
        uv venv --python 3.11
        uv pip install -r requirements.txt
    fi
fi

# ── Verify GPU ──
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" || true

# ── Extract BinauralCuratedDataset to node-local /scr ──
DATASET_TAR=/mmfs1/gscratch/intelligentsystems/common_datasets/SemanticListening/BinauralCuratedDataset.tar

if [[ ! -f /scr/BinauralCuratedDataset/hrtf/CIPIC/train_hrtf.txt ]]; then
    echo ""
    echo "Extracting BinauralCuratedDataset to /scr..."
    rm -rf /scr/BinauralCuratedDataset
    tar -xf \${DATASET_TAR} -C /scr
    echo "Extraction complete."
else
    echo "BinauralCuratedDataset already exists on /scr"
fi

# Verify data structure
echo ""
echo "Data structure check:"
ls /scr/BinauralCuratedDataset/ 2>/dev/null | head -20
echo "FG train classes:"
ls /scr/BinauralCuratedDataset/scaper_fmt/train/ 2>/dev/null | head -5 || echo "  [NOT FOUND]"
echo "Noise train classes:"
ls /scr/BinauralCuratedDataset/noise_scaper_fmt/train/ 2>/dev/null | head -5 || echo "  [NOT FOUND]"
echo "HRTF files:"
ls /scr/BinauralCuratedDataset/hrtf/CIPIC/ 2>/dev/null | head -5 || echo "  [NOT FOUND]"

# ── Phase 1: Training (20 epochs) ──
echo ""
echo "============================================"
echo " Phase 1: Training (20 epochs)"
echo "============================================"
echo "  Config: ${CONFIG}"
echo ""

uv run python -m src.tse.train \
    --config ${CONFIG} \
    --data_dir /scr

echo ""
echo "Training completed at \$(date)"

# ── Phase 2: Evaluation ──
echo ""
echo "============================================"
echo " Phase 2: Evaluation"
echo "============================================"

# Find the best checkpoint from training
CKPT_DIR="runs/tse_test"
BEST_CKPT=\$(find \${CKPT_DIR} -name "best-*.ckpt" 2>/dev/null | head -1)

if [[ -z "\${BEST_CKPT}" ]]; then
    echo "No best checkpoint found in \${CKPT_DIR}, looking for any .ckpt..."
    BEST_CKPT=\$(find \${CKPT_DIR} -name "*.ckpt" 2>/dev/null | head -1)
fi

if [[ -n "\${BEST_CKPT}" ]]; then
    echo "  Checkpoint: \${BEST_CKPT}"
    echo "  Config:     ${CONFIG}"
    echo ""

    uv run python -m src.tse.eval \
        --config ${CONFIG} \
        --checkpoint "\${BEST_CKPT}" \
        --data_dir /scr \
        --output_dir runs/tse_test/eval \
        --num_samples 100

    echo ""
    echo "Evaluation completed at \$(date)"

    if [[ -f runs/tse_test/eval/metrics_total_averages.json ]]; then
        echo ""
        echo "====== Eval Results ======"
        cat runs/tse_test/eval/metrics_total_averages.json
    fi
else
    echo "ERROR: No checkpoint found in \${CKPT_DIR}"
    ls -la \${CKPT_DIR}/ 2>/dev/null || echo "  Directory not found"
fi

echo ""
echo "============================================"
echo " Pipeline Test Complete"
echo "============================================"
echo "  Finished at: \$(date)"
SBATCH_EOF

# Replace placeholder variables in the sbatch script
sed -i 's|\${JOB_NAME}|${JOB_NAME}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${ACCOUNT}|${ACCOUNT}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${PARTITION}|${PARTITION}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${GPUS}|${GPUS}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${CPUS}|${CPUS}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${MEM}|${MEM}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${WALLTIME}|${WALLTIME}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${LOG_DIR}|${LOG_DIR}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${REMOTE_WORKSPACE}|${REMOTE_WORKSPACE}|g' /tmp/tse_pipeline_test.sh
sed -i 's|\${CONFIG}|${CONFIG}|g' /tmp/tse_pipeline_test.sh

sbatch /tmp/tse_pipeline_test.sh
REMOTE_EOF

echo ""
echo "Job submitted. Monitor with:"
echo "  ssh ${REMOTE_HOST} 'squeue -u shoh10'"
echo "  ssh ${REMOTE_HOST} 'tail -f ${LOG_DIR}/${JOB_NAME}_*.out'"
