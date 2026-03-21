#!/usr/bin/env bash
# Submit SED paper eval (softmax scores + val thresholds) to HPC
set -euo pipefail

REMOTE_HOST="klone-login"
REMOTE_REPO="/mmfs1/home/shoh10/shoh10/workspace/project/sementic_listening/fine_grained_soundscape_control_for_augmented_hearing"
LOG_DIR="/gscratch/scrubbed/shoh10/cache/logs"
DATASET_TAR="/gscratch/intelligentsystems/common_datasets/SemanticListening/BinauralCuratedDataset.tar"

PARTITION="gpu-l40s"
ACCOUNT="intelligentsystems"

# Sync code
echo "Syncing code..."
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.venv' --exclude='runs' --exclude='eval_results' \
    --exclude='lightning_logs' \
    "${LOCAL_REPO}/src/" "${REMOTE_HOST}:${REMOTE_REPO}/src/"
rsync -avz "${LOCAL_REPO}/configs/" "${REMOTE_HOST}:${REMOTE_REPO}/configs/"
rsync -avz "${LOCAL_REPO}/data/" "${REMOTE_HOST}:${REMOTE_REPO}/data/"
rsync -avz "${LOCAL_REPO}/scripts/eval/" "${REMOTE_HOST}:${REMOTE_REPO}/scripts/eval/"
echo "Code synced."

JOB_SCRIPT=$(cat <<'JOBEOF'
#!/usr/bin/env bash
#SBATCH --job-name=sed_softmax_eval
#SBATCH --partition=gpu-l40s
#SBATCH --account=intelligentsystems
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --exclude=g3110
#SBATCH --output=/gscratch/scrubbed/shoh10/cache/logs/sed_softmax_eval_%j.log
#SBATCH --error=/gscratch/scrubbed/shoh10/cache/logs/sed_softmax_eval_%j.err

set -euo pipefail

echo "=== SED Eval (softmax scores + val thresholds) ==="
echo "Host: $(hostname), Date: $(date), Job: ${SLURM_JOB_ID:-N/A}"

module load cuda/12.4 2>/dev/null || true
export HF_HOME="/gscratch/scrubbed/shoh10/cache/huggingface"
export TORCH_HOME="/gscratch/scrubbed/shoh10/cache/torch"
export XDG_CACHE_HOME="/gscratch/scrubbed/shoh10/cache"

# Extract dataset
if [[ -d "/scr/BinauralCuratedDataset/scaper_fmt" ]]; then
    echo "Dataset already extracted."
else
    tar xf /gscratch/intelligentsystems/common_datasets/SemanticListening/BinauralCuratedDataset.tar -C /scr/
fi

cd /mmfs1/home/shoh10/shoh10/workspace/project/sementic_listening/fine_grained_soundscape_control_for_augmented_hearing
source .venv/bin/activate
echo "Python: $(python --version)"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true

# Run all 7 conditions with val→threshold→test
bash scripts/eval/run_sed.sh /scr eval_results/table4_softmax

# Summary
echo ""
echo "=== SUMMARY ==="
python -c "
import json, os

conditions = [
    ('tgt1_bg1-1', 'tgt=1 bg=1-1', {'accuracy': 0.992, 'precision': 0.909, 'recall': 0.934, 'f1': 0.921}),
    ('tgt2_bg1-1', 'tgt=2 bg=1-1', {'accuracy': 0.971, 'precision': 0.796, 'recall': 0.901, 'f1': 0.845}),
    ('tgt3_bg1-1', 'tgt=3 bg=1-1', {'accuracy': 0.958, 'precision': 0.804, 'recall': 0.903, 'f1': 0.851}),
    ('tgt4_bg1-1', 'tgt=4 bg=1-1', {'accuracy': 0.943, 'precision': 0.800, 'recall': 0.904, 'f1': 0.849}),
    ('tgt5_bg1-1', 'tgt=5 bg=1-1', {'accuracy': 0.923, 'precision': 0.797, 'recall': 0.884, 'f1': 0.838}),
    ('tgt1_bg1-3', 'tgt=1 bg=1-3', {}),
    ('tgt5_bg1-3', 'tgt=5 bg=1-3', {}),
]

print(f'{\"Condition\":<16} {\"Acc\":>7} {\"(ppr)\":>7} {\"Prec\":>7} {\"(ppr)\":>7} {\"Rec\":>7} {\"(ppr)\":>7} {\"F1\":>7} {\"(ppr)\":>7}')
print('-' * 80)
for tag, label, paper in conditions:
    path = f'eval_results/table4_softmax/{tag}/metrics.json'
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        fmt = lambda v: f'{v:>7.3f}' if v else f'{\"\":>7}'
        print(f'{label:<16} {d.get(\"accuracy\",0):>7.3f} {fmt(paper.get(\"accuracy\"))} {d.get(\"precision\",0):>7.3f} {fmt(paper.get(\"precision\"))} {d.get(\"recall\",0):>7.3f} {fmt(paper.get(\"recall\"))} {d.get(\"f1\",0):>7.3f} {fmt(paper.get(\"f1\"))}')
    else:
        print(f'{label:<16} MISSING')
"
echo ""
echo "Done at $(date)"
JOBEOF
)

echo "Submitting..."
ssh "$REMOTE_HOST" "mkdir -p $LOG_DIR"
JOB_ID=$(echo "$JOB_SCRIPT" | ssh "$REMOTE_HOST" "sbatch --parsable")

echo ""
echo "Job submitted: $JOB_ID"
echo "Monitor: ssh $REMOTE_HOST 'tail -f $LOG_DIR/sed_softmax_eval_${JOB_ID}.log'"
echo "Fetch:   rsync -avz ${REMOTE_HOST}:${REMOTE_REPO}/eval_results/table4_softmax/ eval_results/table4_softmax/"
