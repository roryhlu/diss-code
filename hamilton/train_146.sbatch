#!/bin/bash
#SBATCH --job-name=geotransformer_146
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nobackup/fwvp47/repair_training/slurm_%j.out
#SBATCH --error=/nobackup/fwvp47/repair_training/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fwvp47@durham.ac.uk

# ──────────────────────────────────────────────────────────────────────
#  RePAIR GeoTransformer — 146-fragment training on Hamilton GPU
# ──────────────────────────────────────────────────────────────────────
#
#  Trains the GeoTransformer point-completion network on all 146
#  RePAIR archaeological fragments for MC Dropout epistemic
#  uncertainty quantification and CVaR grasp filtering.
#
#  If the `gpu` partition rejects this job with an error, try:
#    sbatch --partition=gpu_devel train_146.sbatch   (short test)
#    sbatch --partition=gpu --gres=gpu:a100:1 ...    (A100 GPUs)
#    sbatch --partition=gpu --gres=gpu:v100:1 ...    (V100 GPUs)
#    sinfo -p gpu                                    (check available GPUs)
# ──────────────────────────────────────────────────────────────────────

set -e

echo "=== Job started at $(date) ==="
echo "Host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# ── Work directory (use /nobackup — no quota, not backed up) ──
WORK_DIR=/nobackup/fwvp47/repair_training
cd "$WORK_DIR"
echo "Working directory: $(pwd)"

# ── Load modules ──
module purge
module load cuda/12.1
module load python/3.11

# ── Python dependencies (first-run only) ──
pip install --user --quiet torch numpy scipy 2>&1 | tail -1

# ── Verify GPU is visible ──
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ── Ensure project path (for uncertainty.geotransformer import) ──
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# ── Fragment files ──
FRAG_DIR="$WORK_DIR/fragments"
FRAG_COUNT=$(ls "$FRAG_DIR"/*_ds.ply 2>/dev/null | wc -l)
echo "Fragments found: $FRAG_COUNT"
if [ "$FRAG_COUNT" -eq 0 ]; then
    echo "ERROR: No fragment PLYs in $FRAG_DIR"
    echo "Run setup.sh on your laptop first to transfer the data."
    exit 1
fi

# ── Train ──
echo ""
echo "=== Starting training at $(date) ==="
echo "Epochs: 50 | Batch: 16 | Device: cuda"

python3 scripts/train_geotransformer.py \
    "$FRAG_DIR"/*_ds.ply \
    --epochs 50 \
    --device cuda \
    --output-dir checkpoints_146 \
    --patch-size 512 \
    --batch-size 16 \
    --lr 1e-4 \
    --dropout-rate 0.2

EXIT_CODE=$?

echo ""
echo "=== Training finished at $(date), exit code: $EXIT_CODE ==="

# ── Report ──
if ls checkpoints_146/geotransformer_best.pt 2>/dev/null; then
    python3 -c "
import torch
ckpt = torch.load('checkpoints_146/geotransformer_best.pt', map_location='cpu', weights_only=True)
print(f'Best epoch: {ckpt[\"epoch\"]}')
print(f'Best val_loss: {ckpt[\"val_loss\"]:.6f}')
print(f'Fragments: {len(ckpt.get(\"centroids\",[]))}')
"
    echo "Checkpoints saved to $(pwd)/checkpoints_146/"
else
    echo "WARNING: No checkpoint produced — training may have failed."
    if [ -f slurm_${SLURM_JOB_ID}.err ]; then
        echo "See: $(pwd)/slurm_${SLURM_JOB_ID}.err"
    fi
fi

exit $EXIT_CODE
