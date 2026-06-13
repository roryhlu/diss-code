#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
#  RePAIR Hamilton Setup — transfer files + install deps
# ──────────────────────────────────────────────────────────────────────
#
#  Run ONCE from your laptop to:
#    1. Create the work directory on Hamilton
#    2. SCP the 146 fragment PLYs + 3 Python files
#    3. Set up correct directory structure for imports
#
#  Usage:
#    chmod +x hamilton/setup.sh
#    ./hamilton/setup.sh
#
#  Assumes:
#    - SSH key access to Hamilton (no password prompt)
#    - You're in the DISS_CODE directory when running this
# ──────────────────────────────────────────────────────────────────────

set -e

HAMILTON="fwvp47@hamilton.dur.ac.uk"
WORK_DIR="/nobackup/fwvp47/repair_training"

echo "=== RePAIR Hamilton Setup ==="
echo "Target: $HAMILTON:$WORK_DIR"
echo ""

# ── 1. Create directories on Hamilton ──
echo "1. Creating directories on Hamilton ..."
ssh "$HAMILTON" "mkdir -p $WORK_DIR/fragments $WORK_DIR/scripts $WORK_DIR/uncertainty $WORK_DIR/checkpoints_146"

# ── 2. Transfer Python files ──
echo "2. Transferring Python project files ..."
scp scripts/train_geotransformer.py "$HAMILTON:$WORK_DIR/scripts/"
scp uncertainty/geotransformer.py "$HAMILTON:$WORK_DIR/uncertainty/"
scp uncertainty/__init__.py "$HAMILTON:$WORK_DIR/uncertainty/"

# ── 3. Transfer fragment PLYs (146 files, ~1.3 GB) ──
echo "3. Transferring 146 fragment PLY files (~1.3 GB) ..."
echo "   This will take 2-5 minutes depending on network speed."
FRAG_SRC="repair_fragments_ds"
if [ ! -d "$FRAG_SRC" ]; then
    echo "ERROR: $FRAG_SRC/ not found. Run from the DISS_CODE directory."
    exit 1
fi
FRAG_COUNT=$(ls "$FRAG_SRC"/*_ds.ply 2>/dev/null | wc -l)
echo "   Found $FRAG_COUNT fragment files."
scp "$FRAG_SRC"/*_ds.ply "$HAMILTON:$WORK_DIR/fragments/"

# ── 4. Transfer Slurm script ──
echo "4. Transferring Slurm job script ..."
scp hamilton/train_146.sbatch "$HAMILTON:$WORK_DIR/"

# ── 5. Install Python deps on Hamilton ──
echo "5. Installing Python dependencies on Hamilton ..."
ssh "$HAMILTON" "module load python/3.11 && pip install --user --quiet torch numpy scipy 2>&1 | tail -1"

# ── 6. Verify transfer ──
echo ""
echo "6. Verifying transfer ..."
ssh "$HAMILTON" "
    echo '  Fragments:' \$(ls $WORK_DIR/fragments/*_ds.ply 2>/dev/null | wc -l) 'files'
    echo '  Scripts:  ' train_geotransformer.py=\$(test -f $WORK_DIR/scripts/train_geotransformer.py && echo YES || echo MISSING)
    echo '  Model:    ' geotransformer.py=\$(test -f $WORK_DIR/uncertainty/geotransformer.py && echo YES || echo MISSING)
    echo '  Init:     ' __init__.py=\$(test -f $WORK_DIR/uncertainty/__init__.py && echo YES || echo MISSING)
    echo '  Slurm:    ' train_146.sbatch=\$(test -f $WORK_DIR/train_146.sbatch && echo YES || echo MISSING)
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To submit the training job:"
echo "  ssh $HAMILTON"
echo "  cd $WORK_DIR"
echo "  sbatch train_146.sbatch"
echo ""
echo "To check job status:"
echo "  ssh $HAMILTON squeue -u fwvp47"
echo ""
echo "After training finishes, copy checkpoints back:"
echo "  scp -r $HAMILTON:$WORK_DIR/checkpoints_146 ."
