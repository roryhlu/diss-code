# Hamilton Supercomputer — RePAIR GeoTransformer Training

Trains the GeoTransformer on all 146 Pompeii archaeological fragments from the RePAIR dataset using Durham's Hamilton HPC cluster GPU nodes.

## Quick start

```bash
# 1. Transfer everything (run once from your laptop)
cd "/mnt/c/MISCADA/MISCADA DISS/DISS_CODE"
chmod +x hamilton/setup.sh
./hamilton/setup.sh

# 2. SSH in and submit
ssh fwvp47@hamilton.dur.ac.uk
cd /nobackup/fwvp47/repair_training
sbatch train_146.sbatch

# 3. Check status
squeue -u fwvp47
tail -f slurm_<JOBID>.out
```

## What gets transferred

| File | Size | Purpose |
|------|------|---------|
| `fragments/*_ds.ply` (146 files) | ~1.3 GB | Preprocessed fragment point clouds with normals |
| `scripts/train_geotransformer.py` | 20 KB | Training loop with fast binary PLY loader |
| `uncertainty/geotransformer.py` | 20 KB | GeoTransformer neural network architecture |
| `uncertainty/__init__.py` | 1 KB | Package init |
| `train_146.sbatch` | 3 KB | Slurm job script |

## Training configuration

| Parameter | Value |
|-----------|-------|
| Fragments | 146 (19M total points) |
| Model | GeoTransformer (830K params) |
| Epochs | 50 |
| Batch size | 16 (GPU), 8 (CPU) |
| Learning rate | 1e-4 (AdamW + CosineAnnealing) |
| Dropout rate | 0.20 |
| Device | CUDA GPU (auto-detected) |

## Estimated time

| GPU | Per epoch | 50 epochs |
|-----|-----------|-----------|
| A100 (80GB) | ~5 min | **~4 hours** |
| V100 (32GB) | ~8 min | **~7 hours** |
| CPU only | ~2.1 hours | ~4.4 days |

## If the GPU partition rejects the job

Hamilton partition names change. Try:

```bash
# Check available partitions
sinfo -p gpu
sinfo -p gpu_devel
sinfo -p tesla

# Submit with explicit GPU type
sbatch --partition=gpu --gres=gpu:a100:1 train_146.sbatch
sbatch --partition=gpu --gres=gpu:v100:1 train_146.sbatch

# Short test job (15 min max on devel partition)
sbatch --partition=gpu_devel --gres=gpu:1 --time=00:15:00 train_146.sbatch
```

Edit `train_146.sbatch` line 3 (`#SBATCH --partition=gpu`) to match the partition you find.

## After training

```bash
# From your laptop
scp -r fwvp47@hamilton.dur.ac.uk:/nobackup/fwvp47/repair_training/checkpoints_146 .

# Verify
python -c "
import torch
ckpt = torch.load('checkpoints_146/geotransformer_best.pt', map_location='cpu', weights_only=True)
print(f'Epoch: {ckpt[\"epoch\"]}, Val Loss: {ckpt[\"val_loss\"]:.6f}')
print(f'Fragments: {len(ckpt.get(\"centroids\", []))}')
"

# Generate MC Dropout variance cloud with the new model
python scripts/mc_dropout_variance.py repair_fragments_ds/RPf_00577_ds.ply \
    --model checkpoints_146/geotransformer_best.pt \
    --num-passes 50 --output variance_146.pcd
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `sbatch: error: Batch job submission failed: Invalid partition` | Partition name wrong — use `sinfo` to list available partitions |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install --user torch` on Hamilton |
| `CUDA out of memory` | Reduce `--batch-size` to 8 or 4 in the sbatch script |
| `No fragment PLYs found` | SCP transfer didn't complete — re-run `setup.sh` |
| Training slow | Check `nvidia-smi` on the compute node — GPU might not be allocated |
| Checkpoints not saved | Look at `slurm_<JOBID>.err` for Python traceback |

## File manifest on Hamilton

```
/nobackup/fwvp47/repair_training/
├── train_146.sbatch          # Slurm job script
├── scripts/
│   └── train_geotransformer.py
├── uncertainty/
│   ├── __init__.py
│   └── geotransformer.py
├── fragments/
│   ├── RPf_00047_ds.ply
│   ├── RPf_00522_ds.ply
│   ├── ... (146 files total)
│   └── RPf_01004_ds.ply
├── checkpoints_146/          # Created during training
│   ├── geotransformer_best.pt
│   └── geotransformer_latest.pt
├── slurm_<JOBID>.out         # Training log
└── slurm_<JOBID>.err         # Error log
```
