#!/usr/bin/env python3
"""
Monte Carlo Dropout Variance Cloud Generator.

Module 1.4: Uncertainty Modeling for collision avoidance with occluded geometry.

Executes T=50 stochastic forward passes through a GeoTransformer model with
MC Dropout enabled, producing a 3D 'variance cloud' where per-point intensity
encodes epistemic uncertainty — the model's lack of knowledge about hidden
geometry on archaeological fragments.

Pipeline:
  1. Load GeoTransformer model checkpoint (.pt).
  2. Load input point cloud (PCD/PLY) with normals.
  3. Enable MC Dropout: model.set_mc_mode(True).
  4. Run T stochastic forward passes with Welford's online accumulator.
  5. Compute per-point epistemic variance σ² = (1/(T-1)) · Σ ‖ŷ_t − μ‖².
  6. Save variance cloud as PCD with intensity = variance.
  7. Visualise: blue = certain, red = uncertain.

=== MC Dropout as Bayesian Approximation ===

Dropout at inference time approximates variational inference in a deep
Gaussian process (Gal & Ghahramani, 2016).  Each stochastic forward pass
samples a different sub-network, and the empirical distribution of T outputs
approximates the Bayesian posterior predictive p(y | x, D).

The epistemic variance component is:
    σ²_epistemic = (1/(T-1)) · Σ_{t=1}^{T} ‖ŷ_t − μ‖²

High epistemic variance at a point means the model has never seen similar
geometry during training — the signature of occluded regions, erosive wear,
or hidden cavities on archaeological fragments.

=== CVaR Integration ===

This variance cloud feeds directly into the CVaR filter (Module 3):
    For each grasp candidate, evaluate force-closure on the mean geometry
    and check that the grasp survives the worst 5% of structural variations.
    High-variance regions are flagged as collision risks.

Usage:
    python scripts/mc_dropout_variance.py fragment.pcd \\
        --model checkpoints/geotransformer_best.pt \\
        --num-passes 50 --dropout-rate 0.2 \\
        --batch-size 4096 --output variance_cloud.pcd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uncertainty.geotransformer import GeoTransformer  # noqa: E402
from uncertainty.mc_inference import run_mc_passes      # noqa: E402
from uncertainty.variance_cloud import (                # noqa: E402
    print_variance_statistics,
    save_variance_cloud,
    visualise_variance,
)


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------


def load_point_cloud_with_normals(file_path: str) -> torch.Tensor:
    """
    Load a point cloud and ensure it has normals.

    Returns:
        Tensor of shape (N, 6) — [x, y, z, nx, ny, nz].
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in '{file_path}'")

    print(f"  Loaded {len(pcd.points):,} points from {Path(file_path).name}")

    # Estimate normals if missing
    if not pcd.has_normals():
        print("  Estimating surface normals (k=30, radius=0.01m)...")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    # Concatenate: (N, 6)
    data = np.column_stack([points, normals])
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str | None,
    dropout_rate: float = 0.2,
    in_channels: int = 6,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
) -> GeoTransformer:
    """
    Load GeoTransformer model from checkpoint or create fresh.

    Args:
        checkpoint_path: Path to .pt checkpoint file, or None for random init.
        dropout_rate:    MC Dropout rate at bottleneck.
        in_channels:     Input feature dimension (6 = xyz + normals).
        embed_dim:       Transformer embedding dimension.
        num_heads:       Number of attention heads.
        num_layers:      Number of transformer blocks.

    Returns:
        GeoTransformer model (weights loaded if checkpoint provided).
    """
    model = GeoTransformer(
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        bottleneck_dropout=dropout_rate,
    )

    if checkpoint_path is not None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {Path(checkpoint_path).name}")
    else:
        print("  WARNING: No checkpoint provided — using randomly initialised model.")
        print("  Variance values will reflect architectural uncertainty, not learned epistemic uncertainty.")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MC Dropout Variance Cloud Generator (Module 1.4)",
    )
    p.add_argument(
        "input",
        type=str,
        help="Input point cloud (PCD/PLY) with normals",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="GeoTransformer checkpoint (.pt). If None, uses random init.",
    )
    p.add_argument(
        "--num-passes",
        type=int,
        default=50,
        help="Number of stochastic forward passes (default: 50)",
    )
    p.add_argument(
        "--dropout-rate",
        type=float,
        default=0.2,
        help="MC Dropout rate at bottleneck (default: 0.2)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Points per forward pass to avoid OOM (default: 4096)",
    )
    p.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Transformer embedding dimension (default: 128)",
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer blocks (default: 4)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="variance_cloud.pcd",
        help="Output variance cloud file (default: variance_cloud.pcd)",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualisation",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ──
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  Device: {device}")

    # ── Load model ──
    print("\n=== Loading GeoTransformer ===")
    model = load_model(
        checkpoint_path=args.model,
        dropout_rate=args.dropout_rate,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model.to(device)

    # ── Load point cloud ──
    print(f"\n=== Loading point cloud ===")
    point_cloud = load_point_cloud_with_normals(args.input)
    N = point_cloud.shape[0]
    print(f"  Input shape: ({N}, 6) — [x,y,z, nx,ny,nz]")

    # ── MC Dropout inference ──
    print(f"\n=== Running MC Dropout Inference ===")
    print(f"  Passes: {args.num_passes}, Dropout: {args.dropout_rate}, Batch: {args.batch_size}")

    mean, variance = run_mc_passes(
        model=model,
        point_cloud=point_cloud,
        T=args.num_passes,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
    )

    # ── Statistics ──
    print(f"\n=== Variance Statistics ===")
    stats = print_variance_statistics(variance)

    # ── Save ──
    print(f"\n=== Saving Variance Cloud ===")
    save_variance_cloud(mean, variance, args.output)

    # ── Visualise ──
    if not args.no_viz:
        visualise_variance(mean, variance)


if __name__ == "__main__":
    main()
