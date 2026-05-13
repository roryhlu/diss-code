#!/usr/bin/env python3
"""
Standalone FPFH Descriptor Computation and PCA Visualisation.

Computes Fast Point Feature Histograms (33-D per point) for Open3D point clouds
and visualises the descriptor space by projecting from ℝ³³ → RGB via PCA — giving
an intuitive colour map of geometric similarity across the surface.

Usage:
  python scripts/compute_fpfh.py fragment.ply \\
      --voxel-size 0.005  --fpfh-radius 0.025  --output coloured.ply

Rationale for PCA-RGB mapping:
  Points with similar local geometry (flat faces, edges, corners) produce similar
  FPFH vectors and thus similar colours. Uniform regions appear as solid colour
  patches; distinctive features appear as colour transitions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from registration.fpfh_features import compute_fpfh  # noqa: E402


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from PLY, PCD, XYZ, OBJ, or STL."""
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in '{file_path}'")
    print(f"  Loaded {len(pcd.points):,} points from {Path(file_path).name}")
    return pcd


def save_coloured_cloud(
    pcd: o3d.geometry.PointCloud,
    colours: np.ndarray,
    file_path: str,
) -> None:
    """Save a point cloud with per-point RGB colours."""
    coloured = o3d.geometry.PointCloud(pcd)
    coloured.colors = o3d.utility.Vector3dVector(colours)
    o3d.io.write_point_cloud(file_path, coloured)
    print(f"  Saved FPFH-coloured cloud to {file_path}")


# ---------------------------------------------------------------------------
# FPFH → RGB via PCA
# ---------------------------------------------------------------------------


def fpfh_to_rgb(fpfh: o3d.pipelines.registration.Feature) -> np.ndarray:
    r"""
    Map 33-D FPFH descriptors to RGB (0–1) via PCA projection.

    The descriptor matrix F ∈ ℝ^{N × 33} is mean-centred and projected onto
    its top three principal components:

        C = (1/N) F̃ᵀF̃               (33 × 33 covariance)
        C = V diag(λ) Vᵀ              eigenvalue decomposition
        C_rgb = F̃ · V₃                (N × 3) projection
        colours = (C_rgb - min) / (max - min)   normalise to [0, 1]

    Returns:
        colours: (N, 3) numpy array in [0, 1] RGB format.
    """
    F = np.asarray(fpfh.data, dtype=np.float64)  # (N, 33)

    if F.shape[0] < 3:
        raise ValueError(f"Only {F.shape[0]} points — need ≥ 3 for PCA.")

    F_centred = F - F.mean(axis=0, keepdims=True)
    C = (F_centred.T @ F_centred) / (F.shape[0] - 1)  # unbiased covariance

    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh returns ascending order; take top 3
    top3 = eigvecs[:, -3:]  # (33, 3)

    rgb = F_centred @ top3  # (N, 3)

    # Normalise to [0, 1]
    rgb_min = rgb.min(axis=0)
    rgb_max = rgb.max(axis=0)
    span = rgb_max - rgb_min
    span[span == 0] = 1.0  # guard against zero variance
    rgb = (rgb - rgb_min) / span

    return np.asarray(rgb, dtype=np.float64)


def print_descriptor_stats(fpfh: o3d.pipelines.registration.Feature) -> None:
    """Print summary statistics of the FPFH descriptor matrix."""
    F = np.asarray(fpfh.data, dtype=np.float64)
    print(f"\n  === FPFH Descriptor Statistics ===")
    print(f"  Shape:       {F.shape[0]:,} points × {F.shape[1]} bins")
    print(f"  Mean:        {F.mean():.6f}")
    print(f"  Std:         {F.std():.6f}")
    print(f"  Min:         {F.min():.6f}")
    print(f"  Max:         {F.max():.6f}")
    print(f"  Sparsity:    {(F < 1e-6).mean() * 100:.1f}% near-zero bins")
    # Entropy per descriptor (how uniformly distributed the histogram is)
    F_norm = F / (F.sum(axis=1, keepdims=True) + 1e-12)
    F_safe = np.clip(F_norm, 1e-12, 1.0)
    entropy = -np.sum(F_safe * np.log(F_safe), axis=1) / np.log(33)
    print(f"  Mean entropy (normalised): {entropy.mean():.4f}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualise(
    original: o3d.geometry.PointCloud,
    fpfh_coloured: o3d.geometry.PointCloud,
    title: str = "FPFH Descriptors — PCA-RGB Visualisation",
) -> None:
    """Render original and FPFH-coloured point clouds side by side."""
    # Shift the FPFH-coloured cloud to the right for side-by-side view
    bbox = original.get_axis_aligned_bounding_box()
    span = bbox.get_max_bound() - bbox.get_min_bound()
    offset_x = span[0] * 1.2

    shifted = o3d.geometry.PointCloud(fpfh_coloured)
    shifted.translate([offset_x, 0, 0])

    original.paint_uniform_color([0.7, 0.7, 0.7])  # grey = original

    o3d.visualization.draw_geometries(
        [original, shifted],
        window_name=title,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and visualise FPFH descriptors for RePAIR fragments",
    )
    p.add_argument(
        "input",
        type=str,
        help="Input point cloud (PLY, PCD, XYZ, OBJ, STL)",
    )
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel downsampling size (metres, default: 0.005)",
    )
    p.add_argument(
        "--normal-radius",
        type=float,
        default=0.01,
        help="Normal estimation search radius (metres, default: 0.01)",
    )
    p.add_argument(
        "--normal-k",
        type=int,
        default=30,
        help="Max KNN for normal PCA (default: 30)",
    )
    p.add_argument(
        "--fpfh-radius",
        type=float,
        default=0.025,
        help="FPFH search radius (metres, default: 0.025)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save FPFH-coloured cloud to file (PLY/PCD)",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print FPFH descriptor statistics",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualisation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load ──
    print("=== Loading point cloud ===")
    pcd = load_point_cloud(args.input)

    # ── 2. Voxel downsample ──
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"  Downsampled to {len(pcd.points):,} points (voxel={args.voxel_size}m)")

    # ── 3. Compute FPFH ──
    print(f"\n=== Computing FPFH descriptors ===")
    print(f"  Normal radius:  {args.normal_radius} m,  k={args.normal_k}")
    print(f"  FPFH radius:    {args.fpfh_radius} m")

    fpfh = compute_fpfh(
        pcd,
        normal_radius=args.normal_radius,
        normal_k=args.normal_k,
        fpfh_radius=args.fpfh_radius,
    )
    print(f"  Computed {np.asarray(fpfh.data).shape[0]:,} × "
          f"{np.asarray(fpfh.data).shape[1]} descriptors")

    # ── 4. Statistics ──
    if args.stats:
        print_descriptor_stats(fpfh)

    # ── 5. PCA → RGB ──
    print(f"\n=== Mapping FPFH → RGB via PCA ===")
    colours = fpfh_to_rgb(fpfh)
    explained_var = None
    if colours.size:
        F = np.asarray(fpfh.data, dtype=np.float64)
        F_centred = F - F.mean(axis=0)
        C = (F_centred.T @ F_centred) / (F.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(C)
        top3_var = eigvals[-3:].sum() / eigvals.sum()
        explained_var = top3_var * 100
    if explained_var is not None:
        print(f"  Top 3 PCs explain {explained_var:.1f}% of FPFH variance")

    fpfh_coloured = o3d.geometry.PointCloud(pcd)
    fpfh_coloured.colors = o3d.utility.Vector3dVector(colours)

    # ── 6. Save ──
    if args.output:
        save_coloured_cloud(pcd, colours, args.output)

    # ── 7. Visualise ──
    if not args.no_viz:
        visualise(pcd, fpfh_coloured)


if __name__ == "__main__":
    main()
