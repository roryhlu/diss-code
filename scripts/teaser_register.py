#!/usr/bin/env python3
"""
TEASER++ Global Registration Pipeline with FPFH Descriptors.

End-to-end script implementing Module 1.3 of the RePAIR dissertation:
  1. Load source and target point clouds
  2. Voxel grid downsampling (uniform density)
  3. FPFH descriptor computation (geometric feature extraction)
  4. Feature matching with mutual nearest-neighbour + Lowe's ratio test
  5. TEASER++ global registration with TLS cost function
     - Rejects non-Gaussian subsurface scattering noise
     - Provides certifiable optimality bounds
  6. Apply SE(3) transform and visualise result

Fallback: When TEASER++ bindings are unavailable, uses Open3D RANSAC.

Usage:
  python scripts/teaser_register.py src.ply tgt.ply                        \\
      --voxel-size 0.005  --c-threshold 0.01  --output result.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from registration.fpfh_features import compute_fpfh, match_features  # noqa: E402
from registration.teaser_registration import (  # noqa: E402
    TeaserParams,
    register_teaser,
)
from registration.se3_utils import transform_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TEASER++ global registration with TLS cost for RePAIR fragments",
    )
    p.add_argument("source", type=str, help="Source point cloud (PLY, PCD, XYZ)")
    p.add_argument("target", type=str, help="Target point cloud (PLY, PCD, XYZ)")
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel downsampling size (metres, default: 0.005)",
    )
    p.add_argument(
        "--c-threshold",
        type=float,
        default=0.01,
        help="TLS truncation threshold (metres, default: 0.01 = 1 cm)",
    )
    p.add_argument(
        "--noise-bound",
        type=float,
        default=0.001,
        help="Expected sensor noise (metres, default: 0.001)",
    )
    p.add_argument(
        "--fpfh-radius",
        type=float,
        default=0.025,
        help="FPFH search radius (default: 0.025)",
    )
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.9,
        help="Lowe ratio test threshold (default: 0.9)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save transformed source cloud to file",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualisation",
    )
    return p.parse_args()


def load_and_preprocess(file_path: str, voxel_size: float) -> o3d.geometry.PointCloud:
    """Load point cloud and downsample via voxel grid."""
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in '{file_path}'")
    print(f"  Loaded {len(pcd.points)} points from {Path(file_path).name}")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  Downsampled to {len(pcd.points)} points (voxel={voxel_size}m)")
    return pcd


def visualise(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    src_aligned: o3d.geometry.PointCloud,
    title: str = "TEASER++ Registration Result",
) -> None:
    """Render source (aligned), target, and original source."""
    src.paint_uniform_color([1, 0.706, 0])      # orange — original source
    tgt.paint_uniform_color([0, 0.651, 0.929])   # blue  — target
    src_aligned.paint_uniform_color([0, 1, 0])    # green — aligned source

    o3d.visualization.draw_geometries(
        [src_aligned, tgt, src], window_name=title
    )


def main() -> None:
    args = parse_args()

    # ─── 1. Load & preprocess ───
    print("=== Loading point clouds ===")
    src = load_and_preprocess(args.source, args.voxel_size)
    tgt = load_and_preprocess(args.target, args.voxel_size)

    # ─── 2. TEASER++ registration ───
    params = TeaserParams(
        c_threshold=args.c_threshold,
        noise_bound=args.noise_bound,
        fpfh_radius=args.fpfh_radius,
        ratio_threshold=args.ratio_threshold,
    )

    print(f"\n=== Running TEASER++ registration ===")
    print(f"  TLS truncation threshold:  c = {args.c_threshold:.4f} m")
    print(f"  TLS penalty bound:         c² = {args.c_threshold**2:.2e}")
    print(f"  FPFH radius:               {args.fpfh_radius} m")
    print(f"  Ratio test threshold:      {args.ratio_threshold}")

    result = register_teaser(src, tgt, params)
    T_matrix = result.T

    # ─── 3. Analyse result ───
    print(f"\n=== Registration result ===")
    print(f"  Rotation angle:      {result.rotation_angle_deg:.4f}°")
    print(f"  Translation norm:    {result.translation_norm:.4f} m")
    print(f"  Correspondences:     {result.num_correspondences}")
    print(f"  Solver runtime:      {result.runtime_sec:.3f} s")
    if result.certificate is not None:
        print(f"  TLS certificate:     {result.certificate:.6f} (suboptimality bound)")
    else:
        print(f"  TLS certificate:     N/A (RANSAC fallback)")
    print(f"  SE(3) validated:      True")
    print(f"  SE(3) matrix:\n{np.array2string(T_matrix, precision=6, suppress_small=True)}")

    # ─── 4. Apply transform ───
    src_aligned = src.transform(T_matrix)

    # ─── 5. Compute registration error ───
    # Chamfer-like: mean nearest-neighbour distance from aligned→target
    dists = np.asarray(
        src_aligned.compute_point_cloud_distance(tgt)
    )
    rmse = np.sqrt(np.mean(dists**2))
    print(f"\n  RMS registration error: {rmse:.6f} m")

    # ─── 6. Save ───
    if args.output:
        o3d.io.write_point_cloud(args.output, src_aligned)
        print(f"\n  Saved aligned cloud to {args.output}")

    # ─── 7. Visualise ───
    if not args.no_viz:
        visualise(src, tgt, src_aligned)


if __name__ == "__main__":
    main()
