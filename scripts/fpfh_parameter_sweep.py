#!/usr/bin/env python3
"""
FPFH Parameter Sweep for Optimal TEASER++ Registration.

Grid-searches the FPFH parameter space (normal_radius, fpfh_radius,
ratio_threshold) to find the combination that maximises registration
quality for RePAIR archaeological fragments.

For each parameter combination:
  1. Voxel-downsample source and target clouds.
  2. Compute FPFH descriptors with the given radii.
  3. Match features via mutual-NN + Lowe ratio test.
  4. Run TEASER++ GNC-TLS registration (or RANSAC fallback).
  5. Record: correspondences, inliers, TLS certificate, runtime,
     rotation/translation error (if ground truth provided),
     Chamfer-like RMS error.

Results are ranked by a composite score:
  score = w1 * (1 / certificate) + w2 * inlier_ratio + w3 * (1 / runtime)

=== Usage ===

    # Full grid search with ground truth
    python scripts/fpfh_parameter_sweep.py src.ply tgt.ply
        --ground-truth pose_gt.npy
        --normal-radius 0.005 0.010 0.015 0.020
        --fpfh-radius 0.015 0.025 0.035 0.050
        --ratio-threshold 0.70 0.85 0.90 0.95
        --output sweep_results.csv --top-10

    # Quick search over fewer combinations
    python scripts/fpfh_parameter_sweep.py src.ply tgt.ply --quick
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from registration.fpfh_features import compute_fpfh, match_features       # noqa: E402
from registration.teaser_registration import register_teaser, TeaserParams # noqa: E402


@dataclass
class SweepResult:
    """Single parameter combination result."""

    normal_radius: float
    fpfh_radius: float
    ratio_threshold: float
    num_correspondences: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    tls_certificate: float | None = None
    runtime_sec: float = 0.0
    rotation_error_deg: float | None = None
    translation_error_m: float | None = None
    rms_error_m: float = 0.0
    converged: bool = True

    @property
    def score(self) -> float:
        """Composite quality score (higher = better)."""
        cert_score = 0.0
        if self.tls_certificate is not None and self.tls_certificate > 0:
            cert_score = 1.0 / (self.tls_certificate + 1e-12)
        inlier_score = self.inlier_ratio * 10.0
        time_score = 1.0 / (self.runtime_sec + 0.1)
        rot_penalty = 0.0
        if self.rotation_error_deg is not None:
            rot_penalty = max(0.0, 5.0 - self.rotation_error_deg)
        return cert_score * 0.3 + inlier_score * 0.3 + time_score * 0.1 + rot_penalty * 0.3

    def to_dict(self) -> dict:
        return {
            "normal_radius": self.normal_radius,
            "fpfh_radius": self.fpfh_radius,
            "ratio_threshold": self.ratio_threshold,
            "num_correspondences": self.num_correspondences,
            "num_inliers": self.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "tls_certificate": self.tls_certificate,
            "runtime_sec": round(self.runtime_sec, 3),
            "rotation_error_deg": (round(self.rotation_error_deg, 3)
                                   if self.rotation_error_deg is not None else None),
            "translation_error_m": (round(self.translation_error_m, 6)
                                     if self.translation_error_m is not None else None),
            "rms_error_m": round(self.rms_error_m, 6),
            "converged": self.converged,
            "score": round(self.score, 3),
        }


def load_point_cloud(path: str, voxel_size: float) -> o3d.geometry.PointCloud:
    """Load and downsample a point cloud."""
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"No points in '{path}'")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def load_ground_truth(path: str) -> np.ndarray:
    """Load ground truth 4x4 SE(3) pose from .npy file."""
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"Ground truth must be 4x4 matrix, got {T.shape}")
    return T


def rotation_error_degrees(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Angular distance between two SO(3) rotations (degrees)."""
    R_diff = R_pred.T @ R_gt
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    theta = np.arccos((trace - 1.0) / 2.0)
    return float(np.rad2deg(theta))


def compute_inlier_count(
    src_pts: np.ndarray, tgt_pts: np.ndarray, T_pred: np.ndarray,
    threshold: float,
) -> int:
    """Count inlier correspondences within distance threshold."""
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    src_t = (T_pred @ src_h.T).T[:, :3]
    dists = np.linalg.norm(src_t - tgt_pts, axis=1)
    return int(np.sum(dists <= threshold))


def run_single_sweep(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    normal_radius: float,
    fpfh_radius: float,
    ratio_threshold: float,
    c_threshold: float,
    T_gt: np.ndarray | None,
) -> SweepResult:
    """Run one parameter combination."""
    result = SweepResult(
        normal_radius=normal_radius,
        fpfh_radius=fpfh_radius,
        ratio_threshold=ratio_threshold,
    )

    params = TeaserParams(
        c_threshold=c_threshold,
        normal_radius=normal_radius,
        fpfh_radius=fpfh_radius,
        ratio_threshold=ratio_threshold,
    )

    t0 = time.perf_counter()
    try:
        reg_result = register_teaser(src, tgt, params)
    except Exception:
        result.runtime_sec = time.perf_counter() - t0
        result.converged = False
        return result

    result.runtime_sec = time.perf_counter() - t0
    result.tls_certificate = reg_result.certificate
    result.converged = reg_result.converged

    # Correspondence count from the result
    result.num_correspondences = reg_result.num_correspondences

    # Compute inlier count post-hoc
    fpfh_src = compute_fpfh(src, normal_radius=normal_radius, fpfh_radius=fpfh_radius)
    fpfh_tgt = compute_fpfh(tgt, normal_radius=normal_radius, fpfh_radius=fpfh_radius)
    corrset = match_features(fpfh_src, fpfh_tgt, mutual_filter=True,
                             ratio_threshold=ratio_threshold)
    if len(np.asarray(corrset)) > 0:
        corrs_np = np.asarray(corrset)
        src_pts = np.asarray(src.points)[corrs_np[:, 0]]
        tgt_pts = np.asarray(tgt.points)[corrs_np[:, 1]]
        result.num_inliers = compute_inlier_count(
            src_pts, tgt_pts, reg_result.T, c_threshold,
        )
        if result.num_correspondences > 0:
            result.inlier_ratio = result.num_inliers / result.num_correspondences

    # RMS Chamfer-like error
    src_aligned = src.transform(reg_result.T)
    dists = np.asarray(src_aligned.compute_point_cloud_distance(tgt))
    result.rms_error_m = float(np.sqrt(np.mean(dists ** 2)))

    # Ground truth errors
    if T_gt is not None:
        R_pred = reg_result.T[:3, :3]
        t_pred = reg_result.T[:3, 3]
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        result.rotation_error_deg = rotation_error_degrees(R_pred, R_gt)
        result.translation_error_m = float(np.linalg.norm(t_pred - t_gt))

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FPFH Parameter Sweep for Optimal TEASER++ Registration",
    )
    p.add_argument("source", type=str, help="Source point cloud (PLY/PCD)")
    p.add_argument("target", type=str, help="Target point cloud (PLY/PCD)")
    p.add_argument("--ground-truth", type=str, default=None,
                   help="Ground truth 4x4 SE(3) pose (.npy)")
    p.add_argument("--voxel-size", type=float, default=0.005)
    p.add_argument("--c-threshold", type=float, default=0.01,
                   help="TLS truncation threshold (m)")
    p.add_argument("--normal-radius", type=float, nargs="+",
                   default=[0.005, 0.010, 0.015, 0.020])
    p.add_argument("--fpfh-radius", type=float, nargs="+",
                   default=[0.015, 0.025, 0.035, 0.050])
    p.add_argument("--ratio-threshold", type=float, nargs="+",
                   default=[0.70, 0.80, 0.90, 0.95])
    p.add_argument("--quick", action="store_true",
                   help="Run quick sweep over fewer combinations")
    p.add_argument("--output", type=str, default="sweep_results.csv")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of top results to display")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.normal_radius = [0.010, 0.015]
        args.fpfh_radius = [0.025, 0.035]
        args.ratio_threshold = [0.85, 0.90]

    # Load clouds
    print("=== Loading point clouds ===")
    src = load_point_cloud(args.source, args.voxel_size)
    tgt = load_point_cloud(args.target, args.voxel_size)
    print(f"  Source: {len(src.points):,} pts, Target: {len(tgt.points):,} pts")

    # Load ground truth
    T_gt = load_ground_truth(args.ground_truth) if args.ground_truth else None
    if T_gt is not None:
        print(f"  Ground truth pose loaded")

    # Build parameter grid
    param_grid = list(itertools.product(
        args.normal_radius, args.fpfh_radius, args.ratio_threshold,
    ))
    n_combos = len(param_grid)
    print(f"\n=== Sweeping {n_combos} parameter combinations ===")
    print(f"  normal_radius: {args.normal_radius}")
    print(f"  fpfh_radius:   {args.fpfh_radius}")
    print(f"  ratio_threshold: {args.ratio_threshold}")

    # Run sweep
    results: list[SweepResult] = []
    t_start = time.perf_counter()

    for i, (nr, fr, rt) in enumerate(param_grid):
        print(f"  [{i+1}/{n_combos}] nr={nr:.3f} fr={fr:.3f} rt={rt:.2f} ...",
              end="", flush=True)
        r = run_single_sweep(src, tgt, nr, fr, rt, args.c_threshold, T_gt)
        results.append(r)
        status = "✓" if r.converged else "✗"
        inl = f"inl={r.inlier_ratio:.2f}" if r.num_correspondences > 0 else "inl=N/A"
        print(f" {status} corrs={r.num_correspondences} {inl} "
              f"t={r.runtime_sec:.1f}s score={r.score:.2f}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Sweep complete in {total_time:.1f}s "
          f"({total_time / n_combos:.1f}s per combination)")

    # Sort by score (descending)
    results.sort(key=lambda r: r.score, reverse=True)

    # Display top N
    print(f"\n{'='*85}")
    print(f"  Top {min(args.top_n, len(results))} FPFH Parameter Combinations")
    print(f"{'='*85}")
    header = (f"{'Rank':<5} {'nr':>6} {'fr':>6} {'rt':>6} {'corrs':>7} "
              f"{'inl_r':>7} {'cert':>10} {'rot_err':>8} {'tran_err':>9} "
              f"{'rms':>8} {'time':>6} {'score':>7}")
    print(header)
    print("-" * 85)

    for i, r in enumerate(results[:args.top_n]):
        rot_str = f"{r.rotation_error_deg:>7.1f}°" if r.rotation_error_deg is not None else "     N/A"
        tr_str = f"{r.translation_error_m*1000:>7.2f}mm" if r.translation_error_m is not None else "     N/A"
        cert_str = f"{r.tls_certificate:.6f}" if r.tls_certificate is not None else "      N/A"
        print(f"{i+1:<5} {r.normal_radius:>5.3f}m {r.fpfh_radius:>5.3f}m "
              f"{r.ratio_threshold:>5.2f} {r.num_correspondences:>7} "
              f"{r.inlier_ratio:>6.3f} {cert_str:>10} {rot_str:>8} "
              f"{tr_str:>9} {r.rms_error_m:>7.4f}m {r.runtime_sec:>5.1f}s "
              f"{r.score:>7.2f}")

    print(f"\n  Best parameters: "
          f"normal_radius={results[0].normal_radius:.3f} "
          f"fpfh_radius={results[0].fpfh_radius:.3f} "
          f"ratio_threshold={results[0].ratio_threshold:.2f}")

    # Save CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    print(f"\n  Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
