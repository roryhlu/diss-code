#!/usr/bin/env python3
"""
Monte Carlo Dropout Pose Covariance Estimator.

Wraps the full 6-DoF pose inference pipeline in a stochastic Monte Carlo
loop to extract the epistemic covariance matrix Σ ∈ R^{6×6} over SE(3)
pose estimates.  Each MC forward pass samples a different dropout mask,
producing a different set of learned descriptors → different correspondences
→ a different SE(3) registration output.

Pipeline per stochastic pass t = 1 … T:
  1. GeoTransformer extracts per-point features with MC Dropout ON.
  2. Features are matched between source and target clouds.
  3. TEASER++ (or RANSAC fallback) registers source onto target.
  4. The SE(3) pose T_t is collected.

After T passes:
  - Compute 6x6 covariance Σ in the se(3) Lie algebra via twist parameterisation.
  - Project Σ to per-point 3D spatial variance σ²_p_k.
  - Colour-code the source cloud by σ²_p (blue = certain, red = uncertain).
  - Output Σ, T_mean, and the coloured point cloud.

=== Usage ===

    python scripts/mc_pose_covariance.py src.ply tgt.ply \
        --model checkpoints/geotransformer_best.pt \
        --num-passes 50 --dropout-rate 0.2 \
        --voxel-size 0.005 --output covariance_result.pcd
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from registration.fpfh_features import compute_fpfh, match_features  # noqa: E402
from registration.teaser_registration import (          # noqa: E402
    TeaserParams,
    register_teaser,
    SE3Result,
)
from registration.se3_utils import transform_points       # noqa: E402
from uncertainty.geotransformer import GeoTransformer     # noqa: E402
from uncertainty.pose_covariance import (                 # noqa: E402
    compute_pose_covariance,
    print_covariance_report,
    project_spatial_variance,
    variance_to_rgb,
)

# Optional TEASER++ binding
try:
    import teaserpp_python  # noqa: F401
    _HAS_TEASER = True
except ImportError:
    _HAS_TEASER = False


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------

def load_and_preprocess(
    file_path: str,
    voxel_size: float,
    estimate_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """Load point cloud and apply voxel downsampling."""
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points in '{file_path}'")
    print(f"  Loaded {len(pcd.points):,} points from {Path(file_path).name}")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  Downsampled to {len(pcd.points):,} points (voxel={voxel_size}m)")
    if estimate_normals and not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    return pcd


# ---------------------------------------------------------------------------
# Feature matching using GeoTransformer descriptors
# ---------------------------------------------------------------------------

def match_geotransformer_features(
    features_src: torch.Tensor,
    features_tgt: torch.Tensor,
    ratio_threshold: float = 0.9,
    mutual_filter: bool = True,
    max_correspondences: int = 5000,
) -> np.ndarray:
    """
    Match per-point descriptors using nearest-neighbour + Lowe ratio test.

    Args:
        features_src: (N_src, D) source descriptors.
        features_tgt: (N_tgt, D) target descriptors.
        ratio_threshold: Lowe ratio test threshold.
        mutual_filter: Keep only mutual nearest neighbours.
        max_correspondences: Upper bound on returned pairs.

    Returns:
        Array of shape (M, 2) with (src_idx, tgt_idx) correspondence pairs.
    """
    src = features_src.detach().cpu().numpy().astype(np.float64)
    tgt = features_tgt.detach().cpu().numpy().astype(np.float64)

    diff_fwd = src[:, None, :] - tgt[None, :, :]
    dists2_fwd = (diff_fwd ** 2).sum(axis=-1)
    k_eff = min(2, tgt.shape[0] - 1)
    idx_fwd = np.argpartition(dists2_fwd, k_eff, axis=-1)[:, :k_eff]
    dists_fwd = np.take_along_axis(dists2_fwd, idx_fwd, axis=-1)
    sort_fwd = np.argsort(dists_fwd, axis=-1)
    idx_fwd = np.take_along_axis(idx_fwd, sort_fwd, axis=-1)
    dists_fwd = np.take_along_axis(dists_fwd, sort_fwd, axis=-1)

    passed_fwd = dists_fwd[:, 0] < ratio_threshold * dists_fwd[:, 1]

    correspondences = []
    for i_src in np.where(passed_fwd)[0]:
        i_tgt = int(idx_fwd[i_src, 0])
        correspondences.append((i_src, i_tgt))

    if mutual_filter:
        diff_bwd = tgt[:, None, :] - src[None, :, :]
        dists2_bwd = (diff_bwd ** 2).sum(axis=-1)
        k_eff_b = min(2, src.shape[0] - 1)
        idx_bwd = np.argpartition(dists2_bwd, k_eff_b, axis=-1)[:, :k_eff_b]
        dists_bwd = np.take_along_axis(dists2_bwd, idx_bwd, axis=-1)
        sort_bwd = np.argsort(dists_bwd, axis=-1)
        idx_bwd = np.take_along_axis(idx_bwd, sort_bwd, axis=-1)
        dists_bwd = np.take_along_axis(dists_bwd, sort_bwd, axis=-1)
        passed_bwd = dists_bwd[:, 0] < ratio_threshold * dists_bwd[:, 1]

        tgt_to_src = {}
        for i_tgt in np.where(passed_bwd)[0]:
            tgt_to_src[i_tgt] = int(idx_bwd[i_tgt, 0])

        correspondences = [
            (s, t) for (s, t) in correspondences
            if tgt_to_src.get(t) == s
        ]

    if not correspondences:
        return np.empty((0, 2), dtype=np.int32)

    corrs = np.array(correspondences, dtype=np.int32)
    if len(corrs) > max_correspondences:
        scores = np.array([dists_fwd[s, 0] for (s, t) in correspondences])
        keep = np.argsort(scores)[:max_correspondences]
        corrs = corrs[keep]

    return corrs


def extract_correspondence_clouds_np(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    correspondences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matched point sets from (M, 2) correspondence array."""
    src_pts = np.asarray(pcd_src.points)[correspondences[:, 0]]
    tgt_pts = np.asarray(pcd_tgt.points)[correspondences[:, 1]]
    return src_pts.astype(np.float64), tgt_pts.astype(np.float64)


# ---------------------------------------------------------------------------
# Single-pass pose inference with GeoTransformer features
# ---------------------------------------------------------------------------

def infer_pose_single_pass(
    model: GeoTransformer,
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    params: dict,
    device: str,
) -> np.ndarray | None:
    """Run ONE stochastic pose inference pass returning 4x4 SE(3) or None."""
    src_pts_np = np.asarray(pcd_src.points, dtype=np.float32)
    tgt_pts_np = np.asarray(pcd_tgt.points, dtype=np.float32)
    src_norm_np = (np.asarray(pcd_src.normals, dtype=np.float32)
                   if pcd_src.has_normals()
                   else np.zeros_like(src_pts_np))
    tgt_norm_np = (np.asarray(pcd_tgt.normals, dtype=np.float32)
                   if pcd_tgt.has_normals()
                   else np.zeros_like(tgt_pts_np))

    src_tensor = torch.from_numpy(np.column_stack([src_pts_np, src_norm_np])).to(device)
    tgt_tensor = torch.from_numpy(np.column_stack([tgt_pts_np, tgt_norm_np])).to(device)

    with torch.no_grad():
        features_src = model.forward_features(src_tensor)
        features_tgt = model.forward_features(tgt_tensor)

    corrs = match_geotransformer_features(
        features_src, features_tgt,
        ratio_threshold=params.get("ratio_threshold", 0.9),
        mutual_filter=True,
        max_correspondences=params.get("max_correspondences", 5000),
    )

    if len(corrs) < 3:
        return None

    src_matched, tgt_matched = extract_correspondence_clouds_np(pcd_src, pcd_tgt, corrs)

    try:
        if _HAS_TEASER:
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = params.get("c_threshold", 0.01) ** 2
            solver_params.noise_bound = params.get("noise_bound", 0.001)
            solver_params.estimate_scaling = False
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12
            solver_params.rotation_estimation_algorithm = (
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            )
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(src_matched.T.copy(), tgt_matched.T.copy())
            solution = solver.getSolution()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(solution.rotation, dtype=np.float64)
            T[:3, 3] = np.asarray(solution.translation, dtype=np.float64).flatten()
        else:
            # RANSAC fallback via Open3D
            pcd_src_tmp = o3d.geometry.PointCloud()
            pcd_src_tmp.points = o3d.utility.Vector3dVector(src_matched.astype(np.float64))
            pcd_tgt_tmp = o3d.geometry.PointCloud()
            pcd_tgt_tmp.points = o3d.utility.Vector3dVector(tgt_matched.astype(np.float64))
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                pcd_src_tmp, pcd_tgt_tmp, corrs,
                params.get("ransac_max_distance", 0.02),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=3,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    max_iteration=100000, confidence=0.999
                ),
            )
            T = np.asarray(result.transformation, dtype=np.float64)

        return T
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FPFH-based fallback
# ---------------------------------------------------------------------------

def infer_pose_fpfh_single_pass(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    params: dict,
) -> np.ndarray | None:
    """Run ONE FPFH-based registration pass (deterministic)."""
    try:
        result: SE3Result = register_teaser(pcd_src, pcd_tgt, None)
        return result.T.copy()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main MC loop
# ---------------------------------------------------------------------------

def run_mc_pose_inference(
    model: GeoTransformer | None,
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    T: int = 50,
    device: str = "cpu",
    params: dict | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], float]:
    """Run T stochastic pose inference passes."""
    if params is None:
        params = {}
    if model is not None:
        model.set_mc_mode(True)
        model.eval()
        model.to(device)

    poses = []
    t_start = time.perf_counter()

    for t in range(1, T + 1):
        if verbose:
            print(f"  Pass {t}/{T}...", end="", flush=True)

        T_mat = (infer_pose_single_pass(model, pcd_src, pcd_tgt, params, device)
                 if model is not None
                 else infer_pose_fpfh_single_pass(pcd_src, pcd_tgt, params))

        if T_mat is not None:
            poses.append(T_mat)

        if verbose:
            status = "✓" if T_mat is not None else "✗ (failed)"
            print(f" {status}")

    runtime = time.perf_counter() - t_start
    if model is not None:
        model.set_mc_mode(False)
    return poses, runtime


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MC Dropout Pose Covariance Estimator (Module 1.4)",
    )
    p.add_argument("source", type=str, help="Source point cloud (PLY, PCD)")
    p.add_argument("target", type=str, help="Target point cloud (PLY, PCD)")
    p.add_argument("--model", type=str, default=None,
                   help="GeoTransformer checkpoint (.pt)")
    p.add_argument("--num-passes", type=int, default=50)
    p.add_argument("--dropout-rate", type=float, default=0.2)
    p.add_argument("--voxel-size", type=float, default=0.005)
    p.add_argument("--c-threshold", type=float, default=0.01)
    p.add_argument("--noise-bound", type=float, default=0.001)
    p.add_argument("--output", type=str, default="covariance_cloud.pcd")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    print("=== Loading point clouds ===")
    src = load_and_preprocess(args.source, args.voxel_size)
    tgt = load_and_preprocess(args.target, args.voxel_size)

    model = None
    if args.model:
        print(f"\n=== Loading GeoTransformer ===")
        model = GeoTransformer(bottleneck_dropout=args.dropout_rate)
        if Path(args.model).exists():
            state_dict = torch.load(args.model, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            print(f"  Loaded checkpoint: {Path(args.model).name}")
        else:
            print(f"  WARNING: checkpoint not found '{args.model}' — using random init")
    else:
        print("\n=== No GeoTransformer provided — using deterministic FPFH ===")
        print("  WARNING: Pose covariance will be zero (no stochasticity).")

    print(f"\n=== Running {args.num_passes} MC Pose Inference Passes ===")
    print(f"  Dropout rate: {args.dropout_rate}, Device: {device}")

    params = {
        "c_threshold": args.c_threshold,
        "noise_bound": args.noise_bound,
        "ratio_threshold": 0.9,
        "max_correspondences": 5000,
    }

    poses, runtime = run_mc_pose_inference(
        model=model, pcd_src=src, pcd_tgt=tgt,
        T=args.num_passes, device=device, params=params, verbose=True,
    )

    n_valid = len(poses)
    print(f"\n  Valid poses: {n_valid}/{args.num_passes} "
          f"({100*n_valid/max(args.num_passes,1):.1f}%)")
    print(f"  Runtime: {runtime:.2f} sec ({runtime/max(args.num_passes,1):.3f} sec/pass)")

    if n_valid < 2:
        print("\n  ERROR: Fewer than 2 valid poses — cannot compute covariance.")
        sys.exit(1)

    print(f"\n=== Computing 6x6 Pose Covariance ===")
    Sigma, T_mean = compute_pose_covariance(poses)

    src_points = np.asarray(src.points, dtype=np.float64)
    spatial_var = project_spatial_variance(Sigma, src_points)

    print_covariance_report(Sigma, T_mean, spatial_var)

    print(f"\n=== Saving Variance-Coloured Cloud ===")
    colours = variance_to_rgb(spatial_var)
    coloured = o3d.geometry.PointCloud(src)
    coloured.colors = o3d.utility.Vector3dVector(colours)
    o3d.io.write_point_cloud(args.output, coloured)
    print(f"  Saved to {args.output}")

    aligned = src.transform(T_mean)
    out = Path(args.output)
    aligned_path = out.parent / f"{out.stem}_aligned.pcd"
    o3d.io.write_point_cloud(str(aligned_path), aligned)
    print(f"  Saved aligned mean pose to {aligned_path}")

    sigma_path = out.parent / f"{out.stem}_sigma.npy"
    np.save(str(sigma_path), Sigma)
    print(f"  Saved covariance Sigma to {sigma_path}")

    if not args.no_viz:
        print(f"\n=== Visualising ===")
        print("  Blue = certain (low spatial variance)")
        print("  Red  = uncertain (high spatial variance)")
        original = o3d.geometry.PointCloud(src)
        original.paint_uniform_color([0.7, 0.7, 0.7])
        o3d.visualization.draw_geometries(
            [coloured, original],
            window_name="MC Dropout Pose Covariance - Spatial Variance",
        )


if __name__ == "__main__":
    main()
