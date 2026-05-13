"""
TEASER++ Global Registration with Truncated Least Squares (TLS).

Implements Module 1.3 of the RePAIR perception pipeline — certifiable global
registration that rejects non-Gaussian subsurface scattering noise via the
TLS cost function, explicitly avoiding standard ICP algorithms.

TLS Objective:
  min_{T in SE(3)}  sum_i min( e_i^2,  c^2 )

where e_i = ||q_i - T(p_i)|| and c is the truncation threshold.

Contrast with standard Least Squares (L2):
  min_T  sum_i e_i^2    <-- catastrophic sensitivity to outliers

TEASER++ solves this via:
  1. Decoupled rotation estimation  (GNC-TLS)
  2. Decoupled translation estimation (adaptive voting)
  3. Semidefinite programming for optimality certification

When TEASER++ bindings are unavailable, falls back to Open3D FPFH + RANSAC
as an approximate global registration (without sub-surface noise robustness).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from registration.fpfh_features import (
    compute_fpfh,
    extract_correspondence_clouds,
    match_features,
)

# ---------------------------------------------------------------------------
# TEASER++ Python binding (optional dependency)
# ---------------------------------------------------------------------------

try:
    import teaserpp_python  # noqa: F811

    _HAS_TEASER = True
except ImportError:
    _HAS_TEASER = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TeaserParams:
    """TEASER++ solver parameters for TLS registration."""

    c_threshold: float = 0.01
    """TLS truncation threshold (metres). Residuals beyond this
    value contribute constant c^2, bounding outlier influence."""

    noise_bound: float = 0.001
    """Expected sensor noise bound (metres)."""

    rotation_gnc_factor: float = 1.4
    """Graduated Non-Convexity factor for rotation estimation.
    Controls annealing speed of the convex->non-convex transition."""

    rotation_max_iterations: int = 100
    """Maximum GNC iterations for rotation estimation."""

    rotation_cost_threshold: float = 1e-12
    """Convergence threshold for rotation GNC."""

    estimate_scaling: bool = False
    """Whether TEASER++ should estimate uniform scaling (ST(3)).
    Disabled for our rigid SE(3) use case."""

    # ── FPFH computation ──
    normal_radius: float = 0.01
    normal_k: int = 30
    fpfh_radius: float = 0.025

    # ── Feature matching ──
    ratio_threshold: float = 0.9
    mutual_filter: bool = True
    max_correspondences: int = 5000

    # ── RANSAC fallback ──
    ransac_max_iterations: int = 4_000_000
    ransac_confidence: float = 0.999
    ransac_max_correspondence_distance: float = 0.02


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_teaser(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    params: TeaserParams | None = None,
) -> np.ndarray:
    """
    Globally register src onto tgt using TEASER++ with TLS cost.

    If TEASER++ bindings are not installed, falls back to Open3D RANSAC.

    Args:
        pcd_src: Source point cloud.
        pcd_tgt: Target point cloud.
        params:  TEASER++ configuration. Uses defaults if None.

    Returns:
        4x4 SE(3) transformation matrix T such that tgt ≈ T @ src.
    """
    if params is None:
        params = TeaserParams()

    # 1. FPFH descriptors
    fpfh_src = compute_fpfh(
        pcd_src,
        normal_radius=params.normal_radius,
        normal_k=params.normal_k,
        fpfh_radius=params.fpfh_radius,
    )
    fpfh_tgt = compute_fpfh(
        pcd_tgt,
        normal_radius=params.normal_radius,
        normal_k=params.normal_k,
        fpfh_radius=params.fpfh_radius,
    )

    # 2. Feature matching
    corrset = match_features(
        fpfh_src,
        fpfh_tgt,
        mutual_filter=params.mutual_filter,
        ratio_threshold=params.ratio_threshold,
        max_correspondences=params.max_correspondences,
    )

    src_pts, tgt_pts = extract_correspondence_clouds(pcd_src, pcd_tgt, corrset)

    # 3. Solve registration
    if _HAS_TEASER:
        T = _solve_teaser_core(src_pts, tgt_pts, params)
    else:
        warnings.warn(
            "teaserpp_python not installed — falling back to Open3D RANSAC. "
            "Subsurface scattering noise will NOT be rejected. "
            "Install TEASER++ bindings for certifiable TLS registration.",
            stacklevel=2,
        )
        T = _solve_ransac_fallback(pcd_src, pcd_tgt, fpfh_src, fpfh_tgt, params)

    return T


# ---------------------------------------------------------------------------
# TEASER++ Core Solver
# ---------------------------------------------------------------------------


def _solve_teaser_core(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    params: TeaserParams,
) -> np.ndarray:
    """
    Run TEASER++ GNC-TLS registration.

    TLS cost function:
      cbar2 = c_threshold^2  truncates any residual e_i^2 > cbar2.

    The solver:
      1. Estimates rotation via GNC with TLS clamping
      2. Estimates translation via adaptive voting on a maximum clique
      3. Returns SE(3) transform with certifiable suboptimality bound
    """
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = params.c_threshold ** 2
    solver_params.noise_bound = params.noise_bound
    solver_params.estimate_scaling = params.estimate_scaling
    solver_params.rotation_gnc_factor = params.rotation_gnc_factor
    solver_params.rotation_max_iterations = params.rotation_max_iterations
    solver_params.rotation_cost_threshold = params.rotation_cost_threshold
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src_pts.T, tgt_pts.T)  # TEASER expects (3, N) layout

    solution = solver.getSolution()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = solution.rotation
    T[:3, 3] = solution.translation.flatten()

    return T


# ---------------------------------------------------------------------------
# RANSAC Fallback (when TEASER++ is unavailable)
# ---------------------------------------------------------------------------


def _solve_ransac_fallback(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    fpfh_src: o3d.pipelines.registration.Feature,
    fpfh_tgt: o3d.pipelines.registration.Feature,
    params: TeaserParams,
) -> np.ndarray:
    """
    Open3D global RANSAC based on FPFH feature matching.

    This is an APPROXIMATE fallback. It uses L2 cost (least squares)
    and therefore does NOT provide the TLS outlier rejection needed
    for subsurface scattering noise in the RePAIR dataset.

    For production use, install TEASER++ Python bindings.
    """
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            params.ransac_max_correspondence_distance
        ),
    ]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_src,
        pcd_tgt,
        fpfh_src,
        fpfh_tgt,
        mutual_filter=params.mutual_filter,
        max_correspondence_distance=params.ransac_max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        ransac_n=3,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=params.ransac_max_iterations,
            confidence=params.ransac_confidence,
        ),
    )

    return np.asarray(result.transformation)
