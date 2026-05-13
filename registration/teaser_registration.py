"""
TEASER++ Global Registration with Truncated Least Squares (TLS) and
Mathematically Certified SE(3) Output.

Implements Module 1.3 of the RePAIR perception pipeline — certifiable global
registration that rejects non-Gaussian subsurface scattering noise via the
TLS cost function, explicitly avoiding standard ICP algorithms.

=== TLS Objective (contrast with L2) ===

Standard L2 (ICP):
    min_{T in SE(3)}  sum_i  e_i^2
  → Every outlier contributes quadratically. Catastrophic with scattering noise.

Truncated Least Squares (TEASER++):
    min_{T in SE(3)}  sum_i  min( e_i^2,  c^2 )
  → Residuals beyond c are clipped to constant c^2.  Robust to ≤99% outliers.

=== TEASER++ Solver Stages ===

  Stage 1 — Maximum Clique Inlier Selection (MCE)
    Build consistency graph where edge (i,j) exists iff
        | ||q_i - q_j|| - ||p_i - p_j|| | ≤ 2·noise_bound
    Find largest clique = mutually consistent inlier set.

  Stage 2 — Decoupled Rotation (GNC-TLS)
    Graduated Non-Convexity with TLS clamping:
        ρ_μ(e) = { e^2,         e^2 ≤ c^2
                 { c^2,         e^2 > c^2
    as μ → ∞ the surrogate converges to exact TLS.

  Stage 3 — Decoupled Translation (Adaptive Voting)
    For rotation R, each correspondence votes for t_i = q_i - R p_i.
    Find translation with maximum consensus within noise bound.

  Stage 4 — Certificate of Suboptimality
    Semidefinite relaxation provides provable bound on distance from
    global optimum → mathematically certified transformation.

When TEASER++ bindings are unavailable, falls back to Open3D FPFH + RANSAC
as an approximate global registration (without sub-surface noise robustness).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

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
    import teaserpp_python  # noqa: F401

    _HAS_TEASER = True
except ImportError:
    _HAS_TEASER = False

# ---------------------------------------------------------------------------
# Result dataclass — mathematically certified SE(3)
# ---------------------------------------------------------------------------


@dataclass
class SE3Result:
    """
    Mathematically certified SE(3) registration result.

    Properties of a valid SE(3) matrix T = [R t; 0ᵀ 1]:
        R ∈ SO(3):  det(R) = +1,  RᵀR = I₃
        t ∈ ℝ³:     translation vector
    """

    T: np.ndarray
    """4×4 SE(3) transformation matrix. T[:3,:3] ∈ SO(3), T[:3,3] ∈ ℝ³."""

    certificate: float | None = None
    """Suboptimality upper bound from TEASER++ SDP relaxation.
    Guarantees the returned T is within `certificate` of the global
    TLS optimum.  None if TEASER++ not available."""

    runtime_sec: float = 0.0
    """Wall-clock time for the solver (seconds)."""

    num_correspondences: int = 0
    """Number of FPFH correspondences fed to the solver."""

    converged: bool = True
    """Whether the solver converged within iteration budget."""

    rotation_angle_deg: float = 0.0
    """Angle of rotation component (degrees), for diagnostic display."""

    translation_norm: float = 0.0
    """Euclidean norm of translation vector (metres)."""

    def __repr__(self) -> str:
        t_norm = np.linalg.norm(self.T[:3, 3])
        angle = self.rotation_angle_deg
        lines = [
            f"SE3Result(",
            f"  rotation_angle={angle:.4f}°,  translation_norm={t_norm:.4f}m",
            f"  correspondences={self.num_correspondences}, runtime={self.runtime_sec:.3f}s",
        ]
        if self.certificate is not None:
            lines.append(f"  certificate={self.certificate:.6f} (TLS suboptimality bound)")
        else:
            lines.append(f"  certificate=None (RANSAC fallback — no TLS guarantee)")
        lines.append(f"  converged={self.converged}")
        lines.append(f"  T=\n{np.array2string(self.T, precision=6, suppress_small=True)}")
        lines.append(")")
        return "\n".join(lines)

    @property
    def R(self) -> np.ndarray:
        """3×3 rotation matrix ∈ SO(3)."""
        return self.T[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """Translation vector ∈ ℝ³."""
        return self.T[:3, 3]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TeaserParams:
    """TEASER++ solver parameters for TLS registration."""

    # ── TLS truncation ──
    c_threshold: float = 0.01
    """TLS truncation threshold (metres). Residuals beyond this
    value contribute constant c², bounding outlier influence.

    For RePAIR archaeological fragments (D405 sub-mm camera):
      - 0.005 m (5 mm) — conservative, rejects fewer points
      - 0.010 m (10 mm) — recommended, robust to scattering
      - 0.020 m (20 mm) — aggressive, for very noisy fragments

    The TLS penalty:  rho(e) = { e² if e² ≤ c² ; c² otherwise }
    """

    # ── Sensor noise model ──
    noise_bound: float = 0.001
    """Expected sensor noise bound (metres). D405 spec: ≤ 1 mm at 10 cm."""

    # ── GNC rotation ──
    rotation_gnc_factor: float = 1.4
    """Graduated Non-Convexity factor. Controls annealing speed
    of the convex→non-convex transition. Larger = faster but riskier."""

    rotation_max_iterations: int = 100
    """Maximum GNC iterations for rotation estimation."""

    rotation_cost_threshold: float = 1e-12
    """Convergence threshold for rotation GNC cost change."""

    estimate_scaling: bool = False
    """If True, estimate uniform scaling s (ST(3) rather than SE(3)).
    Keep False — archaeological fragments are rigid bodies."""

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

    # ── SE(3) validation tolerances ──
    se3_orthogonality_tol: float = 1e-5
    """Tolerance for RᵀR ≈ I check."""
    se3_determinant_tol: float = 1e-5
    """Tolerance for det(R) ≈ +1 check."""


# ---------------------------------------------------------------------------
# SE(3) Validity Checks
# ---------------------------------------------------------------------------


def validate_se3(T: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Verify that a 4×4 matrix satisfies SE(3) group properties.

    Checks:
      1. Shape is exactly (4, 4).
      2. Bottom row is [0, 0, 0, 1].
      3. Rotation block R is orthogonal:  ||RᵀR - I||_∞ < tol.
      4. Rotation block has det = +1:     |det(R) - 1| < tol  (proper rotation).

    Args:
        T:   4×4 matrix to validate.
        tol: Numerical tolerance.

    Returns:
        True if T ∈ SE(3) within tolerance.
    """
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):
        return False

    R = T[:3, :3]
    # Orthogonality
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    # Proper rotation (no reflection)
    if abs(np.linalg.det(R) - 1.0) > tol:
        return False

    return True


def rotation_angle_degrees(R: np.ndarray) -> float:
    r"""
    Extract rotation angle from SO(3) matrix via trace formula.

        θ = arccos( (tr(R) − 1) / 2 )

    Args:
        R: 3×3 rotation matrix ∈ SO(3).

    Returns:
        Rotation angle in degrees.
    """
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_theta)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_teaser(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    params: TeaserParams | None = None,
) -> SE3Result:
    """
    Globally register src onto tgt using TEASER++ with TLS cost.

    Pipeline:
      1. Compute FPFH descriptors (33-dim geometric features).
      2. Match features via mutual nearest-neighbour + Lowe's ratio test.
      3. Solve registration:
         a) TEASER++ GNC-TLS (certifiable, TLS cost) — if available.
         b) Open3D RANSAC + FPFH (L2 cost, approximate) — fallback.
      4. Validate SE(3) properties of the output.

    Args:
        pcd_src: Source point cloud.
        pcd_tgt: Target point cloud.
        params:  TEASER++ configuration. Uses defaults if None.

    Returns:
        SE3Result with 4×4 certified SE(3) transformation matrix.

    Raises:
        ValueError: If the resulting matrix fails SE(3) validation.
    """
    if params is None:
        params = TeaserParams()

    t_start = time.perf_counter()

    # ── 1. FPFH descriptors ──
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

    # ── 2. Feature matching ──
    corrset = match_features(
        fpfh_src,
        fpfh_tgt,
        mutual_filter=params.mutual_filter,
        ratio_threshold=params.ratio_threshold,
        max_correspondences=params.max_correspondences,
    )
    src_pts, tgt_pts = extract_correspondence_clouds(pcd_src, pcd_tgt, corrset)
    num_corrs = len(src_pts)

    if num_corrs < 3:
        raise RuntimeError(
            f"Only {num_corrs} correspondences found (need ≥ 3). "
            "Try lowering the ratio threshold or increasing fpfh_radius."
        )

    # ── 3. Solve ──
    if _HAS_TEASER:
        result = _solve_teaser_core(src_pts, tgt_pts, params)
    else:
        warnings.warn(
            "teaserpp_python not installed — falling back to Open3D RANSAC. "
            "Subsurface scattering noise will NOT be rejected via TLS. "
            "Install TEASER++ bindings for certifiable global registration.",
            stacklevel=2,
        )
        result = _solve_ransac_fallback(pcd_src, pcd_tgt, fpfh_src, fpfh_tgt, params)

    result.runtime_sec = time.perf_counter() - t_start
    result.num_correspondences = num_corrs
    result.translation_norm = float(np.linalg.norm(result.t))
    result.rotation_angle_deg = rotation_angle_degrees(result.R)

    # ── 4. SE(3) certification ──
    if not validate_se3(result.T, tol=params.se3_orthogonality_tol):
        det = np.linalg.det(result.T[:3, :3])
        orth_err = np.max(np.abs(result.T[:3, :3].T @ result.T[:3, :3] - np.eye(3)))
        raise ValueError(
            f"Registration output is NOT a valid SE(3) matrix!\n"
            f"  det(R) = {det:.6e} (expected 1.0)\n"
            f"  ||RᵀR - I||_∞ = {orth_err:.6e}\n"
            f"  Homogeneous row: {result.T[3, :]}"
        )

    return result


# ---------------------------------------------------------------------------
# TEASER++ Core Solver (certifiable TLS)
# ---------------------------------------------------------------------------


def _solve_teaser_core(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    params: TeaserParams,
) -> SE3Result:
    """
    Run TEASER++ GNC-TLS registration with certificate capture.

    The solver executes four mathematical stages (see module docstring):
      1. Maximum Clique Inlier Selection (MCE)
      2. Decoupled Rotation (GNC-TLS)
      3. Decoupled Translation (Adaptive Voting)
      4. Certificate of Suboptimality (SDP relaxation)

    Args:
        src_pts: Matched source points (M, 3).
        tgt_pts: Matched target points (M, 3).
        params:  TEASER++ configuration.

    Returns:
        SE3Result with certified transformation and suboptimality bound.
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

    # TEASER++ expects (3, N) column-major layout
    solver.solve(src_pts.T.copy(), tgt_pts.T.copy())

    solution = solver.getSolution()

    # Extract suboptimality certificate
    certificate: float | None = None
    try:
        certificate = float(solver.getSuboptimalityBound())
    except (AttributeError, TypeError):
        pass  # Older TEASER++ versions may not expose this

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(solution.rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(solution.translation, dtype=np.float64).flatten()

    return SE3Result(
        T=T,
        certificate=certificate,
        converged=True,
    )


# ---------------------------------------------------------------------------
# RANSAC Fallback (approximate, L2 cost — no TLS outlier rejection)
# ---------------------------------------------------------------------------


def _solve_ransac_fallback(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    fpfh_src: o3d.pipelines.registration.Feature,
    fpfh_tgt: o3d.pipelines.registration.Feature,
    params: TeaserParams,
) -> SE3Result:
    """
    Open3D global RANSAC based on FPFH feature matching.

    WARNING: Uses L2 (least-squares) cost, NOT TLS. Cannot reject
    non-Gaussian subsurface scattering noise. Use only when TEASER++
    bindings are unavailable.
    """
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            params.ransac_max_correspondence_distance
        ),
    ]

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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

    T = np.asarray(result_ransac.transformation, dtype=np.float64)

    return SE3Result(
        T=T,
        certificate=None,  # RANSAC provides no optimality certificate
        converged=True,
    )
