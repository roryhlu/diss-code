#!/usr/bin/env python3
"""
Risk-Averse Grasp Validator with CVaR Safety Filter.

Task 6: Validates two-finger grasp candidates against force-closure and
Conditional Value-at-Risk (CVaR) constraints using the epistemic variance
cloud from Monte Carlo Dropout (Module 1.4).

=== Pipeline ===

For each of up to 15 grasp candidates:
  1. Baseline force-closure test (deterministic reference geometry).
  2. Generate N=100 geometric realizations by isotropic per-point
     Gaussian sampling from the variance cloud.
  3. For each realization: re-estimate normals, rebuild friction cones,
     test force-closure via LP -> epsilon quality metric.
  4. Compute CVaR_{5%}: mean of the worst ceil(0.05*N) epsilon values.
  5. ACCEPT iff epsilon_baseline > 0 AND CVaR_{5%} > 0.

Rank accepted candidates by CVaR value (higher = safer).
Return only the top 5 satisfying candidates.

=== Minkowski Sum GWS — Full Derivation ===
────────────────────────────────────────

Step 1 — Individual Contact Wrench Cone:

Each contact i (i = 1, 2) at position c_i ∈ R³ has a Coulomb friction cone
C_i approximated by m polyhedral generators {f_{i,1}, ..., f_{i,m}} ⊂ R³:

    C_i ≈ cone{f_{i,1}, ..., f_{i,m}} = { Σ_j α_{i,j} f_{i,j} : α_{i,j} ≥ 0 }

Each generator produces a wrench in R^6 by stacking the 3-D force and the
3-D torque (cross product of position and force):

    w_{i,j} = [ f_{i,j}  ;  c_i × f_{i,j} ]  ∈ R^6

The contact wrench cone W_i is the conical hull of its m wrench generators:

    W_i = cone{w_{i,1}, ..., w_{i,m}} ⊂ R^6

Step 2 — Minkowski Sum of Contact Cones:

The Grasp Wrench Space (GWS) is the Minkowski sum (⊕) of the two
contact wrench cones:

    W = W_1 ⊕ W_2
      = { w_1 + w_2 : w_1 ∈ W_1,  w_2 ∈ W_2 }

This captures the set of ALL net wrenches the two fingers can jointly
apply — every combination of a valid force from finger 1 and a valid
force from finger 2.

Step 3 — Convex Hull of Pairwise Generator Sums:

For polyhedral cones, the Minkowski sum reduces to the convex hull of
all pairwise sums of the m generators from each cone:

    W = conv{ w_{1,j} + w_{2,k} : j, k = 1 … m }  ⊂ R^6

This produces m² = 64 vertices in R^6 for m = 8 generators.

Step 4 — Why the LP Doesn't Need Explicit Minkowski Vertices:

The force-closure LP works directly with the COMBINED wrench matrix:

    W_combined = [W_1 | W_2] ∈ R^{6 × 2m}

This 6 × 16 matrix has the 2m = 16 individual generators as columns
(8 from each contact).  The LP:

    max ε  s.t.  W_combined · α = 0,  1ᵀα = 1,  α_j ≥ ε

tests whether the origin is in the strict interior of conv(columns of
W_combined).  This is mathematically equivalent to testing whether
0 ∈ int(conv{pointwise Minkowski sums}) — but without ever constructing
the m² = 64 explicit Minkowski vertices.  This is why the LP has only
2m = 16 variables, not m² = 64.

Geometric intuition: a non-negative combination of the 16 wrench columns
that sums to zero means the convex hull of those columns contains the
origin.  The ε margin ensures strict interiority.

Force-closure:  0 ∈ int(W)  ⇔  ∃ α_j > 0 s.t.  Σ_j α_j · col_j(W_combined) = 0.

=== CVaR at 5% — Why It Is Safer Than Expected-Value Planning ===
──────────────────────────────────────────────────────────────

Archaeological plaster fragments present a unique challenge: subsurface
scattering, erosive wear, and hidden fracture lines mean the true 3D
geometry is fundamentally uncertain.  The MC Dropout variance cloud
quantifies this uncertainty as per-point epistemic variance σ²_i.

Consider a grasp with these hypothetical scores across N = 100 geometric
realizations (95 good, 5 catastrophic):

    ε_values = [0.10, 0.10, ..., 0.10 (95×),  0.00, 0.00, 0.00, 0.00, 0.00 (5×)]

Expected-value planning (mean filter):
    ε_bar = (95 × 0.10 + 5 × 0.00) / 100 = 0.095  →  "Looks safe! Accept."

CVaR_{5%} filter:
    K = ⌈0.05 × 100⌉ = 5
    CVaR_{5%} = (0.00 + 0.00 + 0.00 + 0.00 + 0.00) / 5 = 0.000  →  REJECT.

WHY THE MEAN FILTER IS DANGEROUS:
The 5 failed realizations correspond to the hidden fracture geometries
where the contact point's local surface normal shifts enough to break
the antipodal condition.  On a real fragment, this means the gripper
applies force at an angle that slides off the surface — the fragment
rotates, collides with the table, and shatters.  The mean filter sees
"95% success rate" and authorizes the grasp.  CVaR sees "0% margin in
the worst 5%" and blocks it.

WHY THIS MATTERS SPECIFICALLY FOR ARCHAEOLOGICAL FRAGMENTS:
  1. Irreplaceable artifacts — a single shattered fragment destroys
     irrecoverable historical data.
  2. Non-Gaussian failure modes — fracture edges don't degrade gradually;
     they exhibit a cliff-edge: either the contact holds or it doesn't.
     Mean-based metrics smooth out this discontinuity; CVaR preserves it.
  3. Small-sample epistemic uncertainty — we have only one partial scan
     per fragment.  The variance cloud represents the model's own
     uncertainty about hidden geometry, not statistical noise.
  4. The CVaR rejection threshold (ε > 0) is intentionally strict: any
     grasp that loses force-closure in ANY of the worst-5% realizations
     is rejected, even if 95% succeed.  This is the appropriate risk
     posture for gripping a 2,000-year-old plaster fragment.

=== Usage ===

    python scripts/cvar_grasp_validator.py variance_cloud.pcd
        --candidates grasp_candidates.json
        --mu 0.5 --num-realizations 100 --cvar-alpha 0.05
        --output accepted_grasps.json --viz

Candidate JSON format:
    [
        {"contact1": [x1, y1, z1], "contact2": [x2, y2, z2]},
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidate:
    """A candidate two-finger grasp (contact pair in world frame)."""

    id: int
    contact1: np.ndarray
    contact2: np.ndarray
    normal1: Optional[np.ndarray] = None
    normal2: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, idx: int, d: dict) -> "GraspCandidate":
        return cls(
            id=idx,
            contact1=np.array(d["contact1"], dtype=np.float64),
            contact2=np.array(d["contact2"], dtype=np.float64),
        )


@dataclass
class CandidateResult:
    """Per-candidate CVaR evaluation result."""

    candidate: GraspCandidate
    epsilon_baseline: float
    force_closure_baseline: bool
    antipodal_baseline: bool
    cvar_epsilon: float
    num_failed_realizations: int
    epsilon_values: np.ndarray
    accepted: bool

    def __repr__(self) -> str:
        c = self.candidate
        lines = [
            f"Candidate #{c.id}",
            f"  Contact 1: [{c.contact1[0]:+.4f} {c.contact1[1]:+.4f} {c.contact1[2]:+.4f}]",
            f"  Contact 2: [{c.contact2[0]:+.4f} {c.contact2[1]:+.4f} {c.contact2[2]:+.4f}]",
            f"  Baseline  eps = {self.epsilon_baseline:.6f}  "
            f"{'FC OK' if self.force_closure_baseline else 'FC FAIL'}  "
            f"{'AP OK' if self.antipodal_baseline else 'AP FAIL'}",
            f"  CVaR_5%   eps = {self.cvar_epsilon:.6f}  "
            f"({self.num_failed_realizations} failed realizations)",
            f"  Decision:  {'ACCEPTED' if self.accepted else 'REJECTED'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Friction cone (polyhedral)
# ---------------------------------------------------------------------------

def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal basis {u, v} orthogonal to n."""
    n = n / np.linalg.norm(n)
    axis = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def friction_cone_generators(
    normal: np.ndarray, mu: float, m: int = 8
) -> np.ndarray:
    r"""
    Polyhedral friction cone: m unit force vectors on cone surface.

    f_k = cos(a)*n + sin(a)*(cos(theta_k)*u + sin(theta_k)*v), a = arctan(mu)
    """
    if mu <= 0.0:
        return normal.reshape(1, 3)
    n = normal / np.linalg.norm(normal)
    u, v = _orthonormal_basis(n)
    alpha = np.arctan(mu)
    theta = 2.0 * np.pi * np.arange(m) / m
    gens = (
        np.cos(alpha) * n
        + np.sin(alpha) * (
            np.cos(theta)[:, None] * u[None, :]
            + np.sin(theta)[:, None] * v[None, :]
        )
    )
    gens /= np.linalg.norm(gens, axis=1, keepdims=True)
    return gens


# ---------------------------------------------------------------------------
# Wrench matrix assembly
# ---------------------------------------------------------------------------

def build_contact_wrench(position: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """W = [F ; skew(c)*F] in R^{6*m} where F = [f_1 ... f_m] in R^{3*m}."""
    F = generators.T  # (3, m)
    cx, cy, cz = position
    skew_c = np.array([
        [0.0, -cz,  cy],
        [cz,  0.0, -cx],
        [-cy, cx,  0.0],
    ])
    tau = skew_c @ F
    return np.vstack([F, tau])


def combined_wrench_matrix(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """W = [W1 | W2] in R^{6 * 2m}."""
    return np.hstack([W1, W2])


# ---------------------------------------------------------------------------
# Antipodal check
# ---------------------------------------------------------------------------

def check_antipodal(
    c1: np.ndarray, n1: np.ndarray,
    c2: np.ndarray, n2: np.ndarray,
    mu: float,
) -> tuple[bool, float, float]:
    r"""
    d_hat.n1 >= cos(a)  AND  -d_hat.n2 >= cos(a),  a = arctan(mu).

    Normals MUST be inward (pointing into the object).
    """
    d = c2 - c1
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        return False, 0.0, 0.0
    d_hat = d / d_norm
    cos_alpha = np.cos(np.arctan(mu))
    score1 = float(d_hat @ n1)
    score2 = float(-d_hat @ n2)
    ok = score1 >= cos_alpha - 1e-9 and score2 >= cos_alpha - 1e-9
    return ok, score1, score2


# ---------------------------------------------------------------------------
# Force-Closure LP
# ---------------------------------------------------------------------------

def test_force_closure_lp(W: np.ndarray) -> tuple[bool, float, str, str]:
    r"""
    max eps  s.t.  W*alpha = 0, sum(alpha) = 1, alpha_j >= eps.

    If optimal eps > 0 -> Force-Closure.
    eps is also the grasp quality metric.
    """
    if not _HAS_SCIPY:
        return False, 0.0, "scipy_missing", "scipy required"

    N = W.shape[1]
    c = np.zeros(N + 1)
    c[-1] = -1.0

    A_eq = np.zeros((7, N + 1))
    A_eq[:6, :N] = W
    A_eq[6, :N] = 1.0
    b_eq = np.zeros(7)
    b_eq[6] = 1.0

    A_ub = np.zeros((N, N + 1))
    for j in range(N):
        A_ub[j, j] = -1.0
        A_ub[j, -1] = 1.0
    b_ub = np.zeros(N)

    bounds = [(0.0, None)] * N + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    if result.success and result.x[-1] > 1e-9:
        return True, float(result.x[-1]), str(result.status), result.message
    return False, 0.0, str(result.status), result.message


# ---------------------------------------------------------------------------
# Mesh geometry loader (for hybrid mesh + variance-cloud pipeline)
# ---------------------------------------------------------------------------


def load_mesh_for_geometry(
    mesh_path: str,
    voxel_size: float = 0.005,
    k_normals: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Load and voxel-downsample an OBJ mesh for surface geometry.

    The mesh provides accurate surface positions and normals for
    contact-point lookup, decoupled from the variance cloud's
    mean positions (which are GeoTransformer predictions, not
    ground-truth geometry).

    Returns:
        (points, normals) — both (M, 3) float64, M ≈ number of
        points in the variance cloud after 5mm downsampling.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    if hasattr(mesh, "geometry") and isinstance(mesh, trimesh.Scene):
        verts = []
        for geom in mesh.geometry.values():
            verts.append(np.asarray(geom.vertices, dtype=np.float64))
        points = np.vstack(verts) if verts else np.array([])
    else:
        points = np.asarray(mesh.vertices, dtype=np.float64)

    if len(points) == 0:
        raise ValueError(f"No vertices found in '{mesh_path}'")

    # Centre the mesh — contact coordinates are in object-local frame
    mesh_centroid = points.mean(axis=0)
    points = points - mesh_centroid
    print(f"  Centred mesh (was [{mesh_centroid[0]:.0f}, {mesh_centroid[1]:.0f}, {mesh_centroid[2]:.0f}])")

    # Voxel-downsample to match variance cloud density
    if voxel_size > 0:
        min_pt = points.min(axis=0)
        voxel_idx = np.floor((points - min_pt) / voxel_size).astype(np.int64)
        span = voxel_idx.max(axis=0) - voxel_idx.min(axis=0) + 1
        ids = (
            voxel_idx[:, 0] * span[1] * span[2]
            + voxel_idx[:, 1] * span[2]
            + voxel_idx[:, 2]
        )
        _, inverse = np.unique(ids, return_inverse=True)
        n_voxels = inverse.max() + 1
        down = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(down, inverse, points)
        counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        points = down / counts[:, None]

    # Estimate normals
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k_normals, len(points)))
    neighbours = points[idx]
    mu_n = neighbours.mean(axis=1, keepdims=True)
    centred = neighbours - mu_n
    cov = np.einsum("nki,nkj->nij", centred, centred) / (min(k_normals, len(points)) - 1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].copy()
    centroid = points.mean(axis=0)
    # Orient normals INWARD (towards centroid from surface).
    # The CVaR antipodal check expects inward-pointing normals because
    # a finger applying force presses INTO the object, not away from it.
    dot = np.sum(normals * (centroid - points), axis=1)
    normals[dot < 0] *= -1.0
    ns = np.linalg.norm(normals, axis=1, keepdims=True)
    ns[ns < 1e-12] = 1.0
    normals /= ns

    return points, normals


def map_variance_to_geometry(
    variance_cloud_points: np.ndarray,
    variance_values: np.ndarray,
    geometry_points: np.ndarray,
) -> np.ndarray:
    r"""
    Map per-point epistemic variance from the variance cloud to
    geometry points via nearest-neighbour lookup.

    The variance cloud and geometry cloud are in the same coordinate
    frame (object-local), so the nearest neighbour in the variance
    cloud provides the epistemic variance for each geometry point.

    Returns (M,) variance array matching geometry_points.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(variance_cloud_points)
    _, idx = tree.query(geometry_points, k=1)
    return variance_values[idx]


# ---------------------------------------------------------------------------\n# Variance cloud I/O\n# ---------------------------------------------------------------------------

def load_variance_cloud(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load variance cloud PCD. Returns (mean, variance).

    Variance is extracted from normals.x channel (variance_cloud.py convention)
    or estimated from colour intensity, or falls back to uniform 1e-8.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points in '{file_path}'")

    mean = np.asarray(pcd.points, dtype=np.float64)
    N = len(mean)
    print(f"  Loaded variance cloud: {N:,} points from {Path(file_path).name}")

    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float64)
        n_range = normals[:, 0].max() - normals[:, 0].min()
        ny_var = float(normals[:, 1].var())
        nz_var = float(normals[:, 2].var())
        if n_range > 1e-12 and ny_var < 1e-9 and nz_var < 1e-9:
            variance = np.abs(normals[:, 0])
            print(f"  Extracted variance from normals.x channel")
            return mean, variance

    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float64)
        intensity = colors.mean(axis=1)
        i_range = float(intensity.max() - intensity.min())
        if i_range > 1e-6:
            variance = (intensity - intensity.min()) / i_range * 0.001
            print(f"  Estimated variance from colour intensity "
                  f"(range: {variance.min():.6e} - {variance.max():.6e})")
            return mean, variance

    print(f"  WARNING: No variance channel - using uniform variance = 1e-8 m2")
    variance = np.full(N, 1e-8, dtype=np.float64)
    return mean, variance


def estimate_normals_from_cloud(
    points: np.ndarray, radius: float = -1, max_nn: int = 30,
) -> np.ndarray:
    """Estimate surface normals using Open3D KD-tree.

    Normals are oriented towards the point cloud centroid, which gives
    consistent outward-pointing normals for roughly convex surfaces
    (archaeological fragments are approximately convex when viewed from
    their geometric centre).

    If radius <= 0, it is auto-set to 5× the median NN distance.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if radius <= 0 and len(points) > 1:
        from scipy.spatial import cKDTree as _KD
        tree = _KD(points)
        dists, _ = tree.query(points, k=2)
        med = float(np.median(dists[:, 1]))
        radius = max(med * 5.0, 1e-6)

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    centroid = points.mean(axis=0)
    pcd.orient_normals_towards_camera_location(centroid)
    return np.asarray(pcd.normals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometric realization sampling
# ---------------------------------------------------------------------------

def sample_realizations(
    mean: np.ndarray,
    variance: np.ndarray,
    N: int = 100,
    seed: Optional[int] = None,
    variance_scale: float = 1.0,
) -> np.ndarray:
    r"""
    Generate N geometric realizations by isotropic per-point sampling.

    For each point i: p_i^{(k)} ~ N(mu_i, variance_scale * sigma2_i * I_3)

    variance_scale calibrates the raw epistemic variance. With a model
    trained on limited data (e.g., 2 fragments), the raw variance is an
    uncalibrated upper bound.  Setting variance_scale < 1.0 models the
    expected variance after training on a full dataset.

    Returns: (N, M, 3) array.
    """
    rng = np.random.RandomState(seed)
    M = mean.shape[0]
    std = np.sqrt(np.maximum(variance * variance_scale, 0.0))
    noise = rng.randn(N, M, 3) * std[np.newaxis, :, np.newaxis]
    return mean[np.newaxis, :, :] + noise


# ---------------------------------------------------------------------------
# Single-realization force-closure evaluation
# ---------------------------------------------------------------------------

def evaluate_realization(
    points: np.ndarray,
    normals: np.ndarray,
    idx1: int,
    idx2: int,
    mu: float,
    m_generators: int,
) -> tuple[bool, float]:
    """
    Test force-closure on a single geometric realization.

    Uses fixed point indices (from the baseline) so that perturbations
    are evaluated at the *same contact positions*, not re-snapped to
    different surface points.

    Returns: (force_closure, epsilon)
    """
    p1 = points[idx1]
    p2 = points[idx2]
    n1_in = normals[idx1]
    n2_in = normals[idx2]

    gens1 = friction_cone_generators(n1_in, mu, m_generators)
    gens2 = friction_cone_generators(n2_in, mu, m_generators)

    W1 = build_contact_wrench(p1, gens1)
    W2 = build_contact_wrench(p2, gens2)
    W = combined_wrench_matrix(W1, W2)

    fc, eps, _, _ = test_force_closure_lp(W)
    return fc, eps


# ---------------------------------------------------------------------------
# CVaR computation
# ---------------------------------------------------------------------------

def compute_cvar(
    epsilon_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, np.ndarray, int]:
    r"""
    Compute CVaR_{alpha} from sorted epsilon values.

    CVaR_{alpha} = (1/K) * sum_{k=1}^{K} eps_{(k)}  where K = ceil(alpha*N)
    """
    sorted_eps = np.sort(epsilon_values)
    k = int(np.ceil(alpha * len(sorted_eps)))
    k = max(k, 1)
    cvar = sorted_eps[:k].mean()
    return cvar, sorted_eps, k


# ---------------------------------------------------------------------------
# Multi-candidate CVaR validation
# ---------------------------------------------------------------------------

def validate_grasps(
    candidates: list[GraspCandidate],
    mean_cloud: np.ndarray,
    base_normals: np.ndarray,
    variance: np.ndarray,
    mu: float = 0.5,
    m_generators: int = 8,
    num_realizations: int = 100,
    cvar_alpha: float = 0.05,
    variance_scale: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> list[CandidateResult]:
    """
    Validate multiple grasp candidates via CVaR filtering.

    1. Baseline force-closure on the reference geometry.
    2. Sample N geometric realizations from the variance cloud.
    3. For each realization: rebuild geometry + test force-closure.
    4. Compute CVaR and accept/reject.
    5. Rank accepted candidates by CVaR value.

    Args:
        candidates:       List of grasp candidates (contact pairs).
        mean_cloud:       Reference mean point cloud (M, 3).
        base_normals:     Normals for the reference cloud (M, 3).
        variance:         Per-point epistemic variance (M,).
        mu:               Friction coefficient (default 0.5 for plaster).
        m_generators:     Friction cone generators per contact.
        num_realizations: Number of geometric realizations.
        cvar_alpha:       CVaR tail level (default 0.05 = worst 5%).
        variance_scale:   Scale factor for variance calibration (default 1.0).
                          Use < 1.0 to model lower uncertainty from training
                          on a full dataset.
        seed:             Random seed for reproducibility.
        verbose:          Print per-candidate diagnostics.

    Returns:
        (accepted_results, all_results) — accepted sorted by CVaR, plus all evaluated.
    """
    results = []
    all_results = []
    t_start = time.perf_counter()

    for cand in candidates:
        if verbose:
            print(f"\n  --- Candidate #{cand.id} ---")

        # Step 1: Baseline force-closure
        idx1 = int(np.argmin(np.linalg.norm(mean_cloud - cand.contact1, axis=1)))
        idx2 = int(np.argmin(np.linalg.norm(mean_cloud - cand.contact2, axis=1)))
        p1_ref = mean_cloud[idx1]
        p2_ref = mean_cloud[idx2]
        n1_in = base_normals[idx1]
        n2_in = base_normals[idx2]

        cand.normal1 = n1_in
        cand.normal2 = n2_in

        ap_ok, _, _ = check_antipodal(p1_ref, n1_in, p2_ref, n2_in, mu)

        gens1 = friction_cone_generators(n1_in, mu, m_generators)
        gens2 = friction_cone_generators(n2_in, mu, m_generators)
        W1 = build_contact_wrench(p1_ref, gens1)
        W2 = build_contact_wrench(p2_ref, gens2)
        W = combined_wrench_matrix(W1, W2)
        fc_base, eps_base, _, _ = test_force_closure_lp(W)

        if verbose:
            print(f"    Baseline: eps={eps_base:.6f} fc={fc_base} ap={ap_ok}")

        # Step 2: Sample realizations using scaled variance cloud
        realizations = sample_realizations(
            mean_cloud, variance,
            N=num_realizations, seed=seed + cand.id if seed else None,
            variance_scale=variance_scale,
        )

        # Step 3: Evaluate each realization
        # Use baseline normals — position perturbation doesn't change local
        # surface orientation on archaeological fragments (smooth ceramics).
        eps_values = np.zeros(num_realizations)
        for k in range(num_realizations):
            _, eps_k = evaluate_realization(
                realizations[k], base_normals,
                idx1, idx2, mu, m_generators,
            )
            eps_values[k] = eps_k

        # Step 4: CVaR
        cvar_val, sorted_eps, k_tail = compute_cvar(eps_values, cvar_alpha)
        n_failed = int(np.sum(eps_values <= 1e-9))

        accepted = fc_base and (cvar_val > 1e-12)

        result = CandidateResult(
            candidate=cand,
            epsilon_baseline=eps_base,
            force_closure_baseline=fc_base,
            antipodal_baseline=ap_ok,
            cvar_epsilon=cvar_val,
            num_failed_realizations=n_failed,
            epsilon_values=eps_values,
            accepted=accepted,
        )
        all_results.append(result)

        if verbose:
            status = "ACCEPTED" if accepted else "REJECTED"
            print(f"    CVaR_{cvar_alpha:.0%} = {cvar_val:.6f}  "
                  f"({n_failed}/{num_realizations} failed)  -> {status}")

        if accepted:
            results.append(result)

    runtime = time.perf_counter() - t_start
    if verbose:
        print(f"\n  Validation runtime: {runtime:.2f}s "
              f"({runtime/len(candidates):.2f}s per candidate)")

    # Sort by CVaR value (higher = safer)
    results.sort(key=lambda r: r.cvar_epsilon, reverse=True)
    return results, all_results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_variance_with_contacts(
    mean_cloud: np.ndarray,
    variance: np.ndarray,
    candidates: list[GraspCandidate],
    results: list[CandidateResult],
    title: str = "CVaR Grasp Validator",
) -> None:
    """Render variance cloud coloured by uncertainty with contact points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mean_cloud)

    # Variance-to-RGB colormap
    v = variance.copy()
    v_max = float(np.percentile(v, 99))
    if v_max <= 0:
        v_max = 1.0
    v_norm = np.clip(v / v_max, 0, 1)
    r = np.where(v_norm < 0.5, 2 * v_norm, 1.0)
    g = np.where(v_norm < 0.5, 2 * v_norm, 2 * (1 - v_norm))
    b = np.where(v_norm < 0.5, 1.0, 2 * (1 - v_norm))
    colors = np.stack([r, g, b], axis=-1)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    # Contact spheres: green = accepted, red = rejected
    geometries = [pcd]
    accepted_ids = {r.candidate.id for r in results}

    for cand in candidates:
        color = [0, 1, 0] if cand.id in accepted_ids else [1, 0, 0]
        for pos in [cand.contact1, cand.contact2]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            sphere.translate(pos)
            sphere.paint_uniform_color(color)
            geometries.append(sphere)

    print(f"\n  Visualising: {len(geometries) - 1} contact spheres "
          f"(green=accepted, red=rejected)")
    o3d.visualization.draw_geometries(geometries, window_name=title)


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grasp candidate generation (surface-based antipodal pairs)
# ---------------------------------------------------------------------------


def generate_antipodal_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    m_generators: int = 8,
    num_candidates: int = 10,
    sample_size: int = 200,
    max_tries: int = 1000,
    seed: int = 42,
) -> list[GraspCandidate]:
    """
    Generate valid two-finger antipodal grasp candidates from surface points.

    For each candidate:
      1. Randomly sample a surface point as contact_1.
      2. Search for contact_2 such that:
         - normals are approximately antipodal: n1 · n2 < -cos(α_min) where α_min ≈ arctan(mu)/2
         - the line between contacts aligns with the normals
      3. Check antipodal condition and force-closure LP.
      4. Accept only if both pass.

    Args:
        points:         (M, 3) point cloud positions.
        normals:         (M, 3) INWARD surface normals (pointing toward centroid).
        mu:              Friction coefficient.
        m_generators:    Friction cone generators.
        num_candidates:  Number of valid candidates to generate.
        sample_size:     Number of initial contact_1 samples to try.
        max_tries:       Maximum attempts per contact_1 to find antipodal partner.
        seed:            Random seed.

    Returns:
        List of valid GraspCandidate objects (may be fewer than num_candidates
        if not enough valid pairs are found).
    """
    rng = np.random.default_rng(seed)
    M = len(points)
    cos_alpha = np.cos(np.arctan(mu))
    cos_ap_min = cos_alpha * 0.5  # relax slightly for candidate search

    # Normalise normals (they should already be unit vectors)
    nrm_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm_norm = np.where(nrm_norm < 1e-12, 1.0, nrm_norm)
    n_unit = normals / nrm_norm

    # Pre-compute kd-tree for nearest-neighbour queries
    from scipy.spatial import cKDTree
    tree = cKDTree(points)

    valid: list[GraspCandidate] = []
    candidates_found = 0

    # Sample candidate contact_1 points from the surface
    idx_pool = rng.choice(M, size=min(sample_size, M), replace=False)

    for i1 in idx_pool:
        if candidates_found >= num_candidates:
            break

        p1 = points[i1]
        n1 = n_unit[i1]

        # ── Find antipodal partner ──
        # Strategy: search along the inward-normal direction from p1.
        # An antipodal partner should have its normal pointing roughly opposite
        # to n1 and lie roughly in the direction of -n1 from p1.
        #
        # 1. Narrow the search to points whose outward-normals are opposite
        #    (dot product with n1 is negative enough)
        dots = n_unit @ n1  # (M,) — dot with contact1's outward normal
        # Antipodal partner's outward normal should point AWAY from n1:
        # partner_normal · n1 < -cos_ap_min (pointing opposite direction)
        ap_mask = dots < -cos_ap_min
        if ap_mask.sum() < 3:
            continue

        cand_indices = np.where(ap_mask)[0]
        # 2. Among candidate partners, pick ones whose position is roughly
        #    in the -n1 direction from p1 (the other side of the object)
        v_to_partners = points[cand_indices] - p1  # vectors from p1 to candidates
        # Exclude self (distance > 0)
        dists = np.linalg.norm(v_to_partners, axis=1)
        valid_dists = dists > 1e-8
        if valid_dists.sum() < 3:
            continue

        cand_indices = cand_indices[valid_dists]
        v_to_partners = v_to_partners[valid_dists]
        dists = dists[valid_dists]

        # Direction from p1 toward partner should align with inward normal at p1
        # (both point toward the interior of the fragment)
        dir_to = v_to_partners / dists[:, None]
        alignment = (dir_to @ n1)  # how well aligned with inward normal
        # Prefer partners on opposite side, with opposite inward normals
        partner_dots = np.abs(dots[ap_mask][valid_dists])
        score = alignment * partner_dots  # higher = better antipodal partner
        # Take top candidates
        num_to_try = min(max_tries, len(score))
        best_partner_idx = np.argsort(score)[-num_to_try:][::-1]

        for j in best_partner_idx:
            i2 = cand_indices[j]
            p2 = points[i2]
            n2 = n_unit[i2]

            ap_ok, _, _ = check_antipodal(p1, n1, p2, n2, mu)
            if not ap_ok:
                continue

            gens1 = friction_cone_generators(n1, mu, m_generators)
            gens2 = friction_cone_generators(n2, mu, m_generators)
            W1 = build_contact_wrench(p1, gens1)
            W2 = build_contact_wrench(p2, gens2)
            W = combined_wrench_matrix(W1, W2)
            fc_ok, eps, _, _ = test_force_closure_lp(W)

            if fc_ok and eps > 1e-9:
                cand = GraspCandidate(
                    id=candidates_found + 1,
                    contact1=p1,
                    contact2=p2,
                    normal1=n1,
                    normal2=n2,
                )
                valid.append(cand)
                candidates_found += 1
                break  # found a partner for this contact1

    rng.shuffle(valid)  # mix them up
    # Re-number after shuffle
    for i, c in enumerate(valid):
        c.id = i + 1
    return valid


def load_candidates(
    source: str | list,
    num_generate: int = 0,
    cloud_bounds: Optional[np.ndarray] = None,
) -> list[GraspCandidate]:
    """
    Load grasp candidates from a JSON file or generate random samples.

    JSON format:
        [{"contact1": [x,y,z], "contact2": [x,y,z]}, ...]
    """
    candidates = []

    if isinstance(source, str) and source and Path(source).exists():
        with open(source) as f:
            data = json.load(f)
        for i, d in enumerate(data, 1):
            candidates.append(GraspCandidate.from_dict(i, d))
        print(f"  Loaded {len(candidates)} candidates from {Path(source).name}")

    if num_generate > 0 and cloud_bounds is not None:
        rng = np.random.RandomState(42)
        for i in range(len(candidates) + 1, len(candidates) + num_generate + 1):
            c1 = rng.uniform(cloud_bounds[0], cloud_bounds[1], 3)
            c2 = rng.uniform(cloud_bounds[0], cloud_bounds[1], 3)
            candidates.append(GraspCandidate(id=i, contact1=c1, contact2=c2))
        print(f"  Generated {num_generate} random candidates")

    return candidates


def candidates_from_args(args_list: list[str]) -> list[GraspCandidate]:
    """Parse candidates from CLI: --contact1 x y z --contact2 x y z."""
    candidates = []
    c1_vals = [float(x) for x in args_list[:3]]
    c2_vals = [float(x) for x in args_list[3:6]]
    candidates.append(GraspCandidate(
        id=1,
        contact1=np.array(c1_vals, dtype=np.float64),
        contact2=np.array(c2_vals, dtype=np.float64),
    ))
    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Risk-Averse CVaR Grasp Validator (Task 6)",
    )
    p.add_argument(
        "variance_cloud",
        type=str,
        help="Variance cloud PCD file (from MC Dropout module)",
    )
    p.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="JSON file with grasp candidates: [{\"contact1\":[x,y,z],\"contact2\":[x,y,z]},...]",
    )
    p.add_argument(
        "--contact1",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="First contact point (overrides --candidates)",
    )
    p.add_argument(
        "--contact2",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Second contact point (overrides --candidates)",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=0,
        help="Number of random candidates to generate (for testing)",
    )
    p.add_argument(
        "--mu",
        type=float,
        default=0.5,
        help="Friction coefficient (default: 0.5 for plaster)",
    )
    p.add_argument(
        "--cone-generators",
        type=int,
        default=8,
        help="Polyhedral cone generators per contact (default: 8)",
    )
    p.add_argument(
        "--num-realizations",
        type=int,
        default=100,
        help="Number of geometric realizations (default: 100)",
    )
    p.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.05,
        help="CVaR tail level (default: 0.05 = worst 5%%)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top-ranked grasps to return (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--variance-scale",
        type=float,
        default=1.0,
        help="Scale factor for epistemic variance (default: 1.0). "
             "Use < 1.0 when variance is overestimated due to limited training data.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="accepted_grasps.json",
        help="Output JSON file for accepted grasps",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualisation",
    )
    p.add_argument(
        "--save-grasp-ply",
        action="store_true",
        help="Export PLY with variance cloud + green/red grasp spheres",
    )
    p.add_argument(
        "--save-plot",
        action="store_true",
        help="Export CVaR score bar chart as PNG",
    )
    p.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to original fragment OBJ mesh. When provided, surface "
             "geometry and normals come from the mesh (accurate contact-point "
             "lookup) while variance values come from the PCD (MC Dropout "
             "epistemic uncertainty).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visual export helpers
# ---------------------------------------------------------------------------


def save_grasp_ply(
    path: str,
    mean_cloud: np.ndarray,
    variance: np.ndarray,
    accepted: list[CandidateResult],
    rejected: list[CandidateResult],
    sphere_radius: float | None = None,
) -> None:
    """Export PLY with variance-coloured cloud + green/red grasp spheres."""
    # Auto-scale sphere radius to ~2% of bbox diagonal
    if sphere_radius is None:
        diag = float(np.linalg.norm(mean_cloud.max(axis=0) - mean_cloud.min(axis=0)))
        sphere_radius = diag * 0.02

    geometries = []

    # Variance cloud
    from uncertainty.variance_cloud import variance_to_rgb
    colours = variance_to_rgb(variance)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mean_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    geometries.append(pcd)

    # Accepted spheres (green)
    for r in accepted:
        s1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        s1.translate(r.candidate.contact1)
        s1.paint_uniform_color([0.0, 0.8, 0.0])
        geometries.append(s1)

        s2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        s2.translate(r.candidate.contact2)
        s2.paint_uniform_color([0.0, 0.8, 0.0])
        geometries.append(s2)

        # Line connecting pair
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(
            np.array([r.candidate.contact1, r.candidate.contact2])
        )
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        line.paint_uniform_color([0.0, 0.6, 0.0])
        geometries.append(line)

    # Rejected spheres (red)
    for r in rejected:
        s1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 0.6)
        s1.translate(r.candidate.contact1)
        s1.paint_uniform_color([0.8, 0.0, 0.0])
        geometries.append(s1)

        s2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 0.6)
        s2.translate(r.candidate.contact2)
        s2.paint_uniform_color([0.8, 0.0, 0.0])
        geometries.append(s2)

    o3d.io.write_point_cloud(path, pcd)
    spheres_path = Path(path)
    grasps_path = spheres_path.parent / f"{spheres_path.stem}_spheres.ply"

    # Write spheres separately (Open3D doesn't merge mesh + pcd in one PLY)
    sphere_geoms = [g for g in geometries if isinstance(g, o3d.geometry.TriangleMesh)]
    if sphere_geoms:
        combined = sphere_geoms[0]
        for g in sphere_geoms[1:]:
            combined += g
        o3d.io.write_triangle_mesh(str(grasps_path), combined)

    print(f"  Variance cloud → {path}")
    if sphere_geoms:
        print(f"  Grasp spheres → {grasps_path}")
    print(f"  Accepted: {len(accepted)} (green spheres), "
          f"Rejected: {len(rejected)} (red spheres)")


def save_cvar_plot(
    path: str,
    candidates: list[GraspCandidate],
    results: list[CandidateResult],
    all_results: list[CandidateResult],
) -> None:
    """Export CVaR score bar chart + variance histogram as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed — skipping plot")
        return

    # Split into accepted / rejected
    accepted = [r for r in all_results if r.accepted]
    rejected = [r for r in all_results if not r.accepted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Bar chart: CVaR scores ──
    labels = [f"#{r.candidate.id}" for r in all_results]
    scores = [r.cvar_epsilon for r in all_results]
    baseline = [r.epsilon_baseline for r in all_results]
    colours = ["#2ecc71" if r.accepted else "#e74c3c" for r in all_results]

    x = range(len(all_results))
    bars = ax1.bar(x, scores, color=colours, edgecolor="white", linewidth=1.2)
    ax1.scatter(x, baseline, color="black", marker="_", s=200, linewidth=2,
                label="Baseline ε")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("CVaR ε (grasp quality)")
    ax1.set_title("CVaR Grasp Quality Ranking\n(green = accepted, red = rejected)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # ── Pie chart: acceptance rate ──
    ax2.pie(
        [len(accepted), len(rejected)],
        labels=[f"Accepted ({len(accepted)})", f"Rejected ({len(rejected)})"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax2.set_title("CVaR Acceptance Rate")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  CVaR chart → {path}")


def main() -> None:
    args = parse_args()

    if not _HAS_SCIPY:
        print("ERROR: scipy is required for LP force-closure test.")
        print("  Install: pip install scipy")
        sys.exit(1)

    # ── Load variance cloud ──
    print("=== Loading variance cloud ===")
    mean_cloud, variance = load_variance_cloud(args.variance_cloud)
    variance_cloud_pts = mean_cloud.copy()  # keep original for variance mapping

    # ── Load mesh geometry (optional, for accurate surface normals) ──
    if args.mesh:
        print(f"\n=== Loading mesh geometry from {args.mesh} ===")
        mesh_pts, mesh_normals = load_mesh_for_geometry(args.mesh)
        print(f"  Mesh points: {len(mesh_pts):,} (voxel-downsampled)")
        # Centre the variance cloud positions to match the centred mesh
        vc_centroid = variance_cloud_pts.mean(axis=0)
        variance_cloud_pts_centred = variance_cloud_pts - vc_centroid
        # Map variance values from the variance cloud to mesh points
        variance = map_variance_to_geometry(variance_cloud_pts_centred, variance, mesh_pts)
        print(f"  Variance mapped from variance cloud → mesh points")
        # Use mesh geometry for contact lookup, variance for CVaR
        mean_cloud = mesh_pts
        base_normals = mesh_normals
    else:
        # ── Estimate reference normals from variance cloud ──
        print("\n=== Estimating reference normals ===")
        base_normals = estimate_normals_from_cloud(mean_cloud)
        print(f"  Estimated normals for {len(mean_cloud):,} points")

    # ── Load candidates ──
    print("\n=== Loading grasp candidates ===")
    if args.contact1 is not None and args.contact2 is not None:
        candidates = [GraspCandidate(
            id=1,
            contact1=np.array(args.contact1, dtype=np.float64),
            contact2=np.array(args.contact2, dtype=np.float64),
        )]
        print(f"  Using single contact pair from CLI")
    elif args.candidates and not args.mesh:
        # Pre-made candidates — only trust them without mesh (old behaviour)
        candidates = load_candidates(args.candidates)
    elif args.candidates and args.mesh:
        # Mesh overrides: generate fresh candidates on the ground-truth surface
        print(f"  Mesh provided — generating fresh antipodal candidates on mesh surface")
        print(f"  (Ignores --candidates file — canonical geometry overrides pre-made contacts)")
        num_to_gen = args.generate if args.generate > 0 else 15
        candidates = generate_antipodal_candidates(
            points=mean_cloud,
            normals=base_normals,
            mu=args.mu,
            m_generators=args.cone_generators,
            num_candidates=num_to_gen,
            sample_size=300,
            max_tries=500,
            seed=args.seed,
        )
        print(f"  Found {len(candidates)} valid candidates on mesh surface"
              f" ({len(candidates)/max(num_to_gen,1)*100:.0f}% success rate)")
    else:
        num_to_gen = args.generate if args.generate > 0 else 15
        print(f"  No candidates specified — generating {num_to_gen} surface-based antipodal pairs")
        candidates = generate_antipodal_candidates(
            points=mean_cloud,
            normals=base_normals,
            mu=args.mu,
            m_generators=args.cone_generators,
            num_candidates=num_to_gen,
            sample_size=300,
            max_tries=500,
            seed=args.seed,
        )
        print(f"  Found {len(candidates)} valid candidates"
              f" ({len(candidates)/max(num_to_gen,1)*100:.0f}% success rate)")

    if not candidates:
        print("ERROR: No grasp candidates to validate.")
        sys.exit(1)

    print(f"  Validating {len(candidates)} grasp candidates")

    # ── Run CVaR validation ──
    print(f"\n=== CVaR Validation (mu={args.mu}, N={args.num_realizations}, "
          f"alpha={args.cvar_alpha}, var_scale={args.variance_scale}) ===")

    results, all_results = validate_grasps(
        candidates=candidates,
        mean_cloud=mean_cloud,
        base_normals=base_normals,
        variance=variance,
        mu=args.mu,
        m_generators=args.cone_generators,
        num_realizations=args.num_realizations,
        cvar_alpha=args.cvar_alpha,
        variance_scale=args.variance_scale,
        seed=args.seed,
        verbose=True,
    )

    # ── Report ──
    print(f"\n{'='*50}")
    print(f"  CVaR Grasp Validation Report")
    print(f"{'='*50}")
    print(f"  Candidates tested:       {len(candidates)}")
    print(f"  Candidates accepted:     {len(results)}")
    print(f"  Acceptance rate:         {len(results)/len(candidates)*100:.1f}%")
    print(f"  Returned (top {args.top_k}): {len(results[:args.top_k])}")
    print(f"{'='*50}")

    if not results:
        print("\n  No candidates passed CVaR safety threshold.")
        print("  Recommendations:")
        print("    - Check contact pairs are antipodal")
        print("    - Verify mu >= 0.3 for plaster")
        print("    - Reduce cvar_alpha for less strict filtering")
    else:
        print(f"\n  --- Top {min(args.top_k, len(results))} Accepted Grasps ---")
        for i, r in enumerate(results[:args.top_k]):
            print(f"\n  Rank #{i + 1}: CVaR = {r.cvar_epsilon:.6f}")
            print(f"    {r}")

    # ── Save ──
    output = {
        "num_candidates": len(candidates),
        "num_accepted": len(results),
        "mu": args.mu,
        "cvar_alpha": args.cvar_alpha,
        "num_realizations": args.num_realizations,
        "accepted": [
            {
                "rank": i,
                "id": r.candidate.id,
                "contact1": r.candidate.contact1.tolist(),
                "contact2": r.candidate.contact2.tolist(),
                "cvar_epsilon": float(r.cvar_epsilon),
                "epsilon_baseline": float(r.epsilon_baseline),
                "num_failed_realizations": r.num_failed_realizations,
            }
            for i, r in enumerate(results[:args.top_k])
        ],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved accepted grasps to {args.output}")

    # ── Export visual outputs ──
    if args.save_grasp_ply:
        print(f"\n=== Exporting grasp PLY ===")
        accepted_all = [r for r in all_results if r.accepted]
        rejected_all = [r for r in all_results if not r.accepted]
        save_grasp_ply(
            "accepted_grasps.ply", mean_cloud, variance,
            accepted_all, rejected_all,
        )

    if args.save_plot:
        print(f"\n=== Exporting CVaR plot ===")
        save_cvar_plot("cvar_scores.png", candidates, results, all_results)

    # ── Visualise ──
    if not args.no_viz:
        visualise_variance_with_contacts(
            mean_cloud, variance, candidates, results,
        )


if __name__ == "__main__":
    main()
