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

=== Minkowski Sum GWS ===

The Grasp Wrench Space for a two-finger polyhedral friction cone grasp
is the convex hull of the Minkowski sum of individual contact wrench cones:

    W = conv{ w_{1,j} + w_{2,k} : j,k = 1..m }  in R^6

where w_{i,j} = [f_{i,j} ; c_i x f_{i,j}] and f_{i,j} are the m generators
of the Coulomb friction cone at contact i.

Force-closure: 0 in int(W) iff exists alpha_{i,j} > 0 s.t. sum W_i alpha_i = 0.

=== CVaR at 5% - Why It Is Safer Than Expected-Value Planning ===

A mean-filter would accept a grasp with epsilon = 0.1 in 95% of realizations
but epsilon = 0 in 5%. CVaR_{5%} correctly rejects it at CVaR_{5%} = 0.
This protects against fracture-edge geometries where the fragment
can shatter under grasping force on the worst-5% of hidden variations.

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
# Variance cloud I/O
# ---------------------------------------------------------------------------

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
    points: np.ndarray, radius: float = 0.01, max_nn: int = 30,
) -> np.ndarray:
    """Estimate surface normals using Open3D KD-tree."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return np.asarray(pcd.normals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometric realization sampling
# ---------------------------------------------------------------------------

def sample_realizations(
    mean: np.ndarray,
    variance: np.ndarray,
    N: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""
    Generate N geometric realizations by isotropic per-point sampling.

    For each point i: p_i^{(k)} ~ N(mu_i, sigma2_i * I_3)
    Returns: (N, M, 3) array.
    """
    rng = np.random.RandomState(seed)
    M = mean.shape[0]
    std = np.sqrt(np.maximum(variance, 0.0))
    noise = rng.randn(N, M, 3) * std[np.newaxis, :, np.newaxis]
    return mean[np.newaxis, :, :] + noise


# ---------------------------------------------------------------------------
# Single-realization force-closure evaluation
# ---------------------------------------------------------------------------

def evaluate_realization(
    points: np.ndarray,
    normals: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    mu: float,
    m_generators: int,
) -> tuple[bool, float]:
    """
    Test force-closure on a single geometric realization.

    Returns: (force_closure, epsilon)
    """
    dists1 = np.linalg.norm(points - c1, axis=1)
    dists2 = np.linalg.norm(points - c2, axis=1)
    idx1 = int(np.argmin(dists1))
    idx2 = int(np.argmin(dists2))

    p1 = points[idx1]
    p2 = points[idx2]

    n1_in = -normals[idx1]
    n2_in = -normals[idx2]

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
        seed:             Random seed for reproducibility.
        verbose:          Print per-candidate diagnostics.

    Returns:
        List of CandidateResult, only accepted candidates (CVaR > 0).
    """
    results = []
    t_start = time.perf_counter()

    for cand in candidates:
        if verbose:
            print(f"\n  --- Candidate #{cand.id} ---")

        # Step 1: Baseline force-closure
        idx1 = int(np.argmin(np.linalg.norm(mean_cloud - cand.contact1, axis=1)))
        idx2 = int(np.argmin(np.linalg.norm(mean_cloud - cand.contact2, axis=1)))
        p1_ref = mean_cloud[idx1]
        p2_ref = mean_cloud[idx2]
        n1_in = -base_normals[idx1]
        n2_in = -base_normals[idx2]

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

        # Step 2: Sample realizations using actual variance cloud
        realizations = sample_realizations(
            mean_cloud, variance,
            N=num_realizations, seed=seed + cand.id if seed else None,
        )

        # Step 3: Evaluate each realization
        eps_values = np.zeros(num_realizations)
        for k in range(num_realizations):
            r_normals = estimate_normals_from_cloud(realizations[k])
            _, eps_k = evaluate_realization(
                realizations[k], r_normals,
                cand.contact1, cand.contact2, mu, m_generators,
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
    return results


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

    if isinstance(source, str) and Path(source).exists():
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _HAS_SCIPY:
        print("ERROR: scipy is required for LP force-closure test.")
        print("  Install: pip install scipy")
        sys.exit(1)

    # ── Load variance cloud ──
    print("=== Loading variance cloud ===")
    mean_cloud, variance = load_variance_cloud(args.variance_cloud)

    # ── Estimate reference normals ──
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
    elif args.candidates:
        candidates = load_candidates(args.candidates)
    else:
        print("  No candidates specified - generating random ones for testing")
        bbox_min = mean_cloud.min(axis=0)
        bbox_max = mean_cloud.max(axis=0)
        bounds = np.array([bbox_min, bbox_max])
        candidates = load_candidates(
            "", num_generate=args.generate if args.generate > 0 else 15,
            cloud_bounds=bounds,
        )

    if not candidates:
        print("ERROR: No grasp candidates to validate.")
        sys.exit(1)

    print(f"  Validating {len(candidates)} grasp candidates")

    # ── Run CVaR validation ──
    print(f"\n=== CVaR Validation (mu={args.mu}, N={args.num_realizations}, "
          f"alpha={args.cvar_alpha}) ===")

    results = validate_grasps(
        candidates=candidates,
        mean_cloud=mean_cloud,
        base_normals=base_normals,
        variance=variance,
        mu=args.mu,
        m_generators=args.cone_generators,
        num_realizations=args.num_realizations,
        cvar_alpha=args.cvar_alpha,
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

    # ── Visualise ──
    if not args.no_viz:
        visualise_variance_with_contacts(
            mean_cloud, variance, candidates, results,
        )


if __name__ == "__main__":
    main()
