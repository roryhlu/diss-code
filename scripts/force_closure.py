#!/usr/bin/env python3
"""
Force-Closure Analysis for Two-Finger Grasps on Fragile 3D Fragments.

Implements the Grasp Wrench Space (GWS) computation and Force-Closure
verification from Module 3 of the RePAIR dissertation:

  1. Load a 3D mesh (PLY/OBJ/STL) via trimesh.
  2. Extract surface normals at two user-specified contact points.
  3. Construct polyhedral friction cones (μ = 0.5 default, 8 generators).
  4. Build the 6-D wrench matrices: [f ; c × f] per contact generator.
  5. Assemble the GWS as the convex hull of the Minkowski sum of wrench cones.
  6. Verify Force-Closure: does the origin 0 ∈ ℝ⁶ lie strictly inside the
     convex hull?  Solved via Linear Programming (HiGHS backend).
  7. Compute grasp quality ε — the radius of the largest ball around 0
     inside the GWS (epsilon metric).

=== Mathematical Formulation ===

Friction Cone (Coulomb, point-contact-with-friction):
    C = { f ∈ ℝ³ : f·n̂ ≥ 0,  ‖f - (f·n̂)n̂‖ ≤ μ (f·n̂) }
  Half-angle: α = arctan(μ)
  Polyhedral approximation: m generators uniformly on cone surface.

Contact Wrench Cone at c_i:
    W_i = { [f ; c_i × f] ∈ ℝ⁶ : f ∈ C_i }  ⊂ ℝ⁶

Grasp Wrench Space (Minkowski sum):
    GWS = conv( W₁ ∪ W₂ ∪ … ∪ W_k )  — for k contacts.

Force-Closure:
    0 ∈ int(GWS)
  i.e. there exist α_j > 0 with Σ α_j w_j = 0, Σ α_j = 1.

Grasp Quality (epsilon metric):
    ε = max { r ≥ 0 : B(0, r) ⊆ GWS }
  Largest ball around origin fully contained in the GWS.  Used as
  the cost-to-reject in the CVaR filter (see AGENTS.md §3).

Usage:
    python scripts/force_closure.py fragment.stl \\
        --contact1 0.023 -0.015 0.041 \\
        --contact2 -0.019 0.021 -0.038 \\
        --mu 0.5 --cone-generators 8 --quality
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

try:
    from scipy.optimize import linprog  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Contact:
    """Single contact point with friction cone and wrench matrix."""

    idx: int
    """Contact index (1 or 2)."""
    position: np.ndarray
    """Contact position in world frame, shape (3,)."""
    normal: np.ndarray
    """Unit INWARD surface normal (pointing into the object), shape (3).
    This is the negated trimesh vertex normal — Coulomb friction requires
    compressive forces f satisfying f·n̂ ≥ 0, i.e. pointing INTO the body."""
    generators: np.ndarray
    """Friction cone generators, shape (m, 3).  Each row is a force vector f_k."""
    W: np.ndarray
    """Contact wrench matrix, shape (6, m).  Columns are [f_k ; c × f_k]."""


@dataclass
class GraspResult:
    """Force-closure analysis result for a two-finger grasp."""

    contacts: tuple[Contact, Contact]
    mu: float
    cone_angle_deg: float
    antipodal_satisfied: bool
    antipodal_score_1: float
    antipodal_score_2: float
    force_closure: bool
    epsilon: float
    lp_status: str
    lp_message: str

    def __repr__(self) -> str:
        c1, c2 = self.contacts
        lines = [
            "=" * 50,
            "  Force-Closure Analysis",
            "=" * 50,
            f"  Friction coefficient:  μ = {self.mu:.3f}  (α = {self.cone_angle_deg:.2f}°)",
            f"  Generators per contact: {c1.generators.shape[0]}",
            "",
            f"  Contact 1:  ({c1.position[0]:+.4f}, {c1.position[1]:+.4f}, {c1.position[2]:+.4f})",
            f"    normal =  ({c1.normal[0]:+.4f}, {c1.normal[1]:+.4f}, {c1.normal[2]:+.4f})",
            "",
            f"  Contact 2:  ({c2.position[0]:+.4f}, {c2.position[1]:+.4f}, {c2.position[2]:+.4f})",
            f"    normal =  ({c2.normal[0]:+.4f}, {c2.normal[1]:+.4f}, {c2.normal[2]:+.4f})",
            "",
            "  ── Antipodal Check ──",
            f"    Contact 1:   d̂·n̂₁ = {self.antipodal_score_1:.4f}  ≥ cos(α) = {np.cos(np.arctan(self.mu)):.4f}  {'✓' if self.antipodal_satisfied else '✗ FAIL'}",
            f"    Contact 2:  −d̂·n̂₂ = {self.antipodal_score_2:.4f}  ≥ cos(α) = {np.cos(np.arctan(self.mu)):.4f}  {'✓' if self.antipodal_satisfied else '✗ FAIL'}",
            f"    Antipodal:  {'SATISFIED' if self.antipodal_satisfied else 'NOT SATISFIED'}",
            "",
            "  ── LP Force-Closure Test ──",
            f"    Result:  {'FORCE-CLOSURE ✓' if self.force_closure else 'NOT FORCE-CLOSURE ✗'}",
            f"    LP status:  {self.lp_status}",
            f"    LP message: {self.lp_message}",
        ]
        if self.force_closure:
            lines.append(f"    Grasp quality (ε):  {self.epsilon:.6f}")
        else:
            lines.append(f"    Grasp quality (ε):  {self.epsilon:.6f}  (0 = no closure)")
        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------


def load_mesh(file_path: str) -> trimesh.Trimesh:
    """Load a 3D mesh from PLY, OBJ, STL, or other trimesh-supported format."""
    mesh = trimesh.load(file_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"trimesh loaded a {type(mesh).__name__}; expected Trimesh")
    if len(mesh.vertices) == 0:
        raise ValueError(f"No vertices in mesh '{file_path}'")
    print(f"  Loaded {Path(file_path).name}: "
          f"{len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh


def find_nearest_vertex(mesh: trimesh.Trimesh, point: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Find the nearest mesh vertex to a given point.

    Returns:
        (vertex_index, vertex_position)
    """
    dists = np.linalg.norm(mesh.vertices - point, axis=1)
    idx = int(np.argmin(dists))
    return idx, mesh.vertices[idx].copy()


def get_vertex_normal(mesh: trimesh.Trimesh, vertex_idx: int) -> np.ndarray:
    """
    Extract the unit surface normal at a mesh vertex.

    Uses area-weighted average of incident face normals for smooth meshes
    (trimesh vertex_normals).  Falls back to per-face normal on non-smooth.
    """
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        n = mesh.vertex_normals[vertex_idx].copy()
    else:
        # Per-face normal: find first incident face
        incident = np.any(mesh.faces == vertex_idx, axis=1)
        if not incident.any():
            return np.array([0.0, 0.0, 1.0])
        n = mesh.face_normals[np.argmax(incident)].copy()

    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


# ---------------------------------------------------------------------------
# Friction cone (polyhedral)
# ---------------------------------------------------------------------------


def orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build orthonormal basis {u, v} for the plane orthogonal to n.

    Uses the stable Householder-like construction: pick the axis
    least aligned with n, form u = cross(n, axis), normalise,
    then v = cross(n, u).
    """
    n = n / np.linalg.norm(n)
    if abs(n[0]) < 0.9:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def friction_cone_generators(
    normal: np.ndarray,
    mu: float,
    m: int = 8,
) -> np.ndarray:
    r"""
    Construct polyhedral approximation of Coulomb friction cone.

    Generates m unit-length force vectors uniformly distributed on the
    cone surface:

        f_k = cos(α) · n̂ + sin(α) · (cos θ_k · u + sin θ_k · v)

    where  α = arctan(μ)  is the cone half-angle,
           θ_k = 2πk / m,
           {u, v} ⊥ n̂ form an orthonormal basis.

    Args:
        normal:  Unit surface normal (pointing outward).
        mu:      Friction coefficient (≥ 0).
        m:       Number of generators (polyhedral edges).

    Returns:
        Array of shape (m, 3), each row a unit force vector.
    """
    if mu <= 0.0:
        return normal.reshape(1, 3)

    n = normal / np.linalg.norm(normal)
    u, v = orthonormal_basis(n)
    alpha = np.arctan(mu)

    theta = 2.0 * np.pi * np.arange(m) / m

    # (m, 3)
    gens = (
        np.cos(alpha) * n[np.newaxis, :]
        + np.sin(alpha) * (
            np.cos(theta)[:, np.newaxis] * u[np.newaxis, :]
            + np.sin(theta)[:, np.newaxis] * v[np.newaxis, :]
        )
    )
    # Normalise each generator to unit length
    gens /= np.linalg.norm(gens, axis=1, keepdims=True)
    return gens


# ---------------------------------------------------------------------------
# Wrench matrix assembly
# ---------------------------------------------------------------------------


def build_contact_wrench(
    position: np.ndarray,
    generators: np.ndarray,
) -> np.ndarray:
    r"""
    Build the contact wrench matrix for one finger.

    Each generator  f_k  produces a wrench  w_k ∈ ℝ⁶:

        w_k = [ f_k ;  position × f_k ]

    Convention: the cross product uses the 'skew' formulation
    τ = c × f  where c is the contact point coordinates.

    Args:
        position:   Contact point c ∈ ℝ³.
        generators: Friction cone generators, shape (m, 3).

    Returns:
        Wrench matrix of shape (6, m).  Column j = [f_j; c × f_j].
    """
    m = generators.shape[0]
    F = generators.T  # (3, m)

    # Torque: τ = c × f  →  τ_k = cross(position, F[:,k])
    # Vectorised:  τ = skew(c) @ F   where skew(c) = cross-product matrix
    cx, cy, cz = position
    skew_c = np.array([
        [0.0, -cz,  cy],
        [cz,  0.0, -cx],
        [-cy, cx,  0.0],
    ])
    tau = skew_c @ F  # (3, m)

    W = np.vstack([F, tau])  # (6, m)
    return W


def combined_wrench_matrix(contacts: tuple[Contact, Contact]) -> np.ndarray:
    """Horizontally stack wrench matrices: W = [W₁ | W₂] ∈ ℝ^{6 × 2m}."""
    return np.hstack([contacts[0].W, contacts[1].W])


# ---------------------------------------------------------------------------
# Antipodal check (analytical two-finger condition)
# ---------------------------------------------------------------------------


def check_antipodal(
    c1: np.ndarray,
    n1: np.ndarray,
    c2: np.ndarray,
    n2: np.ndarray,
    mu: float,
) -> tuple[bool, float, float]:
    r"""
    Analytical two-finger antipodal condition.

    Let  d = c₂ − c₁  be the inter-contact axis (unit: d̂).
    The normals n₁, n₂ are *inward* normals (pointing into the object),
    opposite to the outward normals returned by trimesh.

    For a compressive grasp to achieve force-closure, the inter-contact
    axis must lie inside *both* friction cones:

        Contact 1 pushes toward c₂:
            force ∥ d̂  →  d̂ · n̂₁ ≥ cos(α)

        Contact 2 pushes toward c₁:
            force ∥ −d̂  →  −d̂ · n̂₂ ≥ cos(α)

    where α = arctan(μ). This means the angle between the line of
    action and each inward normal is less than the cone half-angle.

    Returns:
        (satisfied, score_1, score_2)
        score_1 = d̂·n̂₁,  score_2 = −d̂·n̂₂;  satisfied if both ≥ cosα.
    """
    d = c2 - c1
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        return False, 0.0, 0.0

    d_hat = d / d_norm
    cos_alpha = np.cos(np.arctan(mu))

    score_1 = float(d_hat @ n1)     # d̂·n̂₁  — must be ≥ cos(α)
    score_2 = float(-d_hat @ n2)    # −d̂·n̂₂ — must be ≥ cos(α)

    ok = score_1 >= cos_alpha - 1e-9 and score_2 >= cos_alpha - 1e-9
    return ok, score_1, score_2


# ---------------------------------------------------------------------------
# Force-Closure via Linear Programming
# ---------------------------------------------------------------------------


def test_force_closure_lp(
    W: np.ndarray,
) -> tuple[bool, float, str, str]:
    r"""
    Test Force-Closure by solving the LP:

        max   ε
        s.t.  W α = 0                  (6 equality constraints)
              1ᵀ α = 1                 (normalisation)
              α_j ≥ ε,  ∀j           (strict positivity margin)

    If the optimal ε > 0, the origin is strictly in the interior
    of the convex hull of the wrench columns → Force-Closure.

    ε is also the grasp quality metric: the maximum uniform margin
    achievable across all cone generators.

    Args:
        W: Combined wrench matrix, shape (6, N) where N = 2m.

    Returns:
        (force_closure, epsilon, lp_status, lp_message)
    """
    if not _HAS_SCIPY:
        print("  WARNING: scipy not installed — LP test disabled. "
              "Install: pip install scipy")
        return False, 0.0, "scipy_missing", "scipy.optimize not available"

    N = W.shape[1]  # number of variables (wrench columns)

    # Variables: [α_0, α_1, ..., α_{N-1}, ε]
    # We maximise ε, so c = [0,...,0, -1]  (linprog minimises)
    c = np.zeros(N + 1)
    c[-1] = -1.0  # maximise ε

    # Equality: Wα = 0  →  A_eq x = [0, ..., 0]
    A_eq = np.zeros((7, N + 1))
    A_eq[:6, :N] = W         # W α = 0
    A_eq[6, :N] = 1.0        # Σ α = 1
    # ε column is zero in all equality rows

    b_eq = np.zeros(7)
    b_eq[6] = 1.0

    # Inequality: α_j − ε ≥ 0  →  ε − α_j ≤ 0
    # Written as A_ub x ≤ b_ub:
    #   For each j:  −α_j + ε ≤ 0
    A_ub = np.zeros((N, N + 1))
    for j in range(N):
        A_ub[j, j] = -1.0    # −α_j
        A_ub[j, -1] = 1.0    # +ε
    b_ub = np.zeros(N)

    # Bounds: α_j ≥ 0, ε free
    bounds = [(0.0, None)] * N + [(None, None)]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.success and result.x[-1] > 1e-9:
        force_closure = True
        epsilon = float(result.x[-1])
    else:
        force_closure = False
        epsilon = 0.0

    status = str(result.status)
    message = result.message

    return force_closure, epsilon, status, message


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualise(
    mesh: trimesh.Trimesh,
    result: GraspResult,
) -> None:
    """
    Render the mesh with contact points and friction cone generators.

    Uses trimesh Scene for mesh rendering supplemented by path primitives
    for the friction cone vectors.
    """
    scene = trimesh.Scene()

    # Mesh — semi-transparent
    mesh_geom = mesh.copy()
    mesh_geom.visual.face_colors = [120, 180, 220, 120]  # semi-transparent blue
    scene.add_geometry(mesh_geom, geom_name="fragment")

    c1, c2 = result.contacts

    def _cone_arrows(origin: np.ndarray, gens: np.ndarray, colour: tuple[int, int, int]):
        """Add arrow lines for each cone generator."""
        for g in gens:
            end = origin + g * 0.005  # scale for visibility
            segment = np.array([origin, end])
            path = trimesh.load_path(segment, file_type="misc")
            path.colors = np.array([colour] * len(path.entities))
            scene.add_geometry(path)

    # Contact spheres
    s1 = trimesh.creation.icosphere(subdivisions=3, radius=0.0015)
    s1.apply_translation(c1.position)
    s1.visual.face_colors = [220, 60, 60, 255]  # red
    scene.add_geometry(s1, geom_name="contact1")

    s2 = trimesh.creation.icosphere(subdivisions=3, radius=0.0015)
    s2.apply_translation(c2.position)
    s2.visual.face_colors = [60, 220, 60, 255]  # green
    scene.add_geometry(s2, geom_name="contact2")

    # Friction cone generators as line segments
    _cone_arrows(c1.position, c1.generators, (220, 60, 60))
    _cone_arrows(c2.position, c2.generators, (60, 220, 60))

    # Inter-contact axis
    axis_seg = np.array([c1.position, c2.position])
    axis_path = trimesh.load_path(axis_seg, file_type="misc")
    axis_path.colors = np.array([(200, 200, 50)] * len(axis_path.entities))
    scene.add_geometry(axis_path)

    # Title
    fc_text = "FORCE-CLOSURE" if result.force_closure else "NO FORCE-CLOSURE"
    print(f"\n  Visualising: {fc_text}")

    scene.show()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def analyse_grasp(
    mesh: trimesh.Trimesh,
    c1_coords: np.ndarray,
    c2_coords: np.ndarray,
    mu: float = 0.5,
    m_generators: int = 8,
    compute_quality: bool = True,
) -> GraspResult:
    """
    Full two-finger Force-Closure pipeline.

    1. Find nearest vertices + extract normals.
    2. Build friction cones.
    3. Compute contact wrenches.
    4. Check antipodal condition.
    5. LP Force-Closure test + quality metric.
    """
    # ── Step 1: find contact vertices & normals ──
    idx1, pos1 = find_nearest_vertex(mesh, c1_coords)
    idx2, pos2 = find_nearest_vertex(mesh, c2_coords)
    n1_out = get_vertex_normal(mesh, idx1)
    n2_out = get_vertex_normal(mesh, idx2)

    # trimesh returns OUTWARD normals — invert to INWARD for grasping math.
    # Coulomb friction cone: forces must point INTO the object (f · n̂_in ≥ 0).
    n1 = -n1_out
    n2 = -n2_out

    dist1 = np.linalg.norm(pos1 - c1_coords)
    dist2 = np.linalg.norm(pos2 - c2_coords)
    if dist1 > 0.001 or dist2 > 0.001:
        print(f"  Note: snapped contact points to nearest vertices "
              f"(distance: {dist1:.4f} m, {dist2:.4f} m)")

    # ── Step 2: friction cones (using INWARD normals) ──
    gens1 = friction_cone_generators(n1, mu, m_generators)
    gens2 = friction_cone_generators(n2, mu, m_generators)

    # ── Step 3: wrench matrices ──
    W1 = build_contact_wrench(pos1, gens1)
    W2 = build_contact_wrench(pos2, gens2)

    contact1 = Contact(idx=1, position=pos1, normal=n1, generators=gens1, W=W1)
    contact2 = Contact(idx=2, position=pos2, normal=n2, generators=gens2, W=W2)

    # ── Step 4: antipodal check ──
    antipodal_ok, score1, score2 = check_antipodal(pos1, n1, pos2, n2, mu)

    # ── Step 5: LP Force-Closure ──
    W_full = combined_wrench_matrix((contact1, contact2))

    if compute_quality:
        fc, eps, lp_stat, lp_msg = test_force_closure_lp(W_full)
    else:
        fc, eps, lp_stat, lp_msg = False, 0.0, "skipped", "quality computation disabled"

    return GraspResult(
        contacts=(contact1, contact2),
        mu=mu,
        cone_angle_deg=float(np.rad2deg(np.arctan(mu))),
        antipodal_satisfied=antipodal_ok,
        antipodal_score_1=score1,
        antipodal_score_2=score2,
        force_closure=fc,
        epsilon=eps,
        lp_status=lp_stat,
        lp_message=str(lp_msg),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_vec3(arg: str) -> np.ndarray:
    """Parse three space-separated floats: 'x y z'."""
    parts = arg.split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 floats, got {len(parts)}: '{arg}'")
    return np.array([float(x) for x in parts], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-finger Force-Closure analysis for RePAIR fragments",
    )
    p.add_argument(
        "mesh",
        type=str,
        help="3D mesh file (PLY, OBJ, STL)",
    )
    p.add_argument(
        "--contact1",
        type=_parse_vec3,
        required=True,
        metavar="'X Y Z'",
        help="First contact point coordinates (quoted, space-separated)",
    )
    p.add_argument(
        "--contact2",
        type=_parse_vec3,
        required=True,
        metavar="'X Y Z'",
        help="Second contact point coordinates (quoted, space-separated)",
    )
    p.add_argument(
        "--mu",
        type=float,
        default=0.5,
        help="Friction coefficient (default: 0.5)",
    )
    p.add_argument(
        "--cone-generators",
        type=int,
        default=8,
        help="Number of polyhedral friction cone generators (default: 8)",
    )
    p.add_argument(
        "--quality",
        action="store_true",
        help="Compute epsilon grasp quality metric via LP",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualisation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load mesh ──
    print("=== Loading mesh ===")
    mesh = load_mesh(args.mesh)

    # ── Run analysis ──
    print(f"\n=== Force-Closure Analysis ===")
    print(f"  Contact 1:  {args.contact1}")
    print(f"  Contact 2:  {args.contact2}")
    print(f"  Friction μ:  {args.mu}")
    print(f"  Generators:  {args.cone_generators}")

    result = analyse_grasp(
        mesh=mesh,
        c1_coords=args.contact1,
        c2_coords=args.contact2,
        mu=args.mu,
        m_generators=args.cone_generators,
        compute_quality=args.quality,
    )

    # ── Print result ──
    print(f"\n{result}")

    # ── Visualise ──
    if not args.no_viz:
        visualise(mesh, result)


if __name__ == "__main__":
    main()
