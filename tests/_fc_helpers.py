"""
Pure-math helpers extracted from scripts/force_closure.py.

Importing force_closure.py triggers Open3D and trimesh imports.
This module provides standalone versions for unit testing.
"""

from typing import Optional

import numpy as np
from scipy.optimize import linprog


def orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct orthonormal basis {u, v} orthogonal to normal n."""
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
    """Polyhedral approximation of Coulomb friction cone."""
    if mu <= 0.0:
        n = normal / np.linalg.norm(normal)
        return n.reshape(1, 3)

    alpha = np.arctan(mu)
    n = normal / np.linalg.norm(normal)
    u, v = orthonormal_basis(n)
    thetas = np.linspace(0, 2 * np.pi, m, endpoint=False)

    generators = np.zeros((m, 3))
    for k in range(m):
        generators[k] = (
            np.cos(alpha) * n
            + np.sin(alpha) * (np.cos(thetas[k]) * u + np.sin(thetas[k]) * v)
        )
    return generators


def build_contact_wrench(
    position: np.ndarray,
    generators: np.ndarray,
) -> np.ndarray:
    """Build 6×m wrench matrix from contact position and force generators."""
    m = generators.shape[0]
    forces = generators.T  # (3, m)
    px, py, pz = position
    skew = np.array([
        [0.0, -pz, py],
        [pz, 0.0, -px],
        [-py, px, 0.0],
    ])
    torques = skew @ forces  # (3, m)
    return np.vstack([forces, torques]).astype(np.float64)  # (6, m)


def check_antipodal(
    c1: np.ndarray,
    n1: np.ndarray,
    c2: np.ndarray,
    n2: np.ndarray,
    mu: float,
) -> tuple[bool, float, float]:
    """Check two-finger antipodal grasp condition."""
    d = c2 - c1
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        return False, 0.0, 0.0
    d_hat = d / d_norm

    alpha = np.arctan(mu)
    cos_alpha = np.cos(alpha)

    s1 = float(np.dot(d_hat, n1))
    s2 = float(np.dot(-d_hat, n2))
    tol = 1e-9
    return (s1 >= cos_alpha - tol and s2 >= cos_alpha - tol), s1, s2


def solve_force_closure_lp(
    W: np.ndarray,
) -> tuple[bool, float, str, str]:
    """Test force-closure via Linear Programming (HiGHS)."""
    n_cols = W.shape[1]
    if n_cols < 1:
        return False, 0.0, "no_columns", ""

    # Variables: α₁ … α_n, ε
    n_vars = n_cols + 1

    # Objective: maximise ε
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimise -ε → maximise ε

    # Equality: W @ α = 0  (6 constraints)
    A_eq = np.zeros((6, n_vars))
    A_eq[:, :n_cols] = W

    # Constraint: Σα = 1
    A_sum = np.zeros((1, n_vars))
    A_sum[0, :n_cols] = 1.0

    A_eq = np.vstack([A_eq, A_sum])
    b_eq = np.zeros(7)
    b_eq[6] = 1.0

    # Inequality: α_j ≥ ε  →  α_j − ε ≥ 0
    A_ub = np.zeros((n_cols, n_vars))
    for j in range(n_cols):
        A_ub[j, j] = -1.0
        A_ub[j, -1] = 1.0
    b_ub = np.zeros(n_cols)

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=[(None, None)] * n_vars,
        method="highs",
    )

    if not res.success:
        return False, 0.0, res.message, str(res.status)

    eps = float(res.x[-1]) if res.x[-1] is not None else 0.0
    return eps > 1e-9, max(0.0, eps), res.message, str(res.status)
