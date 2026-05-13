"""
Epistemic Pose Covariance from Monte Carlo Dropout.

Computes the 6×6 covariance matrix over SE(3) pose estimates produced
by T stochastic forward passes of a GeoTransformer with MC Dropout
enabled at inference.  The covariance quantifies how much the estimated
rigid-body transformation varies due to learned model uncertainty.

=== SE(3) Lie Algebra Parameterisation ===

A rigid-body transform T ∈ SE(3) is parameterised by the 6-D twist
vector ξ = [v ; ω] ∈ se(3), where v ∈ R³ encodes translation and
ω ∈ R³ encodes rotation (axis-angle).  The exponential map recovers T:

    T(ξ) = exp([ξ]_∧)   where   [ξ]_∧ = [[ω]×  v ; 0ᵀ  0] ∈ R^{4×4}

The covariance Σ ∈ R^{6×6} is computed in this Lie algebra space
(not directly on the matrix elements), ensuring proper SE(3) geometry.

=== Per-Point Spatial Variance Projection ===

Given the 6×6 pose covariance Σ and a point p ∈ R³ in the source
frame, the transformed point p' = R p + t has 3×3 spatial covariance:

    Σ_p = J_p · Σ · J_pᵀ   ∈ R^{3×3}

where J_p = ∂p'/∂ξ = [I₃  |  −[p]×] ∈ R^{3×6} is the Jacobian of the
SE(3) action at p, and [p]× is the skew-symmetric cross-product matrix.

The per-point scalar variance (for colour-coding) is:
    σ²_p = trace(Σ_p)  =  trace(Σ_tt) + c(p, Σ_rr, Σ_tr)

where Σ_tt is the 3×3 translation block, Σ_rr is the 3×3 rotation block,
and c(p, ·) captures how rotation uncertainty amplifies with distance
from the origin:
    c(p, Σ_rr, Σ_tr) = pᵀ·[trace(Σ_rr)·I − Σ_rr]·p + 2·p·(Σ_tr · column)
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# SE(3) ←→ se(3) Lie algebra maps
# ---------------------------------------------------------------------------

def se3_log(T: np.ndarray) -> np.ndarray:
    r"""
    Logarithm map: SE(3) → se(3).

    Maps a 4×4 rigid-body transformation to its 6-D twist vector
    ξ = [v ; ω] ∈ R⁶ where ω is the axis-angle rotation vector
    and v is the corresponding translation component.

    For T = [R  t; 0ᵀ  1]:
        θ = arccos((tr(R) − 1)/2)          rotation angle
        ω = θ · (R − Rᵀ)^∨ / (2 sin θ)     axis-angle
        v = J⁻¹(ω) · t                      linear component

    where J(ω) is the left Jacobian of SO(3):

        J(ω) = I + (1−cos θ)/θ² · [ω]× + (θ−sin θ)/θ³ · [ω]×²

    Args:
        T: 4×4 SE(3) transformation matrix.

    Returns:
        Twist vector ξ = [v_x, v_y, v_z, ω_x, ω_y, ω_z], shape (6,).
    """
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    # Rotation angle
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Axis-angle ω
    if theta < 1e-12:
        omega = np.zeros(3, dtype=np.float64)
        v = t.copy()
    else:
        # Rodrigues: ω = θ · (R − Rᵀ)^∨ / (2 sin θ)
        omega_hat = (R - R.T) / (2.0 * np.sin(theta))
        omega_x = omega_hat[2, 1]
        omega_y = omega_hat[0, 2]
        omega_z = omega_hat[1, 0]
        omega = theta * np.array([omega_x, omega_y, omega_z])

        # Left Jacobian inverse
        J_inv = _so3_left_jacobian_inverse(omega)
        v = J_inv @ t

    return np.concatenate([v, omega])  # (6,)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    r"""
    Exponential map: se(3) → SE(3).

    Maps a 6-D twist vector ξ = [v ; ω] to a 4×4 rigid-body transform
    T = [R  t; 0ᵀ  1].

    Args:
        xi: Twist vector [v_x, v_y, v_z, ω_x, ω_y, ω_z], shape (6,).

    Returns:
        4×4 SE(3) transformation matrix.
    """
    v = xi[:3]
    omega = xi[3:]
    theta = np.linalg.norm(omega)

    if theta < 1e-12:
        R = np.eye(3, dtype=np.float64)
        t = v.copy()
    else:
        # Rodrigues rotation formula
        axis = omega / theta
        K = np.array([
            [0.0,      -axis[2],  axis[1]],
            [axis[2],   0.0,     -axis[0]],
            [-axis[1],  axis[0],   0.0],
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

        # Left Jacobian
        J = _so3_left_jacobian(omega)
        t = J @ v

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# SO(3) Jacobian helpers
# ---------------------------------------------------------------------------

def _so3_left_jacobian(omega: np.ndarray) -> np.ndarray:
    r"""
    Left Jacobian of SO(3):

        J(ω) = I + (1−cos θ)/θ² · [ω]× + (θ−sin θ)/θ³ · [ω]×²
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = omega / theta
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],   0.0],
    ])

    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * K + b * (K @ K)


def _so3_left_jacobian_inverse(omega: np.ndarray) -> np.ndarray:
    r"""
    Inverse of the left Jacobian of SO(3):

        J⁻¹(ω) = I − (1/2)·[ω]× + (1/θ² − (1+cos θ)/(2θ sin θ))·[ω]×²
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = omega / theta
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],   0.0],
    ])

    a = -0.5
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    b = (1.0 / theta**2) - (1.0 + cos_t) / (2.0 * theta * sin_t)
    return np.eye(3) + a * K + b * (K @ K)


# ---------------------------------------------------------------------------
# Pose covariance computation
# ---------------------------------------------------------------------------

def compute_pose_covariance(
    poses: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute 6×6 epistemic covariance from T SE(3) pose samples.

    Each pose T_k ∈ SE(3) is mapped to its twist ξ_k ∈ R⁶ via the
    logarithm map.  The covariance is computed in this Lie algebra:

        ξ̄   = (1/T) Σ_k ξ_k                        (mean twist)
        Σ    = (1/(T−1)) Σ_k (ξ_k − ξ̄)(ξ_k − ξ̄)ᵀ   (sample covariance)

    Σ encodes the full 6-DoF epistemic uncertainty:
        Σ[:3, :3]  → translation covariance    (m²)
        Σ[3:, 3:]  → rotation covariance       (rad²)
        Σ[:3, 3:]  → cross-covariance

    Also returns the mean SE(3) pose (via exponential map of ξ̄).

    Args:
        poses: List of T 4×4 SE(3) transformation matrices.

    Returns:
        (Sigma, T_mean)
        Sigma:   6×6 covariance matrix.
        T_mean:  4×4 mean SE(3) pose (exponential of mean twist).
    """
    T = len(poses)
    if T < 2:
        raise ValueError(f"Need ≥ 2 pose samples, got {T}")

    twists = np.array([se3_log(P) for P in poses], dtype=np.float64)  # (T, 6)
    mean_twist = twists.mean(axis=0)  # (6,)
    centered = twists - mean_twist[np.newaxis, :]  # (T, 6)

    Sigma = (centered.T @ centered) / (T - 1)  # (6, 6)

    T_mean = se3_exp(mean_twist)

    return Sigma, T_mean


def pose_covariance_statistics(Sigma: np.ndarray) -> dict:
    r"""
    Decompose the 6×6 pose covariance into interpretable scalar metrics.

    Returns:
        Dictionary with:
            sigma_t:  RMS translation uncertainty (metres).
            sigma_r:  RMS rotation uncertainty (degrees).
            trace:    Total uncertainty (trace of Σ).
    """
    Sigma_t = Sigma[:3, :3]   # translation block
    Sigma_r = Sigma[3:, 3:]   # rotation block

    # RMS translation uncertainty = sqrt(mean of eigenvalues)
    eig_t = np.linalg.eigvalsh(Sigma_t)
    eig_t = np.maximum(eig_t, 0.0)  # numerical guard
    sigma_t = float(np.sqrt(eig_t.mean()))

    # RMS rotation uncertainty = sqrt(mean of eigenvalues) in rad
    eig_r = np.linalg.eigvalsh(Sigma_r)
    eig_r = np.maximum(eig_r, 0.0)
    sigma_r_rad = float(np.sqrt(eig_r.mean()))
    sigma_r_deg = sigma_r_rad * (180.0 / math.pi)

    return {
        "sigma_t_m": sigma_t,
        "sigma_r_deg": sigma_r_deg,
        "trace": float(np.trace(Sigma)),
    }


# ---------------------------------------------------------------------------
# Per-point spatial variance projection
# ---------------------------------------------------------------------------

def project_spatial_variance(
    Sigma: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    r"""
    Project 6×6 pose covariance to per-point scalar spatial variance.

    For each point p_k, the 3×3 spatial covariance is:
        Σ_{p_k} = J_k · Σ · J_kᵀ   ∈ R^{3×3}

    where J_k = [I₃  |  −[p_k]×] ∈ R^{3×6} is the Jacobian of the
    SE(3) point action ∂(Rp + t)/∂ξ.

    The scalar variance per point is:
        σ²_k = trace(Σ_{p_k})
             = trace(Σ_tt)  +  p_kᵀ · M · p_k  +  2 · rᵀ(p_k)

    where:
        Σ_tt = Σ[:3, :3]                    (3×3 translation covariance)
        Σ_rr = Σ[3:, 3:]                    (3×3 rotation covariance)
        Σ_tr = Σ[:3, 3:]                    (3×3 cross-covariance)
        M    = trace(Σ_rr)·I₃ − Σ_rr        (3×3 quadratic form matrix)
        r(p) = Σ_trᵀ · p × ...  (cross-coupling term)

    Derived from:
        Σ_p = Σ_tt − Σ_tr·[p]×ᵀ − [p]×·Σ_trᵀ + [p]×·Σ_rr·[p]×ᵀ

        trace(Σ_p) = trace(Σ_tt) − 2·trace(Σ_tr·[p]×ᵀ) + trace([p]×·Σ_rr·[p]×ᵀ)

    Using the identities:
        trace([p]×·Σ_rr·[p]×ᵀ) = pᵀ·(trace(Σ_rr)·I − Σ_rr)·p
        trace(Σ_tr·[p]×ᵀ) = ...  (cross term)

    This is computed in fully vectorised form for efficiency.

    Args:
        Sigma:  6×6 pose covariance matrix.
        points: (N, 3) array of 3D point coordinates.

    Returns:
        Array of shape (N,) with per-point scalar variance σ²_k.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]

    # Extract blocks
    Sigma_tt = Sigma[:3, :3]    # (3, 3) translation
    Sigma_rr = Sigma[3:, 3:]    # (3, 3) rotation
    Sigma_tr = Sigma[:3, 3:]    # (3, 3) cross

    # Base translation variance (same for all points)
    base_var = float(np.trace(Sigma_tt))

    # Rotation contribution: pᵀ · M · p  where M = tr(Σ_rr)·I − Σ_rr
    tr_Sigma_rr = float(np.trace(Sigma_rr))
    M = tr_Sigma_rr * np.eye(3, dtype=np.float64) - Sigma_rr  # (3, 3)

    # Quadratic form: p · (M @ p) = Σ p_i M_ij p_j
    rot_var = np.einsum('ni,ij,nj->n', points, M, points)  # (N,)

    # Cross term: −2 · trace(Σ_tr · [p]×ᵀ)
    # trace(Σ_tr · [p]×ᵀ) = Σ_tr[1,2]*p[0] + Σ_tr[2,1]*p[0] + ...  messy
    # Better: compute directly per column of Σ_tr
    # For matrix A and skew-symmetric [p]×:
    # trace(A · [p]×ᵀ) =  p · (col vector from off-diagonal sums)
    # Specifically: [A[2,1]-A[1,2], A[0,2]-A[2,0], A[1,0]-A[0,1]] · [px,py,pz]
    a = np.array([
        Sigma_tr[2, 1] - Sigma_tr[1, 2],
        Sigma_tr[0, 2] - Sigma_tr[2, 0],
        Sigma_tr[1, 0] - Sigma_tr[0, 1],
    ])
    cross_var = -2.0 * (points @ a)  # (N,)

    # Total per-point scalar variance
    spatial_var = base_var + rot_var + cross_var  # (N,)

    # Numerical safety: clip negative values to 0
    return np.maximum(spatial_var, 0.0)


def project_spatial_covariance_full(
    Sigma: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Project 6×6 pose covariance to per-point 3×3 spatial covariance matrices.

    This is the full version — each point gets its own 3×3 covariance.

    Args:
        Sigma:  6×6 pose covariance.
        points: (N, 3) points.

    Returns:
        Array of shape (N, 3, 3) with per-point spatial covariances.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]

    Sigma_tt = Sigma[:3, :3]
    Sigma_rr = Sigma[3:, 3:]
    Sigma_tr = Sigma[:3, 3:]

    J = np.zeros((N, 3, 6), dtype=np.float64)
    J[:, :, :3] = np.eye(3)  # ∂p'/∂t = I

    # ∂p'/∂ω = −[p]×  (skew-symmetric)
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    J[:, 0, 4] =  pz   # skew[0,1] = -pz, but J = -skew → J[0,4]=+pz
    J[:, 0, 5] = -py
    J[:, 1, 3] = -pz
    J[:, 1, 5] =  px
    J[:, 2, 3] =  py
    J[:, 2, 4] = -px

    # Σ_{p_k} = J_k Σ J_kᵀ
    Sigma_p = J @ Sigma[np.newaxis, :, :] @ J.transpose(0, 2, 1)  # (N, 3, 3)
    return Sigma_p


# ---------------------------------------------------------------------------
# Covariance-to-RGB colour mapping
# ---------------------------------------------------------------------------

def variance_to_rgb(
    spatial_variance: np.ndarray,
    *,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """
    Map per-point scalar variance to RGB colours using a diverging colormap.

    Blue (low variance) → White → Red (high variance).

    Args:
        spatial_variance: (N,) scalar variance values.
        clip_percentile:  Upper percentile to clip at.

    Returns:
        RGB colours, shape (N, 3), values in [0, 1].
    """
    v = spatial_variance.copy()

    v_max = float(np.percentile(v, clip_percentile))
    if v_max <= 0.0:
        v_max = 1.0
    v = np.clip(v, 0.0, v_max)
    v_norm = v / v_max

    r = np.where(v_norm < 0.5, 2.0 * v_norm, 1.0)
    g = np.where(v_norm < 0.5, 2.0 * v_norm, 2.0 * (1.0 - v_norm))
    b = np.where(v_norm < 0.5, 1.0, 2.0 * (1.0 - v_norm))

    colours = np.stack([r, g, b], axis=-1)
    return np.clip(colours, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pose summary
# ---------------------------------------------------------------------------

def print_covariance_report(
    Sigma: np.ndarray,
    T_mean: np.ndarray,
    spatial_var: np.ndarray,
) -> None:
    """Print a human-readable covariance analysis report."""
    stats = pose_covariance_statistics(Sigma)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    idx = np.argsort(-eigvals)

    lines = [
        "=" * 58,
        "  Epistemic Pose Covariance Report (MC Dropout)",
        "=" * 58,
        f"  RMS translation uncertainty:  {stats['sigma_t_m']:.6f} m",
        f"  RMS rotation uncertainty:    {stats['sigma_r_deg']:.3f} deg",
        f"  Trace(Σ):                    {stats['trace']:.6e}",
        "",
        "  ── Mean Pose (SE(3)) ──",
    ]
    for row in T_mean:
        lines.append(f"    [{row[0]:+12.6f}  {row[1]:+12.6f}  {row[2]:+12.6f}  {row[3]:+12.6f}]")
    lines.append("")

    lines.append("  ── Covariance Eigenvalues ──")
    for i, k in enumerate(idx):
        pct = 100.0 * eigvals[k] / eigvals.sum()
        lines.append(f"    λ_{i+1} = {eigvals[k]:.6e}  ({pct:5.1f}%)")
    lines.append("")

    lines.append("  ── Per-Point Spatial Variance ──")
    lines.append(f"    Mean σ²_p:   {spatial_var.mean():.6e} m²")
    lines.append(f"    Max σ²_p:    {spatial_var.max():.6e} m²")
    lines.append(f"    95th %ile:   {np.percentile(spatial_var, 95):.6e} m²")
    lines.append(f"    99th %ile:   {np.percentile(spatial_var, 99):.6e} m²")
    lines.append("=" * 58)

    print("\n".join(lines))
