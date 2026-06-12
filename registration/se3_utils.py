"""
SE(3) Utility Functions.

Provides point transformation, composition, inversion, and extraction
utilities for rigid body transforms in the Special Euclidean group SE(3).

All functions support batched (B, ...) and unbatched tensor inputs.
"""

from __future__ import annotations

import numpy as np
import torch


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) transform to points:  P' = R @ P + t.

    Args:
        T:      4x4 or (B,4,4) transformation matrix.
        points: (N,3) or (B,N,3) point cloud(s).

    Returns:
        Transformed points, same shape as input.
    """
    batched = T.dim() == 3
    if not batched:
        T = T.unsqueeze(0)
        points = points.unsqueeze(0)

    R = T[:, :3, :3]
    t = T[:, :3, 3]

    rotated = torch.bmm(points, R.transpose(-2, -1))  # (B, N, 3)
    transformed = rotated + t[:, None, :]

    return transformed if batched else transformed.squeeze(0)


def extract_rt(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split 4x4 or (B,4,4) SE(3) matrix into rotation (3,3) and translation (3,).
    """
    return T[..., :3, :3], T[..., :3, 3]


def compose(T_ab: torch.Tensor, T_bc: torch.Tensor) -> torch.Tensor:
    """
    Compose two SE(3) transforms: T_ac = T_ab @ T_bc.
    Supports batched inputs.
    """
    return T_ab @ T_bc


def inverse_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Compute SE(3) inverse fast.

    Given T = [R  t; 0^T  1], returns [R^T  -R^T t; 0^T  1].

    Uses the SE(3) inverse identity for computational efficiency
    without performing a full 4x4 matrix inverse.

    Supports batched (B, 4, 4) inputs.
    """
    batched = T.dim() == 3
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_inv = R.transpose(-2, -1)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    if batched:
        T_inv = T_inv.expand(T.shape[0], 4, 4).clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    return T_inv


# ---------------------------------------------------------------------------
# NumPy SE(3) utilities for Open3D-based pipelines
# ---------------------------------------------------------------------------


def random_rotation_matrix(
    max_angle_deg: float = 30.0,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    r"""
    Generate a random SO(3) rotation matrix with bounded angle.

    Samples a uniform axis on S² and a random angle up to max_angle_deg,
    then builds the matrix via Rodrigues' rotation formula:

        R = I + sin(θ)·K + (1 − cos(θ))·K²

    where K is the cross-product matrix of the unit rotation axis ω:
        K = [[0, −ω_z, ω_y], [ω_z, 0, −ω_x], [−ω_y, ω_x, 0]]

    Uniform axis sampling on S² uses the Archimedean param:
        z   ∼ U(−1, 1)
        θ_h ∼ U(0, 2π)
        ω   = [√(1−z²)·cos(θ_h),  √(1−z²)·sin(θ_h),  z]

    Args:
        max_angle_deg: Upper bound on the rotation angle in degrees.
        seed:          Optional RNG seed for reproducibility.

    Returns:
        (R_3x3, angle_rad) — the 3×3 SO(3) matrix and the sampled angle.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0)
    theta_h = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    axis = np.array([s * np.cos(theta_h), s * np.sin(theta_h), z])
    angle = rng.uniform(0.0, np.deg2rad(max_angle_deg))

    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R, float(angle)


def random_se3(
    max_angle_deg: float = 25.0,
    max_translation: float = 0.03,
    seed: int | None = None,
) -> tuple[np.ndarray, float, float]:
    r"""
    Generate a random valid 4×4 SE(3) transformation matrix.

    Composes a bounded uniform-axis/angle SO(3) rotation with a uniform
    R³ translation vector and returns the 4×4 homogeneous matrix:

        T = [R   t]
            [0   1]    ∈  SE(3)

    The translation direction is uniform on S² and magnitude uniform on
    [0, max_translation].

    Args:
        max_angle_deg:    Upper bound on rotation angle in degrees.
        max_translation:  Upper bound on translation norm in metres.
        seed:             Optional RNG seed.

    Returns:
        (T_4x4, angle_deg, t_norm) — 4×4 SE(3) matrix, rotation angle in
        degrees, and translation Euclidean norm in metres.
    """
    rng = np.random.default_rng(seed)
    R, angle_rad = random_rotation_matrix(max_angle_deg, seed=seed)

    # uniform direction on S² for translation
    z = rng.uniform(-1.0, 1.0)
    th = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    direction = np.array([s * np.cos(th), s * np.sin(th), z])
    norm = rng.uniform(0.0, max_translation)
    t = direction * norm

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, float(np.rad2deg(angle_rad)), float(norm)


def transform_points_np(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    r"""
    Apply SE(3) transform to a NumPy point cloud.

        p' = R·p + t    for each p ∈ ℝ³

    Args:
        T:      4×4 SE(3) matrix  [R t; 0 1].
        points: (N, 3) array of point coordinates.

    Returns:
        (N, 3) array of transformed points.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t
