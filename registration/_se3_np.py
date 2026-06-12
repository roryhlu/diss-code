"""
Pure NumPy SE(3) utility functions.

Isolated from se3_utils.py so they can be imported and tested without
triggering the torch dependency (which hangs in headless environments).

Provides random SE(3) generation and point transformation.
"""

from __future__ import annotations

import numpy as np


def random_rotation_matrix(
    max_angle_deg: float = 30.0,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    r"""
    Generate a random SO(3) rotation matrix with bounded angle.

    Samples a uniform axis on S² and a random angle up to max_angle_deg,
    then builds the matrix via Rodrigues' rotation formula:

        R = I + sin(θ)·K + (1 − cos(θ))·K²

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
    R³ translation vector.

        T = [R   t]
            [0   1]    ∈  SE(3)

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


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
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
