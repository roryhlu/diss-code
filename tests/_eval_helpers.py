"""
Pure-math helpers extracted from scripts/evaluate_registration.py.

Importing the full evaluate_registration.py triggers sys.path manipulation
and optional Open3D imports.  This module re-exports only the stateless
metric-computation functions for direct unit testing.
"""

import numpy as np
from scipy.spatial import cKDTree


def _transform_points_np(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply SE(3) transform to NumPy points: p' = R·p + t."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def _extract_rt_np(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split 4×4 SE(3) into (R_3x3, t_3)."""
    return T[:3, :3], T[:3, 3]


def compute_add_s(
    points_est: np.ndarray,
    points_model: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """ADD-S: mean, median, p95 nearest-neighbour distance."""
    tree = cKDTree(points_model)
    dists, _ = tree.query(points_est, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return (
        float(np.mean(dists)),
        float(np.median(dists)),
        float(np.percentile(dists, 95)) if len(dists) > 0 else float("nan"),
        dists,
    )


def compute_chamfer(
    points_est: np.ndarray,
    points_model: np.ndarray,
) -> tuple[float, float, float]:
    """Bidirectional Chamfer distance = forward + backward."""
    t_model = cKDTree(points_model)
    t_est = cKDTree(points_est)
    fwd = float(np.mean(t_model.query(points_est, k=1)[0]))
    bwd = float(np.mean(t_est.query(points_model, k=1)[0]))
    return fwd, bwd, fwd + bwd


def compute_rms_pose_error(
    T_est: np.ndarray,
    T_gt: np.ndarray,
) -> tuple[float, float]:
    """
    RMS rotation error (°) and translation error (m).

    T_gt: model → scene.  T_est: scene → model.
    Compares T_est with T_gt⁻¹.
    """
    R_est, t_est = _extract_rt_np(T_est)
    R_gt, t_gt = _extract_rt_np(T_gt)

    R_gt = np.asarray(R_gt, dtype=np.float64)
    t_gt = np.asarray(t_gt, dtype=np.float64)
    R_est = np.asarray(R_est, dtype=np.float64)
    t_est = np.asarray(t_est, dtype=np.float64)

    tr = np.trace(R_gt @ R_est)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.rad2deg(np.arccos(cos_theta)))

    t_gt_inv = -R_gt.T @ t_gt
    trans_err = float(np.linalg.norm(t_est - t_gt_inv))

    return rot_err_deg, trans_err
