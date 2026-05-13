#!/usr/bin/env python3
"""
AX = XB Hand-Eye Calibration (Tsai-Lenz Formulation).

Determines the fixed rigid-body transformation X ∈ SE(3) between a
camera frame and a robot reference frame from a sequence of paired
motions (A_k, B_k) where:

    A_k : robot end-effector motion between poses k and k+1
          (from forward kinematics, known)

    B_k : camera/scene motion between the same time steps
          (from TEASER++ registration or fiducial detection)

    X   : unknown hand-eye transform to estimate

=== AX = XB  (Eye-in-Hand) ===

Camera rigidly mounted on the robot end-effector.  X maps from
camera frame to end-effector frame.  For each motion pair:

    A_k · X = X · B_k

where A_k, B_k, X ∈ SE(3) are 4×4 homogeneous matrices.

=== AX = ZB  (Eye-to-Hand) ===

Camera fixed in the world frame.  X maps from camera frame to the
robot base frame.  Z is a calibration-pattern-to-end-effector
transform (known).  For each motion:

    A_k · Z = X · B_k

In RePAIR, the D405 depth camera is mounted on the table (eye-to-hand),
observing the workspace.  We solve for X using the Tsai-Lenz
decomposition.

=== Tsai-Lenz Decomposition ===

Separate rotation and translation:

    R_A · R_X = R_X · R_B          (rotation — solve first)
    R_A · t_X + t_A = R_X · t_B + t_X   (translation — solve second)

Step 1 — Rotation (axis-angle formulation):
    Let  r_a, r_b, r_x ∈ R³  be the axis-angle representations of
    R_A, R_B, R_X.  Then:

        R_A · R_X = R_X · R_B
    ⇔  skew(r_a + r_b) · P'_x = (r_b − r_a)

    Solve for P'_x via least squares from ≥ 2 motion pairs, recover R_X.

Step 2 — Translation:
    (R_A − I) · t_X = R_X · t_B − t_A

    Solve for t_X via least squares from the motion pairs.

=== Usage ===

    from repair_simulation.hand_eye import HandEyeCalibration

    calib = HandEyeCalibration("eye_to_hand")
    X = calib.calibrate(motions_A, motions_B)
    # X: 4×4 SE(3) camera → robot base frame transform
"""

from __future__ import annotations

import math
import numpy as np


class HandEyeCalibration:
    """
    Solve the AX = XB or AX = ZB hand-eye calibration problem.

    Supports both eye-in-hand (camera on end-effector) and eye-to-hand
    (camera fixed in world) configurations.
    """

    def __init__(
        self,
        mode: str = "eye_to_hand",
        X_init: np.ndarray | None = None,
    ):
        """
        Args:
            mode:   "eye_to_hand" (camera fixed in world, RePAIR default)
                    or "eye_in_hand" (camera on end-effector).
            X_init: Optional initial guess for X (4×4 SE(3)).  Default
                    is identity with 0.5m X-offset (D405 at table edge).
        """
        if mode not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"Unknown mode '{mode}'")
        self.mode = mode

        if X_init is None:
            X_init = np.eye(4, dtype=np.float64)
            if mode == "eye_to_hand":
                # Camera 0.5m in front of robot base (typical D405 placement)
                X_init[0, 3] = 0.500
                X_init[2, 3] = 0.350  # table height
        self._X = X_init.copy()

    @property
    def X(self) -> np.ndarray:
        """Current hand-eye calibration 4×4 SE(3) matrix."""
        return self._X.copy()

    @property
    def R(self) -> np.ndarray:
        """Rotation block of X."""
        return self._X[:3, :3].copy()

    @property
    def t(self) -> np.ndarray:
        """Translation vector of X."""
        return self._X[:3, 3].copy()

    def calibrate(
        self,
        motions_A: list[np.ndarray],
        motions_B: list[np.ndarray],
    ) -> np.ndarray:
        """
        Solve for X from a sequence of paired SE(3) motions.

        Args:
            motions_A: List of 4×4 SE(3) robot end-effector motions.
                       For eye-to-hand: A_k = T_{ee}_{k+1}⁻¹ · T_{ee}_k.
            motions_B: List of 4×4 SE(3) camera/scene motions.
                       For eye-to-hand: B_k = T_{cam}_{k+1}⁻¹ · T_{cam}_k.

        Returns:
            4×4 SE(3) hand-eye calibration matrix X.
        """
        n = len(motions_A)
        if n < 2:
            raise ValueError(f"Need ≥ 2 motion pairs, got {n}")
        if n != len(motions_B):
            raise ValueError("motions_A and motions_B must have same length")

        # ── Step 1: Solve rotation ──
        # Build linear system for P'_x = 2 sin(θ/2) · axis
        M = np.zeros((3 * n, 3), dtype=np.float64)
        v = np.zeros(3 * n, dtype=np.float64)

        for k in range(n):
            RA = motions_A[k][:3, :3]
            RB = motions_B[k][:3, :3]

            # Axis-angle representations
            ra = _so3_log(RA)
            rb = _so3_log(RB)

            # skew(ra + rb) @ P'x = rb - ra
            rsum = ra + rb
            Skew = np.array([
                [0.0,        -rsum[2],   rsum[1]],
                [rsum[2],     0.0,      -rsum[0]],
                [-rsum[1],    rsum[0],    0.0],
            ])
            M[3*k:3*k+3, :] = Skew
            v[3*k:3*k+3] = rb - ra

        # Least-squares solve
        Px_prime, _, _, _ = np.linalg.lstsq(M, v, rcond=None)

        # Recover rotation from P'_x
        # P'_x = 2 sin(θ/2) · axis
        theta_x = 2.0 * math.asin(np.clip(np.linalg.norm(Px_prime) / 2.0, -1.0, 1.0))
        if np.linalg.norm(Px_prime) < 1e-12:
            axis_x = np.array([0.0, 0.0, 1.0])
        else:
            axis_x = Px_prime / np.linalg.norm(Px_prime)
        RX = _so3_exp(axis_x * theta_x)

        # ── Step 2: Solve translation ──
        # (RA - I) · tX = RX · tB - tA
        M_t = np.zeros((3 * n, 3), dtype=np.float64)
        v_t = np.zeros(3 * n, dtype=np.float64)

        for k in range(n):
            RA = motions_A[k][:3, :3]
            RB = motions_B[k][:3, :3]
            tA = motions_A[k][:3, 3]
            tB = motions_B[k][:3, 3]

            M_t[3*k:3*k+3, :] = RA - np.eye(3)
            v_t[3*k:3*k+3] = RX @ tB - tA

        tX, _, _, _ = np.linalg.lstsq(M_t, v_t, rcond=None)

        # ── Assemble X ──
        self._X = np.eye(4, dtype=np.float64)
        self._X[:3, :3] = RX
        self._X[:3, 3] = tX

        return self._X

    def transform_pose(self, T_camera: np.ndarray) -> np.ndarray:
        """
        Transform a camera-frame 6D pose to the robot base frame.

        Eye-to-hand:  T_robot = X · T_camera
        Eye-in-hand:  T_robot = T_ee · X · T_camera
                       (requires current end-effector pose T_ee)

        Args:
            T_camera: 4×4 SE(3) pose in camera frame.

        Returns:
            4×4 SE(3) pose in robot base frame.
        """
        if self.mode == "eye_to_hand":
            return self._X @ T_camera
        else:
            # Eye-in-hand requires end-effector pose — not available here
            raise RuntimeError(
                "Eye-in-hand requires end-effector pose. "
                "Use transform_pose_with_ee() instead."
            )

    def transform_pose_with_ee(
        self, T_camera: np.ndarray, T_ee: np.ndarray
    ) -> np.ndarray:
        """
        Transform camera-frame pose to robot base (eye-in-hand).

        T_robot = T_ee · X · T_camera

        Args:
            T_camera: 4×4 pose in camera frame.
            T_ee:     4×4 current end-effector pose from FK.

        Returns:
            4×4 pose in robot base frame.
        """
        return T_ee @ self._X @ T_camera

    def set_calibration(self, X: np.ndarray):
        """Manually set the calibration matrix."""
        if X.shape != (4, 4):
            raise ValueError(f"X must be 4×4, got {X.shape}")
        self._X = X.copy()

    @staticmethod
    def generate_motion(
        R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Pack rotation R (3×3) and translation t (3,) into SE(3) matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------

def _so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map: SO(3) → so(3), returns axis-angle vector ω."""
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    omega_hat = (R - R.T) / (2.0 * np.sin(theta))
    omega = theta * np.array([
        omega_hat[2, 1],
        omega_hat[0, 2],
        omega_hat[1, 0],
    ], dtype=np.float64)
    return omega


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map: so(3) → SO(3), returns 3×3 rotation matrix."""
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = omega / theta
    K = np.array([
        [0.0,       -axis[2],   axis[1]],
        [axis[2],    0.0,      -axis[0]],
        [-axis[1],   axis[0],    0.0],
    ])
    R = (np.eye(3) + np.sin(theta) * K +
         (1.0 - np.cos(theta)) * K @ K)
    return R
