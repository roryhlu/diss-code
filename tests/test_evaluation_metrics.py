"""Tests for evaluation metrics (scripts/evaluate_registration.py)."""

import numpy as np
import pytest
from scipy.spatial import cKDTree as KDTree

# Inline the pure functions — avoids importing the script which has
# sys.path manipulation and Open3D dependencies.
from tests._eval_helpers import (
    _transform_points_np,
    _extract_rt_np,
    compute_add_s,
    compute_chamfer,
    compute_rms_pose_error,
)


class TestExtractRT:
    def test_identity(self):
        R, t = _extract_rt_np(np.eye(4))
        np.testing.assert_allclose(R, np.eye(3))
        np.testing.assert_allclose(t, [0, 0, 0])

    def test_arbitrary(self):
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        T[:3, 3] = [5, -3, 2]
        R, t = _extract_rt_np(T)
        np.testing.assert_allclose(R, T[:3, :3])
        np.testing.assert_allclose(t, [5, -3, 2])


class TestTransformPointsNP:
    def test_identity(self):
        pts = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.float64)
        np.testing.assert_allclose(_transform_points_np(np.eye(4), pts), pts)

    def test_translation(self):
        T = np.eye(4)
        T[:3, 3] = [2, -3, 1]
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        np.testing.assert_allclose(_transform_points_np(T, pts), [[2, -3, 1]])

    def test_rotation_90_z(self):
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pts = np.array([[1, 0, 0]], dtype=np.float64)
        np.testing.assert_allclose(_transform_points_np(T, pts), [[0, 1, 0]])


class TestAddS:
    def test_exact_match(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        mean, med, p95, per_pt = compute_add_s(pts, pts)
        assert mean == 0.0
        assert med == 0.0
        assert p95 == 0.0
        np.testing.assert_allclose(per_pt, 0.0, atol=1e-15)

    def test_uniform_offset(self):
        # Use a simple case: all points shifted by exactly 3.0 in Z
        pts_model = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]], dtype=np.float64)
        pts_est = pts_model + np.array([0, 0, 3.0])
        mean, med, p95, _ = compute_add_s(pts_est, pts_model)
        assert mean == pytest.approx(3.0, abs=1e-10)
        assert med == pytest.approx(3.0, abs=1e-10)
        assert p95 >= med

    def test_single_est_multiple_model(self):
        pts_est = np.array([[0, 0, 0]], dtype=np.float64)
        pts_model = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0.5]], dtype=np.float64)
        mean, _, _, _ = compute_add_s(pts_est, pts_model)
        assert mean == pytest.approx(0.5, abs=1e-10)

    def test_asymmetric_sets(self):
        pts_est = np.random.default_rng(1).uniform(-3, 3, (50, 3))
        pts_model = np.random.default_rng(2).uniform(-3, 3, (500, 3))
        mean, med, p95, per_pt = compute_add_s(pts_est, pts_model)
        assert mean >= 0
        assert med <= p95
        assert len(per_pt) == 50

    def test_returns_float(self):
        pts = np.random.default_rng(0).uniform(0, 1, (10, 3))
        mean, med, p95, _ = compute_add_s(pts, pts)
        assert isinstance(mean, float)
        assert isinstance(med, float)
        assert isinstance(p95, float)

    def test_empty_est(self):
        pts_est = np.empty((0, 3), dtype=np.float64)
        pts_model = np.array([[0, 0, 0]])
        mean, med, p95, per_pt = compute_add_s(pts_est, pts_model)
        assert np.isnan(mean)  # np.mean([]) = nan
        assert len(per_pt) == 0 and np.isnan(p95)

    def test_identical_output(self):
        """Permuted points should give same result (KDTree is index-agnostic)."""
        pts = np.random.default_rng(0).uniform(-5, 5, (200, 3))
        pts_perm = pts[np.random.default_rng(1).permutation(200)]
        m1, _, _, _ = compute_add_s(pts, pts_perm)
        m2, _, _, _ = compute_add_s(pts_perm, pts)
        assert m1 == 0.0 and m2 == 0.0  # same set of points


class TestChamfer:
    def test_exact_match(self):
        pts = np.random.default_rng(0).uniform(-5, 5, (100, 3))
        fwd, bwd, tot = compute_chamfer(pts, pts)
        assert fwd == 0.0
        assert bwd == 0.0
        assert tot == 0.0

    def test_single_point_pair(self):
        fwd, bwd, tot = compute_chamfer(
            np.array([[0, 0, 0]], dtype=np.float64),
            np.array([[3, 4, 0]], dtype=np.float64),
        )
        assert fwd == 5.0
        assert bwd == 5.0
        assert tot == 10.0

    def test_asymmetric_clouds(self):
        """Forward and backward may differ with different cloud sizes."""
        pts_est = np.array([[0, 0, 0]], dtype=np.float64)
        pts_model = np.array([[3, 0, 0], [0, 4, 0]], dtype=np.float64)
        fwd, bwd, tot = compute_chamfer(pts_est, pts_model)
        assert fwd == pytest.approx(3.0)  # (3)/1
        assert bwd == pytest.approx((3.0 + 4.0) / 2.0)  # (3+4)/2 = 3.5
        assert tot == pytest.approx(6.5)

    def test_large_disjoint_clouds(self):
        pts_est = np.random.default_rng(0).uniform(0, 1, (20, 3))
        pts_model = np.random.default_rng(1).uniform(100, 101, (20, 3))
        fwd, bwd, tot = compute_chamfer(pts_est, pts_model)
        assert fwd > 99
        assert bwd > 99
        assert tot > 198


class TestRMSPoseError:
    def test_perfect_registration(self):
        """T_est = T_gt⁻¹ → both errors = 0."""
        T_gt = np.eye(4)
        T_est = np.eye(4)
        rot, trans = compute_rms_pose_error(T_est, T_gt)
        assert rot == 0.0
        assert trans == 0.0

    def test_known_90_degree_error(self):
        T_gt = np.eye(4)
        R_est = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        T_est = np.eye(4)
        T_est[:3, :3] = R_est
        rot, trans = compute_rms_pose_error(T_est, T_gt)
        assert rot == pytest.approx(90.0, abs=1e-10)
        assert trans == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation_error(self):
        T_gt = np.eye(4)
        T_est = np.eye(4)
        T_est[:3, 3] = [0.1, 0.0, 0.0]
        rot, trans = compute_rms_pose_error(T_est, T_gt)
        assert rot == 0.0
        assert trans == pytest.approx(0.1, abs=1e-10)

    def test_roundtrip_with_gt_inverse(self):
        """Construct T_est = exact inverse of T_gt → errors = 0."""
        # Use a proper SO(3) rotation via scipy
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', [0.1, -0.2, 0.3]).as_matrix()
        t = np.array([0.02, -0.01, 0.015], dtype=np.float64)
        T_gt = np.eye(4)
        T_gt[:3, :3] = R
        T_gt[:3, 3] = t

        T_est = np.eye(4)
        T_est[:3, :3] = R.T
        T_est[:3, 3] = -R.T @ t

        rot, trans = compute_rms_pose_error(T_est, T_gt)
        assert rot == pytest.approx(0.0, abs=1e-5)
        assert trans == pytest.approx(0.0, abs=1e-10)

    def test_arccos_domain_clipping(self):
        """Numerical values slightly outside [-1, 1] must not crash."""
        # near-180° rotation: tr = -1 - ε
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1 - 1e-15]], dtype=np.float64)
        T_gt = np.eye(4)
        T_est = np.eye(4)
        T_est[:3, :3] = R
        rot, _ = compute_rms_pose_error(T_est, T_gt)
        assert rot >= 0 and rot <= 180

    def test_rotation_180_degrees(self):
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        T_gt = np.eye(4)
        T_est = np.eye(4)
        T_est[:3, :3] = R
        rot, _ = compute_rms_pose_error(T_est, T_gt)
        assert rot == pytest.approx(180.0, abs=1e-10)
