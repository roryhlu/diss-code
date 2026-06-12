"""Tests for SE(3) NumPy utilities (registration/_se3_np.py)."""

import numpy as np
import pytest
from registration._se3_np import random_rotation_matrix, random_se3, transform_points


class TestRandomRotationMatrix:
    def test_identity_at_zero_angle(self):
        R, angle = random_rotation_matrix(max_angle_deg=0.0, seed=42)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)
        assert angle == 0.0

    def test_deterministic_with_seed(self):
        R1, a1 = random_rotation_matrix(max_angle_deg=30.0, seed=42)
        R2, a2 = random_rotation_matrix(max_angle_deg=30.0, seed=42)
        np.testing.assert_array_equal(R1, R2)
        assert a1 == a2

    def test_different_seeds_different_output(self):
        R1, _ = random_rotation_matrix(max_angle_deg=30.0, seed=42)
        R2, _ = random_rotation_matrix(max_angle_deg=30.0, seed=99)
        assert not np.allclose(R1, R2)

    def test_is_valid_so3(self):
        for seed in [0, 1, 42, 1000]:
            R, _ = random_rotation_matrix(max_angle_deg=180.0, seed=seed)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_angle_within_bounds(self):
        for _ in range(50):
            max_ang = np.random.uniform(0, 180)
            _, angle = random_rotation_matrix(max_angle_deg=max_ang, seed=None)
            assert 0 <= angle <= np.deg2rad(max_ang)

    def test_full_180_degrees(self):
        """Full 180° range: Rodrigues handles sin(π)=0 correctly."""
        R, angle = random_rotation_matrix(max_angle_deg=180.0, seed=42)
        assert 0 <= angle <= np.pi
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_tiny_angle(self):
        R, angle = random_rotation_matrix(max_angle_deg=1e-6, seed=1)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)

    def test_extracts_rotation_angle(self):
        """Verify the returned angle matches arachos trace formula."""
        R, angle_rad = random_rotation_matrix(max_angle_deg=90.0, seed=42)
        tr = np.trace(R)
        cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        recovered = np.arccos(cos_theta)
        assert abs(recovered - angle_rad) < 1e-12


class TestRandomSE3:
    def test_identity_when_both_zero(self):
        T, ang, t_norm = random_se3(max_angle_deg=0.0, max_translation=0.0, seed=42)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-15)
        assert ang == 0.0
        assert t_norm == 0.0

    def test_deterministic_with_seed(self):
        T1, a1, t1 = random_se3(max_angle_deg=25.0, max_translation=0.03, seed=42)
        T2, a2, t2 = random_se3(max_angle_deg=25.0, max_translation=0.03, seed=42)
        np.testing.assert_array_equal(T1, T2)
        assert a1 == a2 and t1 == t2

    def test_se3_properties(self):
        for seed in [0, 1, 42]:
            T, _, _ = random_se3(max_angle_deg=60.0, max_translation=0.05, seed=seed)
            np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_angle_and_translation_within_bounds(self):
        for _ in range(30):
            T, ang, t_norm = random_se3(max_angle_deg=30.0, max_translation=0.05, seed=None)
            assert 0 <= ang <= 30.0
            assert 0 <= t_norm <= 0.05

    def test_pure_rotation_no_translation(self):
        T, ang, t_norm = random_se3(max_angle_deg=45.0, max_translation=0.0, seed=7)
        np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=1e-15)
        assert t_norm == 0.0

    def test_pure_translation_no_rotation(self):
        T, ang, t_norm = random_se3(max_angle_deg=0.0, max_translation=0.1, seed=7)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-15)
        assert ang == 0.0
        assert 0 <= t_norm <= 0.1


class TestTransformPoints:
    def test_identity(self):
        T = np.eye(4)
        pts = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
        np.testing.assert_allclose(transform_points(T, pts), pts, atol=1e-15)

    def test_pure_translation(self):
        T = np.eye(4)
        T[:3, 3] = [2, -3, 4]
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        expected = np.array([[2, -3, 4], [3, -2, 5]], dtype=np.float64)
        np.testing.assert_allclose(transform_points(T, pts), expected)

    def test_rotation_90_z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        pts = np.array([[1, 0, 0]], dtype=np.float64)
        np.testing.assert_allclose(transform_points(T, pts), [[0, 1, 0]])

    def test_rotation_plus_translation(self):
        # 180° about X
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [1, 1, 1]
        pts = np.array([[0, 1, 0]], dtype=np.float64)
        expected = np.array([[1, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(transform_points(T, pts), expected)

    def test_origin_stays_at_translation(self):
        T, _, _ = random_se3(seed=42)
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        result = transform_points(T, pts)
        np.testing.assert_allclose(result.ravel(), T[:3, 3])

    def test_empty_points(self):
        T = np.eye(4)
        pts = np.empty((0, 3), dtype=np.float64)
        result = transform_points(T, pts)
        assert result.shape == (0, 3)

    def test_single_point(self):
        T, _, _ = random_se3(seed=0)
        pts = np.array([[1.5, -2.0, 3.0]], dtype=np.float64)
        result = transform_points(T, pts)
        assert result.shape == (1, 3)

    def test_batch_of_points(self):
        T, _, _ = random_se3(seed=5)
        pts = np.random.default_rng(0).uniform(-10, 10, (1000, 3))
        result = transform_points(T, pts)
        assert result.shape == (1000, 3)
        # Verify against manual computation for a few points
        R = T[:3, :3]
        t = T[:3, 3]
        for i in [0, 100, 999]:
            expected = R @ pts[i] + t
            np.testing.assert_allclose(result[i], expected, atol=1e-12)

    def test_roundtrip_with_inverse(self):
        """Applying T then T⁻¹ should recover original points."""
        T, _, _ = random_se3(max_angle_deg=45.0, max_translation=0.1, seed=42)
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t

        pts = np.random.default_rng(1).uniform(-5, 5, (100, 3))
        forward = transform_points(T, pts)
        back = transform_points(T_inv, forward)
        np.testing.assert_allclose(back, pts, atol=1e-12)
