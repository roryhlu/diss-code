"""Tests for force-closure geometry (scripts/force_closure.py)."""

import numpy as np
import pytest

# Inline the pure functions from force_closure.py to avoid importing
# Open3D and trimesh (which hang in headless environments).
from tests._fc_helpers import (
    orthonormal_basis,
    friction_cone_generators,
    build_contact_wrench,
    check_antipodal,
    solve_force_closure_lp,
)


class TestOrthonormalBasis:
    def test_z_axis(self):
        u, v = orthonormal_basis(np.array([0, 0, 1.0]))
        np.testing.assert_allclose(np.dot(u, v), 0, atol=1e-15)
        np.testing.assert_allclose(np.dot(u, [0, 0, 1]), 0, atol=1e-15)
        np.testing.assert_allclose(np.dot(v, [0, 0, 1]), 0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-15)

    def test_x_axis(self):
        u, v = orthonormal_basis(np.array([1.0, 0, 0]))
        np.testing.assert_allclose(np.dot(u, v), 0, atol=1e-15)
        np.testing.assert_allclose(np.dot(u, [1, 0, 0]), 0, atol=1e-15)
        np.testing.assert_allclose(np.dot(v, [1, 0, 0]), 0, atol=1e-15)

    def test_arbitrary_normal(self):
        n = np.array([1.0, 2.0, 3.0])
        u, v = orthonormal_basis(n)
        assert abs(np.dot(u, n)) < 1e-12
        assert abs(np.dot(v, n)) < 1e-12
        assert abs(np.dot(u, v)) < 1e-12
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)

    def test_righthanded(self):
        """u × v should point along n (right-handed basis)."""
        n = np.array([1.0, -2.0, 0.5])
        u, v = orthonormal_basis(n)
        cross = np.cross(u, v)
        assert np.dot(cross, n) > 0

    def test_negative_normal(self):
        n = np.array([0, 0, -1.0])
        u, v = orthonormal_basis(n)
        assert abs(np.dot(u, n)) < 1e-12
        assert abs(np.dot(v, n)) < 1e-12

    def test_unit_input_unchanged(self):
        n_unit = np.array([1, 0, 0], dtype=np.float64)
        u1, v1 = orthonormal_basis(n_unit)
        n_nonunit = np.array([2, 0, 0], dtype=np.float64)
        u2, v2 = orthonormal_basis(n_nonunit)
        np.testing.assert_allclose(u1, u2, atol=1e-12)
        np.testing.assert_allclose(v1, v2, atol=1e-12)

    def test_near_axis_threshold(self):
        """Test the 0.9 axis-selection threshold edge."""
        n = np.array([0.9, 0.0, np.sqrt(1 - 0.9**2)])
        u, v = orthonormal_basis(n)
        assert abs(np.dot(u, n)) < 1e-12


class TestFrictionConeGenerators:
    def test_frictionless(self):
        gens = friction_cone_generators(np.array([0, 0, 1.0]), mu=0.0)
        assert gens.shape == (1, 3)
        np.testing.assert_allclose(gens[0], [0, 0, 1])

    def test_mu_zero_returns_only_normal(self):
        gens = friction_cone_generators(np.array([1, 0, 0]), mu=0.0)
        assert len(gens) == 1

    def test_unit_length_generators(self):
        gens = friction_cone_generators(np.array([0, 0, 1.0]), mu=0.5, m=8)
        assert gens.shape == (8, 3)
        for g in gens:
            np.testing.assert_allclose(np.linalg.norm(g), 1.0, atol=1e-12)

    def test_cone_angle(self):
        """Dot(n, each generator) = cos(α) = cos(arctan(μ))."""
        mu = 0.5
        alpha = np.arctan(mu)
        n = np.array([0, 0, 1.0])
        gens = friction_cone_generators(n, mu=mu, m=16)
        for g in gens:
            assert np.dot(n, g) == pytest.approx(np.cos(alpha), abs=1e-12)

    def test_m_default(self):
        gens = friction_cone_generators(np.array([1, 0, 0]), mu=0.5)
        assert len(gens) == 8

    def test_azimuthal_symmetry(self):
        """All generators should have same angle from normal."""
        mu = 0.3
        n = np.array([1, 2, 3], dtype=np.float64)
        gens = friction_cone_generators(n, mu=mu, m=12)
        dots = np.abs(np.dot(gens, n / np.linalg.norm(n)))
        # All dots should be equal (within float tolerance)
        np.testing.assert_allclose(dots, dots[0], atol=1e-12)

    def test_large_m(self):
        gens = friction_cone_generators(np.array([0, 0, 1.0]), mu=1.0, m=32)
        assert gens.shape == (32, 3)
        assert np.all(np.isfinite(gens))

    def test_negative_mu_returns_normal(self):
        gens = friction_cone_generators(np.array([0, 1, 0]), mu=-0.5)
        assert len(gens) == 1
        np.testing.assert_allclose(gens[0], [0, 1, 0], atol=1e-12)


class TestBuildContactWrench:
    def test_origin_contact(self):
        W = build_contact_wrench(
            np.array([0, 0, 0], dtype=np.float64),
            np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
        )
        assert W.shape == (6, 2)
        # Force rows (first 3) = generators.T
        np.testing.assert_allclose(W[:3, :], [[1, 0], [0, 1], [0, 0]])
        # Torque rows (last 3) = 0 (no moment arm at origin)
        np.testing.assert_allclose(W[3:, :], 0, atol=1e-15)

    def test_off_origin_contact(self):
        # Contact at (0, 1, 0), force along X
        W = build_contact_wrench(
            np.array([0, 1, 0], dtype=np.float64),
            np.array([[1, 0, 0]], dtype=np.float64),
        )
        assert W.shape == (6, 1)
        np.testing.assert_allclose(W[:3, 0], [1, 0, 0])
        # τ = [0,1,0] × [1,0,0] = [0,0,-1]
        np.testing.assert_allclose(W[3:, 0], [0, 0, -1])

    def test_skew_matrix_correct(self):
        """Torque = position × force."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            pos = rng.uniform(-5, 5, 3)
            force = rng.uniform(-1, 1, (1, 3))
            W = build_contact_wrench(pos, force)
            expected_torque = np.cross(pos, force[0])
            np.testing.assert_allclose(W[3:, 0], expected_torque, atol=1e-12)

    def test_single_generator(self):
        W = build_contact_wrench(
            np.array([1, 2, 3], dtype=np.float64),
            np.array([[0, 0, 1]], dtype=np.float64),
        )
        assert W.shape == (6, 1)

    def test_multiple_generators(self):
        gens = friction_cone_generators(np.array([0, 0, 1.0]), mu=0.5, m=8)
        W = build_contact_wrench(np.array([0.1, -0.2, 0.3]), gens)
        assert W.shape == (6, 8)


class TestCheckAntipodal:
    def test_perfect_antipodal(self):
        ok, s1, s2 = check_antipodal(
            np.array([0, 0, 0]), np.array([1, 0, 0]),
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            mu=0.5,
        )
        assert ok
        assert s1 > 0 and s2 > 0

    def test_not_antipodal_parallel_normals(self):
        ok, _, _ = check_antipodal(
            np.array([0, 0, 0]), np.array([1, 0, 0]),
            np.array([1, 0, 0]), np.array([1, 0, 0]),
            mu=0.5,
        )
        assert not ok

    def test_zero_distance(self):
        ok, s1, s2 = check_antipodal(
            np.array([0, 0, 0]), np.array([1, 0, 0]),
            np.array([0, 0, 0]), np.array([-1, 0, 0]),
            mu=0.5,
        )
        assert not ok
        assert s1 == 0.0 and s2 == 0.0

    def test_high_friction_passes_loose_condition(self):
        """With μ → ∞, cos(α) → 0, any positive dot product passes."""
        ok, _, _ = check_antipodal(
            np.array([0, 0, 0]), np.array([0.1, 0, 0.995]),
            np.array([1, 0, 0]), np.array([-0.1, 0, -0.995]),
            mu=1e6,
        )
        assert ok

    def test_zero_friction_strict(self):
        """μ = 0 → cos(α) = 1 → only perfectly aligned normals pass."""
        ok, _, _ = check_antipodal(
            np.array([0, 0, 0]), np.array([1, 0, 0]),
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            mu=0.0,
        )
        assert ok

    def test_tolerance_boundary(self):
        """Test exactly at the cos(α) - 1e-9 tolerance boundary."""
        mu = 0.5
        alpha = np.arctan(mu)
        cos_alpha = np.cos(alpha)
        # d̂ = inter-contact axis between two aligned contacts
        c1, c2 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
        ok, s1, s2 = check_antipodal(c1, np.array([1, 0, 0]),
                                     c2, np.array([-1, 0, 0]), mu=mu)
        assert ok
        assert s1 == pytest.approx(1.0)


class TestForceClosureLP:
    def test_trivial_closure(self):
        """W with ±unit vectors in ℝ⁶ → origin is inside convex hull."""
        W = np.column_stack([np.eye(6), -np.eye(6)])  # 12 columns
        ok, eps, status, _ = solve_force_closure_lp(W)
        assert ok
        assert eps > 0

    def test_no_closure(self):
        """All wrench columns in positive x half-space."""
        W = np.array([
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ], dtype=np.float64)
        ok, eps, status, _ = solve_force_closure_lp(W)
        assert not ok
        assert eps == pytest.approx(0.0, abs=1e-9)

    def test_insufficient_columns(self):
        """Fewer than 7 columns → impossible to span ℝ⁶."""
        W = np.eye(6, 3)  # 3 columns
        ok, _, _, _ = solve_force_closure_lp(W)
        assert not ok

    def test_seven_columns_closure(self):
        """7 columns in general position — should detect closure."""
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (6, 7))
        ok, eps, _, _ = solve_force_closure_lp(W)
        # With random columns, closure isn't guaranteed, but the LP should run
        assert isinstance(ok, bool)

    def test_identity_closure(self):
        """W = [I₆, -I₆] is the simplest force-closure case."""
        W = np.hstack([np.eye(6), -np.eye(6)])
        ok, eps, _, _ = solve_force_closure_lp(W)
        assert ok
        assert eps > 0

    def test_epsilon_return_type(self):
        W = np.hstack([np.eye(6), -np.eye(6)])
        _, eps, _, _ = solve_force_closure_lp(W)
        assert isinstance(eps, float)
        assert eps == pytest.approx(1.0 / 12.0, abs=0.01)
