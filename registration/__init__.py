"""
Registration sub-package — SE(3) registration primitives.

Provides:
  register_teaser       — TEASER++ global registration with TLS cost + SE(3) cert
  register_scene_to_cad — Scene→CAD registration with domain-optimised presets
  weighted_svd_se3      — Optimal SE(3) via Weighted SVD (Kabsch)
  compute_fpfh          — FPFH descriptor computation
  match_features        — Feature matching with Lowe ratio test
  validate_se3          — Verify SO(3) orthogonality + det(R)=+1
  transform_points      — Apply SE(3) transform to point clouds
  extract_rt            — Split SE(3) into R, t
  compose               — Compose two SE(3) transforms
  inverse_transform     — Fast SE(3) inverse
"""

from registration.fpfh_features import compute_fpfh, match_features
from registration.se3_utils import (
    compose,
    extract_rt,
    inverse_transform,
    transform_points,
)
from registration.teaser_registration import (
    SE3Result,
    TeaserParams,
    register_scene_to_cad,
    register_teaser,
    validate_se3,
)

from registration.weighted_svd import weighted_svd_se3

__all__ = [
    "register_teaser",
    "register_scene_to_cad",
    "SE3Result",
    "TeaserParams",
    "validate_se3",
    "weighted_svd_se3",
    "compute_fpfh",
    "match_features",
    "transform_points",
    "extract_rt",
    "compose",
    "inverse_transform",
]
