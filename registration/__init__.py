"""
Registration sub-package — SE(3) registration primitives.

Provides:
  weighted_svd_se3    — Optimal SE(3) via Weighted SVD (Kabsch)
  transform_points    — Apply SE(3) transform to point clouds
  extract_rt          — Split SE(3) into R, t
  compose             — Compose two SE(3) transforms
  inverse_transform   — Fast SE(3) inverse
"""

from registration.se3_utils import (
    compose,
    extract_rt,
    inverse_transform,
    transform_points,
)
from registration.weighted_svd import weighted_svd_se3

__all__ = [
    "weighted_svd_se3",
    "transform_points",
    "extract_rt",
    "compose",
    "inverse_transform",
]
