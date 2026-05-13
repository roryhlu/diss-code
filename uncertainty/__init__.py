"""
Uncertainty sub-package — Monte Carlo Dropout for epistemic uncertainty.

Provides:
  GeoTransformer          — Geometric transformer with MC Dropout bottleneck
  run_mc_passes           — T stochastic forward passes with Welford's accumulator
  compute_pose_covariance — 6×6 Σ from T SE(3) pose samples
  project_spatial_variance— Per-point 3D spatial variance from pose Σ
  print_covariance_report — Human-readable Σ analysis
  variance_to_rgb         — Scalar variance → RGB colour map
  compute_variance_cloud  — Per-point epistemic variance from MC samples
  save_variance_cloud     — PCD output with intensity = variance
  visualise_variance      — Colour-mapped uncertainty rendering
"""

from uncertainty.geotransformer import GeoTransformer
from uncertainty.mc_inference import run_mc_passes
from uncertainty.pose_covariance import (
    compute_pose_covariance,
    print_covariance_report,
    project_spatial_variance,
    variance_to_rgb,
)
from uncertainty.variance_cloud import (
    compute_variance_cloud,
    save_variance_cloud,
    visualise_variance,
)

__all__ = [
    "GeoTransformer",
    "run_mc_passes",
    "compute_pose_covariance",
    "project_spatial_variance",
    "print_covariance_report",
    "variance_to_rgb",
    "compute_variance_cloud",
    "save_variance_cloud",
    "visualise_variance",
]
