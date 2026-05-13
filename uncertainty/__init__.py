"""
Uncertainty sub-package — Monte Carlo Dropout for epistemic uncertainty.

Provides:
  GeoTransformer          — Geometric transformer with MC Dropout bottleneck
  run_mc_passes           — T stochastic forward passes with Welford's accumulator
  compute_variance_cloud  — Per-point epistemic variance from MC samples
  save_variance_cloud     — PCD output with intensity = variance
  visualise_variance      — Colour-mapped uncertainty rendering
"""

from uncertainty.geotransformer import GeoTransformer
from uncertainty.mc_inference import run_mc_passes
from uncertainty.variance_cloud import (
    compute_variance_cloud,
    save_variance_cloud,
    visualise_variance,
)

__all__ = [
    "GeoTransformer",
    "run_mc_passes",
    "compute_variance_cloud",
    "save_variance_cloud",
    "visualise_variance",
]
