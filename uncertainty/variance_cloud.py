"""
Variance Cloud I/O and Visualisation.

Converts MC Dropout output (mean positions + per-point epistemic variance)
into a 3D point cloud file (PCD) with intensity = variance, suitable for
visual inspection and downstream CVaR filtering.

=== Variance-to-RGB Colormap ===

High-variance regions (occluded geometry, hidden cavities) are rendered
in warm colours (red/orange), while low-variance regions (well-observed
surfaces) are rendered in cool colours (blue/cyan).  This provides an
immediate visual diagnostic of where the model is uncertain.

The colormap uses a perceptually-uniform diverging scale:
    σ² = 0        → deep blue   (0.0, 0.0, 0.5)
    σ² = median   → white       (1.0, 1.0, 1.0)
    σ² = max      → deep red    (0.5, 0.0, 0.0)
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import torch


def compute_variance_cloud(
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> o3d.geometry.PointCloud:
    """
    Build an Open3D point cloud from MC Dropout results.

    Each point carries:
      - position: predictive mean (x, y, z)
      - colour:   variance-to-RGB mapping for visualisation
      - intensity: scalar epistemic variance σ²

    Args:
        mean:     (N, 3) predictive mean positions.
        variance: (N,) per-point epistemic variance.

    Returns:
        Open3D PointCloud with colours and intensity.
    """
    mean_np = mean.detach().cpu().numpy()
    var_np = variance.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mean_np)

    # Colour by variance
    colours = variance_to_rgb(var_np)
    pcd.colors = o3d.utility.Vector3dVector(colours)

    # Store variance as per-point intensity (custom attribute)
    pcd.normals = o3d.utility.Vector3dVector(
        np.column_stack([var_np, np.zeros_like(var_np), np.zeros_like(var_np)])
    )

    return pcd


def variance_to_rgb(
    variance: np.ndarray,
    *,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """
    Map scalar variance values to RGB colours using a diverging colormap.

    Uses a blue-white-red scale where:
      - Low variance (certain) → blue
      - Median variance → white
      - High variance (uncertain) → red

    The upper bound is clipped at the given percentile to prevent a few
    extreme outliers from compressing the entire colour range.

    Args:
        variance:       (N,) scalar variance values.
        clip_percentile: Upper percentile for colour scaling (default 99).

    Returns:
        RGB colours, shape (N, 3), values in [0, 1].
    """
    v = variance.copy()

    # Clip extreme outliers
    v_max = float(np.percentile(v, clip_percentile))
    if v_max <= 0:
        v_max = 1.0
    v = np.clip(v, 0.0, v_max)
    v_norm = v / v_max  # normalise to [0, 1]

    # Diverging colormap: blue (0) → white (0.5) → red (1)
    # Using a smooth interpolation for perceptual uniformity
    r = np.where(v_norm < 0.5, 2 * v_norm, 1.0)
    g = np.where(v_norm < 0.5, 2 * v_norm, 2 * (1 - v_norm))
    b = np.where(v_norm < 0.5, 1.0, 2 * (1 - v_norm))

    # Stack to (N, 3)
    colours = np.stack([r, g, b], axis=-1)
    return np.clip(colours, 0.0, 1.0)


def save_variance_cloud(
    mean: torch.Tensor,
    variance: torch.Tensor,
    output_path: str,
    *,
    clip_percentile: float = 99.0,
) -> None:
    """
    Save variance cloud as PCD file with per-point RGB colours.

    The output file can be opened in any point cloud viewer (CloudCompare,
    Open3D, MeshLab) to visually inspect uncertainty hotspots.

    Args:
        mean:            (N, 3) predictive mean positions.
        variance:        (N,) per-point epistemic variance.
        output_path:     Output file path (.pcd or .ply).
        clip_percentile: Colour scaling upper bound.
    """
    pcd = compute_variance_cloud(mean, variance, clip_percentile=clip_percentile)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"  Saved variance cloud to {output_path}")
    print(f"  Points: {len(pcd.points):,}")
    print(f"  Variance range: [{variance.min().item():.6e}, {variance.max().item():.6e}] m²")


def visualise_variance(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    title: str = "MC Dropout Variance Cloud",
    clip_percentile: float = 99.0,
) -> None:
    """
    Render the variance cloud with colour-mapped uncertainty.

    Blue = certain (low variance), Red = uncertain (high variance).
    A colour bar legend is printed to the console.

    Args:
        mean:            (N, 3) predictive mean positions.
        variance:        (N,) per-point epistemic variance.
        title:           Window title.
        clip_percentile: Colour scaling upper bound.
    """
    pcd = compute_variance_cloud(mean, variance, clip_percentile=clip_percentile)

    # Print colour legend
    var_np = variance.detach().cpu().numpy()
    v_max = float(np.percentile(var_np, clip_percentile))
    print(f"\n  ── Variance Colour Map ──")
    print(f"    Blue  (σ² ≈ 0.0000)  =  Certain (well-observed surface)")
    print(f"    White (σ² ≈ {v_max/2:.4f})  =  Moderate uncertainty")
    print(f"    Red   (σ² ≈ {v_max:.4f})  =  Uncertain (occluded/hidden geometry)")
    print(f"  ─────────────────────────\n")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
    )


def print_variance_statistics(variance: torch.Tensor) -> dict:
    """
    Print and return summary statistics of the epistemic variance.

    Args:
        variance: (N,) per-point epistemic variance.

    Returns:
        Dictionary with mean, median, std, min, max, p95, p99.
    """
    var_np = variance.detach().cpu().numpy()
    stats = {
        "mean": float(np.mean(var_np)),
        "median": float(np.median(var_np)),
        "std": float(np.std(var_np)),
        "min": float(np.min(var_np)),
        "max": float(np.max(var_np)),
        "p95": float(np.percentile(var_np, 95)),
        "p99": float(np.percentile(var_np, 99)),
    }

    print(f"\n  ── Variance Statistics ──")
    print(f"    Mean σ²:    {stats['mean']:.6e} m²")
    print(f"    Median σ²:  {stats['median']:.6e} m²")
    print(f"    Std σ²:     {stats['std']:.6e} m²")
    print(f"    Min σ²:     {stats['min']:.6e} m²")
    print(f"    Max σ²:     {stats['max']:.6e} m²  (occluded region)")
    print(f"    95th %ile:  {stats['p95']:.6e} m²")
    print(f"    99th %ile:  {stats['p99']:.6e} m²")
    print(f"  ─────────────────────────\n")

    return stats
