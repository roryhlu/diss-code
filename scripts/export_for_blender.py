#!/usr/bin/env python3
"""
Blender export helpers for the RePAIR visual pipeline.

Provides mesh loading, normal estimation, and grasp-sphere generation
for the visual pipeline test.  Also provides standalone export
functions for direct use.

Usage
-----
    # As a module (imported by visual_pipeline_test.py)
    from scripts.export_for_blender import load_mesh, estimate_normals, generate_grasp_visualisation

    # Standalone: colour an existing PLY
    python scripts/export_for_blender.py input.ply --colour green --output output.ply
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_mesh(path: str) -> np.ndarray:
    """
    Load OBJ mesh vertices as (N, 3) float64.

    Uses trimesh if available, otherwise falls back to line-by-line
    OBJ parsing for vertex-only extraction.
    """
    ext = Path(path).suffix.lower()

    if ext == ".obj":
        # Fast line-by-line vertex extraction (no mesh processing needed)
        vertices = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not vertices:
            raise ValueError(f"No vertices found in '{path}'")
        return np.array(vertices, dtype=np.float64)

    if ext in (".ply", ".pcd"):
        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if line.strip() == b"end_header":
                    break
            raw = f.read()
        stride = 6  # assume x,y,z,nx,ny,nz from batch_preprocess
        data = np.frombuffer(raw, dtype=np.float64)
        if len(data) % 3 == 0 and len(data) % 6 != 0:
            stride = 3  # x,y,z only
        elif len(data) % 6 == 0:
            stride = 6
        else:
            stride = 3
        data = data.reshape(-1, stride)
        return data[:, :3].astype(np.float64)

    raise ValueError(f"Unsupported format: {ext}")


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Estimate surface normals via vectorised PCA.

    Returns (N, 3) unit normals oriented towards the centroid.
    """
    from scipy.spatial import cKDTree

    k_eff = min(k, len(points))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_eff)

    neighbours = points[idx]
    mu = neighbours.mean(axis=1, keepdims=True)
    centred = neighbours - mu
    cov = np.einsum("nki,nkj->nij", centred, centred) / (k_eff - 1)

    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].copy()

    centroid = points.mean(axis=0)
    dot = np.sum(normals * (centroid - points), axis=1)
    normals[dot < 0] *= -1.0

    ns = np.linalg.norm(normals, axis=1, keepdims=True)
    ns[ns < 1e-12] = 1.0
    return normals / ns


def generate_antipodal_pairs(
    points: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    max_pairs: int = 15,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """
    Find antipodal point pairs on the mesh surface.

    Returns list of (idx1, idx2) pairs.
    """
    rng = np.random.default_rng(seed)
    cos_alpha_min = np.cos(np.arctan(mu)) * 0.5  # relaxed for search
    n_pts = len(points)
    pairs = []

    for _ in range(max_pairs * 100):
        if len(pairs) >= max_pairs:
            break
        i = rng.integers(0, n_pts)
        j = rng.integers(0, n_pts)
        if i == j:
            continue
        d = points[j] - points[i]
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            continue
        d_hat = d / dist
        s1 = float(np.dot(d_hat, normals[i]))
        s2 = float(np.dot(-d_hat, normals[j]))
        if s1 >= cos_alpha_min and s2 >= cos_alpha_min:
            pairs.append((i, j))

    return pairs


def compute_grasp_spheres(
    points: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    seed: int | None = None,
) -> tuple[list, list]:
    """
    Generate accepted/rejected grasp pairs with contact sphere positions.

    Returns (accepted_pairs, rejected_pairs) where each pair is (c1, c2)
    numpy arrays.
    """
    pairs = generate_antipodal_pairs(points, normals, mu=mu, seed=seed)

    accepted = []
    rejected = []

    for idx1, idx2 in pairs:
        c1 = points[idx1]
        c2 = points[idx2]
        n1 = normals[idx1]
        n2 = normals[idx2]

        # Quick antipodal check
        d = c2 - c1
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            rejected.append((c1, c2))
            continue
        d_hat = d / dist
        alpha = np.arctan(mu)
        cos_alpha = np.cos(alpha)
        s1 = float(np.dot(d_hat, n1))
        s2 = float(np.dot(-d_hat, n2))

        if s1 >= cos_alpha - 1e-9 and s2 >= cos_alpha - 1e-9:
            accepted.append((c1, c2))
        else:
            rejected.append((c1, c2))

    return accepted, rejected


# ── CLI: colour an existing PLY ─────────────────────────────────────


def colour_ply(input_path: str, output_path: str, colour_name: str) -> None:
    """Read a PLY, apply a uniform colour, write it."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        raise SystemExit(f"No points in '{input_path}'")
    points = np.asarray(pcd.points, dtype=np.float64)

    colour_map = {
        "grey": [0.6, 0.6, 0.6],
        "blue": [0.1, 0.3, 0.9],
        "green": [0.0, 0.8, 0.2],
        "orange": [1.0, 0.5, 0.0],
        "red": [0.9, 0.1, 0.1],
        "yellow": [1.0, 1.0, 0.0],
    }
    rgb = colour_map.get(colour_name, [0.5, 0.5, 0.5])
    colours = np.full((len(points), 3), rgb)

    # Inline PLY colur writer
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    colours_u8 = np.clip(np.array(colours) * 255, 0, 255).astype(np.uint8)
    data = np.column_stack([points.astype(np.float64), colours_u8])
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="RePAIR Blender export helpers")
    p.add_argument("input", nargs="?", help="Input PLY to colour")
    p.add_argument("--colour", default="grey", help="Colour name: grey, blue, green, orange, red")
    p.add_argument("--output", help="Output PLY path")
    args = p.parse_args()

    if args.input and args.output:
        colour_ply(args.input, args.output, args.colour)
    else:
        print("Use: python export_for_blender.py input.ply --colour green --output output.ply")


if __name__ == "__main__":
    main()
