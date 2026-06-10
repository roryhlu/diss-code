#!/usr/bin/env python3
"""
Synthetic archaeological fragment generator for GeoTransformer pre-training.

Creates diverse, textureless 3D point clouds that mimic real RePAIR
fragments — pottery sherds, fresco plates, natural rubble — with
physically plausible broken edges, surface roughness, and varied
curvature.  All fragments have PCA-estimated normals for FPFH
compatibility.

Fragment types
--------------
  pottery    — Spherical/cylindrical section with variable curvature,
               thickness, and jagged broken edges.  Mimics amphora and
               bowl sherds from the Pompeii dataset.
  fresco     — Flat plate with irregular fractured boundary, variable
               thickness, and surface undulation.  Mimics wall fresco
               fragments.
  rubble     — Randomly-deformed convex polyhedra with Perlin-like
               surface noise.  Mimics natural stone debris and heavily
               eroded fragments.

Output
------
  synthetic_fragments/FRAG_000.ply … FRAG_029.ply
  Each PLY contains (N, 6) double-precision fields: x,y,z,nx,ny,nz
  Point count ranges from 800–8,000 depending on fragment size.

Usage
-----
    python scripts/generate_synthetic_fragments.py --num 30 --output-dir synthetic_fragments
    python scripts/train_geotransformer.py synthetic_fragments/FRAG_*.ply RPf_00577_ds.ply RPf_00579_ds.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ── Core geometry primitives ─────────────────────────────────────────


def _sample_sphere_shell(
    radius: float,
    thickness: float,
    n_points: int,
    theta_range: tuple[float, float] = (0.0, np.pi),
    phi_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample points on a spherical shell section.

    Points uniformly fill volume between radius and radius+thickness,
    restricted to angular ranges theta/phi.
    """
    if rng is None:
        rng = np.random.default_rng()
    r = rng.uniform(radius, radius + thickness, n_points)
    theta = rng.uniform(*theta_range, n_points)
    phi = rng.uniform(*phi_range, n_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def _sample_cylinder_shell(
    radius: float,
    thickness: float,
    height: float,
    n_points: int,
    theta_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample points on a cylindrical shell section."""
    if rng is None:
        rng = np.random.default_rng()
    r = rng.uniform(radius, radius + thickness, n_points)
    theta = rng.uniform(*theta_range, n_points)
    z = rng.uniform(0.0, height, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def _sample_flat_plate(
    width: float,
    depth: float,
    thickness: float,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample points in a rectangular volume (plate)."""
    if rng is None:
        rng = np.random.default_rng()
    x = rng.uniform(-width / 2, width / 2, n_points)
    y = rng.uniform(-depth / 2, depth / 2, n_points)
    z = rng.uniform(-thickness / 2, thickness / 2, n_points)
    return np.column_stack([x, y, z])


def _sample_ellipsoid(
    radii: tuple[float, float, float],
    n_surface: int,
    n_bulk: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample points on and optionally inside an ellipsoid surface.

    Surface points: uniform on S², scaled by radii.
    Bulk points: uniform in the ellipsoid volume.
    """
    if rng is None:
        rng = np.random.default_rng()
    rx, ry, rz = radii

    # Surface — uniform on sphere then scale
    theta = rng.uniform(0.0, np.pi, n_surface)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_surface)
    x = rx * np.sin(theta) * np.cos(phi)
    y = ry * np.sin(theta) * np.sin(phi)
    z = rz * np.cos(theta)

    pts = [np.column_stack([x, y, z])]

    if n_bulk > 0:
        # Rejection sampling for volume
        bulk = []
        while len(bulk) < n_bulk:
            cand = rng.uniform(-1, 1, (n_bulk * 3, 3))
            inside = (
                (cand[:, 0] / rx) ** 2
                + (cand[:, 1] / ry) ** 2
                + (cand[:, 2] / rz) ** 2
            ) < 1.0
            if inside.sum() > 0:
                bulk.append(cand[inside])
        bulk_pts = np.vstack(bulk)[:n_bulk]
        pts.append(bulk_pts)

    return np.vstack(pts)


# ── Edge fracture ────────────────────────────────────────────────────


def _fracture_edges(
    points: np.ndarray,
    rng: np.random.Generator,
    n_clips: int = 3,
    keep_prob: float = 0.85,
) -> np.ndarray:
    """
    Clip the point cloud with random planes to create jagged broken edges.

    Each clip: sample a random plane (normal + offset), remove points on
    one side.  Repeats n_clips times.  keep_prob controls how many points
    survive; each clip removes ~(1-keep_prob)/n_clips fraction.
    """
    for _ in range(n_clips):
        normal = rng.normal(0, 1, 3)
        normal /= np.linalg.norm(normal)
        centroid = points.mean(axis=0)
        offset = rng.uniform(
            -np.max(np.abs(points - centroid)) * 0.3,
            np.max(np.abs(points - centroid)) * 0.3,
        )
        side = points @ normal - (normal @ centroid + offset)
        if side.mean() < 0:
            normal = -normal
            side = -side
        # Keep points on the "inside" (positive side) with keep_prob
        mask = side > 0
        if mask.sum() < 10:
            continue
        discard = rng.random(mask.sum()) > keep_prob
        mask_idx = np.where(mask)[0]
        mask[mask_idx[discard]] = False
        points = points[mask]
        if len(points) < 10:
            break
    return points


# ── Perlin-like surface noise ────────────────────────────────────────


def _surface_roughness(
    points: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 0.02,
    octaves: int = 3,
) -> np.ndarray:
    """
    Add multi-octave noise to point positions to simulate surface roughness.

    Uses a simple sum-of-sinusoids approximation of Perlin noise.
    amplitude is relative to the point cloud's bounding-box diagonal.
    """
    bbox = points.max(axis=0) - points.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    if diag < 1e-8:
        return points
    amp = diag * amplitude

    noise = np.zeros_like(points)
    for octave in range(octaves):
        freq = 2 ** octave
        amp_oct = amp / (octave + 1)
        for axis in range(3):
            phase = rng.uniform(0, 2 * np.pi)
            noise[:, axis] += amp_oct * np.sin(
                freq * 2 * np.pi * points[:, (axis + 1) % 3] / diag + phase
            )
    return points + noise


# ── Normal estimation ────────────────────────────────────────────────


def _estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    PCA normal estimation via KD-tree local neighbourhood.

    Returns (N, 3) normals oriented such that the z-component is positive
    (consistent with Open3D convention).
    """
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    normals = np.empty((len(points), 3), dtype=np.float64)
    for i in range(len(points)):
        neighbours = points[idx[i]]
        cov = np.cov(neighbours.T, bias=False)
        _, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, 0]  # smallest eigenvalue
        # Orient consistently
        if np.dot(n, points[i] - points.mean(axis=0)) < 0:
            n = -n
        normals[i] = n
    return normals


# ── Fragment type generators ─────────────────────────────────────────


def generate_pottery_sherd(
    rng: np.random.Generator,
    n_points: int | None = None,
) -> np.ndarray:
    """
    Generate a pottery-sherd-like point cloud.

    Uses spherical or cylindrical shell geometry, fractured edges,
    and surface roughness.
    """
    if n_points is None:
        n_points = int(rng.integers(2000, 6000))

    # Vary between spherical (bowl) and cylindrical (amphora body) shapes
    if rng.random() < 0.5:
        radius = rng.uniform(0.3, 1.2)
        thickness = rng.uniform(0.02, 0.12) * radius
        theta_range = (rng.uniform(0.1, 0.6), rng.uniform(1.8, np.pi - 0.1))
        phi_range = (rng.uniform(0.1, 1.0), rng.uniform(1.5, 2.0 * np.pi - 0.1))
        pts = _sample_sphere_shell(
            radius, thickness, n_points * 2,
            theta_range=theta_range, phi_range=phi_range, rng=rng,
        )
    else:
        radius = rng.uniform(0.2, 0.8)
        thickness = rng.uniform(0.02, 0.10)
        height = rng.uniform(0.3, 1.5)
        theta_range = (rng.uniform(0.1, 1.5), rng.uniform(2.0, 2.0 * np.pi - 0.1))
        pts = _sample_cylinder_shell(
            radius, thickness, height, n_points * 2,
            theta_range=theta_range, rng=rng,
        )

    # Fracture edges to create broken pottery look
    pts = _fracture_edges(pts, rng, n_clips=rng.integers(3, 7), keep_prob=0.85)

    # Add surface roughness
    pts = _surface_roughness(pts, rng, amplitude=rng.uniform(0.005, 0.03))

    # Random rotation so fragments have varied orientations
    rot = Rotation.random(random_state=rng.integers(1_000_000))
    pts = pts @ rot.as_matrix().T

    # Trim to target point count
    if len(pts) > n_points:
        idx = rng.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    elif len(pts) < n_points:
        extra = rng.choice(len(pts), n_points - len(pts), replace=True)
        pts = np.vstack([pts, pts[extra]])

    return pts


def generate_fresco_fragment(
    rng: np.random.Generator,
    n_points: int | None = None,
) -> np.ndarray:
    """
    Generate a fresco-fragment-like point cloud.

    Flat plate with irregular broken boundary, varied thickness,
    and gentle surface undulation.
    """
    if n_points is None:
        n_points = int(rng.integers(1500, 5000))

    width = rng.uniform(0.3, 1.5)
    depth = rng.uniform(0.3, 1.5)
    thickness = rng.uniform(0.01, 0.06)

    # Dense sampling on the plate
    pts = _sample_flat_plate(width, depth, thickness, n_points * 3, rng=rng)

    # Add gentle surface undulation
    bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    undulations = np.zeros_like(pts)
    for scale, amp in [(0.05, 0.3), (0.02, 0.15), (0.01, 0.07)]:
        for ax in [0, 1]:
            phase = rng.uniform(0, 2 * np.pi)
            pm = pts[:, (ax + 1) % 2] if ax == 0 else pts[:, ax]
            undulations[:, 2] += amp * bbox_diag * 0.01 * np.sin(
                2 * np.pi * pm / width / scale + phase
            )
    pts[:, 2] += undulations[:, 2]

    # Fracture edges
    pts = _fracture_edges(pts, rng, n_clips=rng.integers(4, 9), keep_prob=0.88)

    # Surface roughness
    pts = _surface_roughness(pts, rng, amplitude=rng.uniform(0.003, 0.015))

    # Random rotation
    rot = Rotation.random(random_state=rng.integers(1_000_000))
    pts = pts @ rot.as_matrix().T

    if len(pts) > n_points:
        idx = rng.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    elif len(pts) < n_points:
        extra = rng.choice(len(pts), n_points - len(pts), replace=True)
        pts = np.vstack([pts, pts[extra]])

    return pts


def generate_rubble_chunk(
    rng: np.random.Generator,
    n_points: int | None = None,
) -> np.ndarray:
    """
    Generate a rubble-like irregular chunk.

    Uses random ellipsoid geometry, heavy surface noise, and multiple
    fracture clips to create natural stone/debris appearance.
    """
    if n_points is None:
        n_points = int(rng.integers(1000, 4000))

    rx = rng.uniform(0.2, 0.8)
    ry = rng.uniform(0.15, 0.7)
    rz = rng.uniform(0.1, 0.5)

    pts = _sample_ellipsoid(
        (rx, ry, rz),
        n_surface=n_points,
        n_bulk=n_points // 4,
        rng=rng,
    )

    # Heavy surface noise
    pts = _surface_roughness(pts, rng, amplitude=rng.uniform(0.02, 0.08), octaves=4)

    # Irregular fracture edges
    pts = _fracture_edges(pts, rng, n_clips=rng.integers(2, 5), keep_prob=0.90)

    # Varied orientations
    rot = Rotation.random(random_state=rng.integers(1_000_000))
    pts = pts @ rot.as_matrix().T

    if len(pts) > n_points:
        idx = rng.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    elif len(pts) < n_points:
        extra = rng.choice(len(pts), n_points - len(pts), replace=True)
        pts = np.vstack([pts, pts[extra]])

    return pts


# ── PLY writer ───────────────────────────────────────────────────────


def save_ply(path: str, points: np.ndarray, normals: np.ndarray) -> None:
    """
    Write an ASCII PLY file with x,y,z,nx,ny,nz in double precision.

    Uses ASCII format for maximum compatibility with Open3D and
    downstream tools (voxel_downsample_normals.py, FPFH pipeline).
    """
    n = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"comment Created by synthetic fragment generator\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(
                f"{points[i, 0]:.8f} {points[i, 1]:.8f} {points[i, 2]:.8f} "
                f"{normals[i, 0]:.8f} {normals[i, 1]:.8f} {normals[i, 2]:.8f}\n"
            )


# ── Batch generator ──────────────────────────────────────────────────


def generate_fragments(
    num: int,
    output_dir: Path,
    seed: int = 42,
    types: tuple[str, ...] = ("pottery", "fresco", "rubble"),
) -> list[Path]:
    """
    Generate `num` synthetic fragments and save as PLY files.

    Fragments are distributed across the requested types with random
    variation in size, shape, and surface detail.  Each fragment has
    PCA-estimated normals.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []

    for i in range(num):
        frag_type = types[i % len(types)]
        frag_rng = np.random.default_rng(seed + i * 137 + 1)

        if frag_type == "pottery":
            pts = generate_pottery_sherd(frag_rng)
        elif frag_type == "fresco":
            pts = generate_fresco_fragment(frag_rng)
        else:
            pts = generate_rubble_chunk(frag_rng)

        # Estimate normals via PCA
        normals = _estimate_normals(pts)

        path = output_dir / f"FRAG_{i:03d}.ply"
        save_ply(str(path), pts, normals)
        paths.append(path)

        print(
            f"  [{i+1:3d}/{num}] {frag_type:>8s}  "
            f"pts={len(pts):>5d}  bbox=["
            f"{pts.max(axis=0)[0]-pts.min(axis=0)[0]:.2f}, "
            f"{pts.max(axis=0)[1]-pts.min(axis=0)[1]:.2f}, "
            f"{pts.max(axis=0)[2]-pts.min(axis=0)[2]:.2f}]"
        )

    return paths


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic archaeological fragments for GeoTransformer training",
    )
    p.add_argument(
        "--num", type=int, default=30,
        help="Number of fragments to generate (default: 30)",
    )
    p.add_argument(
        "--output-dir", type=str, default="synthetic_fragments",
        help="Output directory for PLY files (default: synthetic_fragments/)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--types", type=str, nargs="+",
        default=["pottery", "fresco", "rubble"],
        help="Fragment types to generate: pottery fresco rubble (default: all three)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Generating {args.num} synthetic fragments into {args.output_dir}/")
    print(f"  Types: {', '.join(args.types)}")
    print(f"  Seed:  {args.seed}\n")

    paths = generate_fragments(
        num=args.num,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        types=tuple(args.types),
    )
    print(f"\nDone. {len(paths)} fragments saved to {args.output_dir}/")
    print(f"Train with:")
    print(f"  python scripts/train_geotransformer.py {args.output_dir}/FRAG_*.ply "
          "RPf_00577_ds.ply RPf_00579_ds.ply")


if __name__ == "__main__":
    main()
