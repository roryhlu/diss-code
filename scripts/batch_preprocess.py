#!/usr/bin/env python3
"""
Batch RePAIR fragment preprocessing — OBJ/PLY → downsampled PLY.

Converts every fragment in a directory to a voxel-downsampled PLY with
PCA-estimated normals, ready for GeoTransformer training or TEASER++
registration.

Handles:
  - OBJ meshes (samples vertices → point cloud)
  - PLY/PCD point clouds (reads directly)
  - Missing normals (estimates via PCA k-NN)

Usage
-----
    # Process a directory of OBJ files
    python scripts/batch_preprocess.py fragments_raw/ --output-dir fragments_ds/

    # Process specific files
    python scripts/batch_preprocess.py RPf_*.obj --output-dir fragments_ds/

    # With custom voxel size (default 0.005 m)
    python scripts/batch_preprocess.py fragments_raw/ --voxel-size 0.003
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


# ── PCA normal estimation (no Open3D dependency) ────────────────────


def estimate_normals(
    points: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """
    Estimate surface normals via PCA on k-nearest neighbours.

    Uses vectorised covariance computation and batched eigendecomposition
    for speed — processes all N points in a single tensor operation.

    Normals are oriented such that the z-component of the mean-centred
    normal is positive (consistent with Open3D convention).

    Args:
        points: (N, 3) float64 point coordinates.
        k:      Number of neighbours (default 30).

    Returns:
        (N, 3) float64 unit normal vectors.
    """
    n_pts = len(points)
    k_eff = min(k, n_pts)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_eff)
    neighbours = points[idx]  # (N, k, 3)

    # Vectorised covariance: C_i = (1/(k-1)) * Σ (p_ij − μ_i)(p_ij − μ_i)^T
    mu = neighbours.mean(axis=1, keepdims=True)  # (N, 1, 3)
    centred = neighbours - mu                     # (N, k, 3)
    cov = np.einsum("nki,nkj->nij", centred, centred) / (k_eff - 1)  # (N, 3, 3)

    # Batched eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs[:, :, 0] = smallest eval

    centroid = points.mean(axis=0)
    normals = eigvecs[:, :, 0].copy()  # (N, 3) — smallest eigenvalue direction

    # Orient consistently relative to centroid
    dot = np.sum(normals * (points - centroid), axis=1)
    normals[dot < 0] *= -1.0

    # Normalise (already unit from eigh, but ensure numerically)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return normals


# ── Voxel downsampling ──────────────────────────────────────────────


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform voxel-grid downsampling via vectorised accumulation.

    Returns (downsampled_points, voxel_ids) — each voxel cell is
    collapsed to its centroid.
    """
    if voxel_size <= 0 or len(points) < 2:
        return points.copy(), np.arange(len(points), dtype=np.int64)

    min_pt = points.min(axis=0)
    voxel_idx = np.floor((points - min_pt) / voxel_size).astype(np.int64)
    # Unique voxel identifier (1D flatten)
    span = voxel_idx.max(axis=0) - voxel_idx.min(axis=0) + 1
    ids = (
        voxel_idx[:, 0] * span[1] * span[2]
        + voxel_idx[:, 1] * span[2]
        + voxel_idx[:, 2]
    )
    # Remap to dense 0..n_voxels-1
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    n_voxels = len(unique_ids)

    # Vectorised accumulation
    down = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(down, inverse, points)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    down /= counts[:, None]

    return down, inverse


# ── OBJ reader (mesh vertices → point cloud) ────────────────────────


def load_obj_as_points(path: str) -> np.ndarray:
    """
    Read an OBJ file and return vertex positions as (N, 3) float64.

    Handles both standard v-lines and large files by streaming.
    """
    vertices: list[tuple[float, float, float]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                ))
    if not vertices:
        raise ValueError(f"No vertices found in '{path}'")
    return np.array(vertices, dtype=np.float64)


# ── PLY reader (streaming, no Open3D) ──────────────────────────────


def load_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read an ASCII or binary PLY file. Returns (points, normals_or_None).

    Supports: property double x, property double y, property double z,
              property double nx, property double ny, property double nz
    """
    with open(path, "rb") as fh:
        header_lines = []
        while True:
            line = fh.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    header = "\n".join(header_lines)
    is_ascii = "ascii" in header_lines[0] if len(header_lines) > 1 else False

    # Parse header for element count and properties
    n_vertices = 0
    props: list[str] = []
    for line in header_lines:
        if line.startswith("element vertex "):
            n_vertices = int(line.split()[-1])
        elif line.startswith("property "):
            props.append(line.split()[-1])

    has_normals = all(n in props for n in ["nx", "ny", "nz"])
    x_idx = props.index("x") if "x" in props else 0
    y_idx = props.index("y") if "y" in props else 1
    z_idx = props.index("z") if "z" in props else 2

    if is_ascii:
        points = np.empty((n_vertices, 3), dtype=np.float64)
        normals = np.empty((n_vertices, 3), dtype=np.float64) if has_normals else None
        with open(path, encoding="ascii") as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            for i, line in enumerate(f):
                if i >= n_vertices:
                    break
                vals = line.split()
                points[i] = [float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])]
                if has_normals:
                    nx_i = props.index("nx")
                    ny_i = props.index("ny")
                    nz_i = props.index("nz")
                    normals[i] = [float(vals[nx_i]), float(vals[ny_i]), float(vals[nz_i])]
        return points, normals

    # Binary — read raw doubles after header
    header_bytes = fh.tell() if hasattr(fh, "tell") else len(header.encode())
    # Re-open for binary read from header end
    header_b = b""
    with open(path, "rb") as fh2:
        while True:
            line = fh2.readline()
            header_b += line
            if line.strip() == b"end_header":
                break
        stride = len(props)
        raw = np.frombuffer(fh2.read(), dtype=np.float64)
        raw = raw.reshape(-1, stride)
        points = raw[:, [x_idx, y_idx, z_idx]].copy()
        normals = raw[:, [props.index("nx"), props.index("ny"), props.index("nz")]].copy() if has_normals else None
    return points[:n_vertices], normals[:n_vertices] if normals is not None else None


def read_fragment(path: str) -> np.ndarray:
    """
    Read a fragment from OBJ, PLY, or PCD file.

    OBJ files: fast line-by-line vertex extraction (no mesh processing).
    PLY/PCD files: native binary reader.

    Returns (N, 3) float64 point positions.
    """
    ext = Path(path).suffix.lower()
    if ext == ".obj":
        return load_obj_as_points(path)
    elif ext in (".ply", ".pcd"):
        pts, _ = load_ply(path)
        return pts
    else:
        raise ValueError(f"Unsupported format: {ext}")


# ── PLY writer ──────────────────────────────────────────────────────


def save_ply(path: str, points: np.ndarray, normals: np.ndarray) -> None:
    """Write binary PLY with x,y,z,nx,ny,nz in double precision."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment Created by batch_preprocess.py\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "end_header\n"
    )
    data = np.column_stack([points, normals]).astype(np.float64)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ── Main processing logic ───────────────────────────────────────────


def process_fragment(
    path: Path,
    output_dir: Path,
    voxel_size: float,
    k_normals: int,
) -> tuple[str, int, int, float]:
    """
    Load, downsample, estimate normals, and save a single fragment.

    Returns (stem, n_input, n_output, runtime_sec).
    """
    t0 = time.perf_counter()
    stem = path.stem

    # Load
    points = read_fragment(str(path))

    # Downsample
    if voxel_size > 0:
        ds_points, _ = voxel_downsample(points, voxel_size)
    else:
        ds_points = points

    # Estimate normals
    normals = estimate_normals(ds_points, k=k_normals)

    # Save
    out_path = output_dir / f"{stem}_ds.ply"
    save_ply(str(out_path), ds_points, normals)

    elapsed = time.perf_counter() - t0
    return stem, len(points), len(ds_points), elapsed


def find_fragment_paths(entries: list[str]) -> list[Path]:
    """Resolve CLI entries to a flat list of fragment files (recursive)."""
    paths: list[Path] = []
    for entry in entries:
        p = Path(entry)
        if p.is_dir():
            for ext in ("*.obj", "*.ply", "*.pcd"):
                paths.extend(sorted(p.rglob(ext)))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Warning: '{entry}' not found — skipping")
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-preprocess RePAIR fragments for GeoTransformer training",
    )
    p.add_argument(
        "fragments", nargs="+",
        help="Fragment files (OBJ/PLY/PCD) or directories containing them",
    )
    p.add_argument(
        "--output-dir", type=str, default="fragments_ds",
        help="Output directory for downsampled PLY files (default: fragments_ds/)",
    )
    p.add_argument(
        "--voxel-size", type=float, default=0.005,
        help="Voxel downsampling size in metres (default: 0.005 = 5 mm)",
    )
    p.add_argument(
        "--k-normals", type=int, default=30,
        help="Neighbour count for PCA normal estimation (default: 30)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fragment_paths = find_fragment_paths(args.fragments)
    if not fragment_paths:
        print("No fragment files found.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(fragment_paths)} fragment(s)")
    print(f"  Voxel size: {args.voxel_size} m")
    print(f"  Normals k:  {args.k_normals}")
    print(f"  Output:     {output_dir}/\n")

    total_in = 0
    total_out = 0
    for i, p in enumerate(fragment_paths):
        stem, n_in, n_out, elapsed = process_fragment(
            p, output_dir, args.voxel_size, args.k_normals,
        )
        total_in += n_in
        total_out += n_out
        pct = (n_out / n_in * 100) if n_in > 0 else 0
        print(
            f"  [{i+1:3d}/{len(fragment_paths)}] {stem:<30s} "
            f"{n_in:>7,d} → {n_out:>6,d} pts ({pct:5.1f}%)  "
            f"{elapsed:.1f}s"
        )

    print(f"\nDone. {total_in:,} → {total_out:,} points across {len(fragment_paths)} fragments")
    print(f"Output: {output_dir}/")
    print(f"\nTrain with:")
    print(f"  python scripts/train_geotransformer.py {output_dir}/*_ds.ply --device cpu")


if __name__ == "__main__":
    main()
