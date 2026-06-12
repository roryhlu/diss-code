#!/usr/bin/env python3
"""
Synthesize cluttered tabletop scenes from individual RePAIR fragments.

Places multiple fragments at random non-overlapping positions on a
virtual table plane with random orientations, producing realistic
multi-fragment scenes for testing the segmentation and registration
pipeline end-to-end.

Table
-----
  Synthetic flat plane with configurable dimensions and Gaussian
  height noise to simulate a real table surface.

Fragment placement
------------------
  Each fragment is placed at a random (x, y) position on the table
  and rotated randomly about the vertical axis.  A rejection-sampling
  strategy prevents overlap: new placements are checked against
  existing ones using bounding-box distance.

Output
------
  <stem>_scene.ply         — combined cluttered scene
  <stem>_scene.json        — ground-truth fragment poses + metadata

Usage
------
    python scripts/synthesize_cluttered_scene.py fragment1.ply fragment2.ply ... \\
        --num-fragments 8 --output-dir cluttered_scenes/

    # Use pre-processed fragments from repair_fragments_ds/
    python scripts/synthesize_cluttered_scene.py repair_fragments_ds/RPf_0053*_ds.ply \\
        --num-fragments 10 --table-width 0.5 --table-depth 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesize cluttered tabletop scenes from individual fragments",
    )
    p.add_argument("fragments", nargs="+", help="Fragment PLY files to place on table")
    p.add_argument("--num-fragments", type=int, default=8,
                   help="Number of fragments to place (randomly selected, default: 8)")
    p.add_argument("--output-dir", type=str, default="cluttered_scenes",
                   help="Output directory (default: cluttered_scenes/)")
    p.add_argument("--scene-name", type=str, default=None,
                   help="Scene name stem (default: auto-generated)")
    p.add_argument("--table-width", type=float, default=0.4,
                   help="Table width in metres (default: 0.4)")
    p.add_argument("--table-depth", type=float, default=0.3,
                   help="Table depth in metres (default: 0.3)")
    p.add_argument("--table-noise", type=float, default=0.0003,
                   help="Table surface Gaussian noise σ in metres (default: 0.0003)")
    p.add_argument("--table-density", type=int, default=20000,
                   help="Number of points on the table plane (default: 20000)")
    p.add_argument("--table-z", type=float, default=0.0,
                   help="Table Z coordinate (default: 0.0 = origin)")
    p.add_argument("--min-separation", type=float, default=0.01,
                   help="Minimum centre separation between fragments (m, default: 0.01)")
    p.add_argument("--max-place-attempts", type=int, default=500,
                   help="Max rejection-sampling attempts per fragment (default: 500)")
    p.add_argument("--no-table", action="store_true",
                   help="Don't generate table plane (fragments only)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def _make_table(
    width: float,
    depth: float,
    z: float,
    noise_sigma: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a table plane with Gaussian height noise."""
    x = rng.uniform(-width / 2, width / 2, n_points)
    y = rng.uniform(-depth / 2, depth / 2, n_points)
    z_vals = np.full(n_points, z) + rng.normal(0, noise_sigma, n_points)
    return np.column_stack([x, y, z_vals])


def _rot_z(angle: float) -> np.ndarray:
    """SO(3) rotation about Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _fragment_bbox_xy(points: np.ndarray) -> tuple[float, float]:
    """Estimate fragment XY span after placing at origin."""
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.sqrt(extent[0]**2 + extent[1]**2))


def _place_fragment(
    points: np.ndarray,
    rotation_angle: float,
    center_xy: np.ndarray,
    table_z: float,
) -> np.ndarray:
    """Rotate fragment about Z, then translate to table position."""
    pts_centered = points - points.mean(axis=0)  # centre at origin
    R = _rot_z(rotation_angle)
    pts_rotated = pts_centered @ R.T
    # Place on table: XY = centre position, Z = table surface
    z_min = pts_rotated[:, 2].min()
    pts_rotated[:, 2] -= z_min  # sit on XY plane
    pts_rotated[:, 0] += center_xy[0]
    pts_rotated[:, 1] += center_xy[1]
    pts_rotated[:, 2] += table_z
    return pts_rotated


def _check_overlap(
    center: np.ndarray,
    radius: float,
    placed_centers: list[np.ndarray],
    placed_radii: list[float],
    min_separation: float,
) -> bool:
    """Check if a new fragment overlaps with any existing ones."""
    for pc, pr in zip(placed_centers, placed_radii):
        dist = np.linalg.norm(center[:2] - pc[:2])
        if dist < (radius + pr + min_separation):
            return True
    return False


def main() -> None:
    args = parse_args()
    import open3d as o3d

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Select fragments ──
    n_frags = min(args.num_fragments, len(args.fragments))
    chosen = list(rng.choice(args.fragments, n_frags, replace=False))

    # ── Load fragments ──
    print(f"Loading {n_frags} fragments ...")
    fragment_data: list[tuple[str, np.ndarray]] = []
    for path in chosen:
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            print(f"  Warning: {path} has no points — skipping")
            continue
        pts = np.asarray(pcd.points, dtype=np.float64)
        fragment_data.append((Path(path).stem, pts))
        print(f"  {Path(path).name}: {len(pts):,} pts")
    n_frags = len(fragment_data)

    # ── Table ──
    scene_parts: list[np.ndarray] = []
    if not args.no_table:
        print(f"\nGenerating table ({args.table_width}×{args.table_depth}m, "
              f"{args.table_density} pts) ...")
        table_pts = _make_table(
            args.table_width, args.table_depth, args.table_z,
            args.table_noise, args.table_density, rng,
        )
        scene_parts.append(table_pts)

    # ── Place fragments ──
    print(f"\nPlacing {n_frags} fragments on table ...")
    placed_centers: list[np.ndarray] = []
    placed_radii: list[float] = []
    placements: list[dict] = []
    scene_points_list: list[np.ndarray] = []

    for name, pts in fragment_data:
        rot_angle = rng.uniform(0, 2 * np.pi)
        radius = _fragment_bbox_xy(pts) * 0.55  # half the XY span

        placed = False
        for attempt in range(args.max_place_attempts):
            cx = rng.uniform(-args.table_width / 2 * 0.8, args.table_width / 2 * 0.8)
            cy = rng.uniform(-args.table_depth / 2 * 0.8, args.table_depth / 2 * 0.8)
            center = np.array([cx, cy])
            if not _check_overlap(center, radius, placed_centers, placed_radii, args.min_separation):
                placed = True
                break

        if not placed:
            print(f"  {name}: FAILED to place after {args.max_place_attempts} attempts")
            continue

        placed_centers.append(center)
        placed_radii.append(radius)

        # Place fragment at final pose
        placed_pts = _place_fragment(pts, rot_angle, center, args.table_z)
        scene_points_list.append(placed_pts)

        # Compute SE(3) transform for this placement
        R = _rot_z(rot_angle)
        t = np.zeros(3)
        t[:2] = center
        z_min_orig = np.min(pts - pts.mean(axis=0), axis=0)[2]
        t[2] = args.table_z - z_min_orig
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        placements.append({
            "fragment": name,
            "num_points": len(pts),
            "rotation_angle_deg": float(np.rad2deg(rot_angle)),
            "center_xy": [float(center[0]), float(center[1])],
            "radius_m": float(radius),
            "se3_matrix": T.tolist(),
        })
        print(f"  {name}: (x={center[0]*1000:.0f}, y={center[1]*1000:.0f})mm  "
              f"rot={np.rad2deg(rot_angle):.0f}°  r={radius*1000:.0f}mm")

    if not placements:
        raise SystemExit("No fragments could be placed — table too small?")

    # ── Combine scene ──
    all_scene = np.vstack(scene_parts + scene_points_list)
    scene_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(all_scene.astype(np.float64))
    )

    # ── Save ──
    stem = args.scene_name or f"cluttered_{n_frags}frags_seed{args.seed}"
    scene_path = output_dir / f"{stem}_scene.ply"
    o3d.io.write_point_cloud(str(scene_path), scene_pcd)
    print(f"\nScene saved: {scene_path} ({len(all_scene):,} total pts)")

    manifest_path = output_dir / f"{stem}_scene.json"
    manifest = {
        "scene": str(scene_path),
        "num_fragments_placed": len(placements),
        "num_fragments_attempted": n_frags,
        "table_size_m": [args.table_width, args.table_depth],
        "table_z_m": args.table_z,
        "seed": args.seed,
        "placements": placements,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    print(f"\nNext — segment this scene:")
    print(f"  python scripts/segment_cluttered_scene.py {scene_path} "
          f"--output-dir {output_dir}/clusters/")
    print(f"\nThen register each cluster against CAD models:")
    print(f"  python scripts/batch_evaluate.py {output_dir}/clusters/{stem}_cluster_*.ply "
          f"--output results/segmented")


if __name__ == "__main__":
    main()
