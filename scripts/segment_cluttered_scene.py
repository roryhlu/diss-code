#!/usr/bin/env python3
"""
Cluttered scene segmentation for RePAIR tabletop fragments.

Separates a multi-fragment scene point cloud into individual fragment
clusters via table-plane removal and DBSCAN clustering.  Each cluster
is saved as a PLY file for downstream registration and grasping.

Pipeline
--------
  1. RANSAC plane fitting → detect and remove the table surface.
  2. Statistical outlier removal → clean sensor noise.
  3. DBSCAN clustering → group remaining points into fragments.
  4. Size filtering → reject tiny clusters (noise) and huge ones
     (table remnants, merged fragments).
  5. Save each cluster as a PLY file + manifest JSON.

Usage
-----
    python scripts/segment_cluttered_scene.py scene.ply --output-dir clusters/

    # With custom parameters
    python scripts/segment_cluttered_scene.py scene.ply \
        --plane-distance 0.005 --cluster-eps 0.008 --min-points 200 \
        --output-dir clusters/
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
        description="Segment a cluttered tabletop scene into individual fragments",
    )
    p.add_argument("scene", type=str, help="Scene point cloud (PLY, PCD)")
    p.add_argument("--output-dir", type=str, default="segmented_clusters",
                   help="Output directory for cluster PLY files")
    p.add_argument("--plane-distance", type=float, default=0.005,
                   help="Max distance to table plane for RANSAC inlier (m, default: 0.005)")
    p.add_argument("--plane-n-iterations", type=int, default=1000,
                   help="RANSAC iterations for plane fitting (default: 1000)")
    p.add_argument("--cluster-eps", type=float, default=0.008,
                   help="DBSCAN eps — max distance between cluster neighbours (m, default: 0.008)")
    p.add_argument("--min-points", type=int, default=100,
                   help="Minimum points per cluster (DBSCAN min_samples, default: 100)")
    p.add_argument("--max-points", type=int, default=50000,
                   help="Maximum points per cluster — larger clusters are re-clustered (default: 50000)")
    p.add_argument("--min-cluster-size", type=int, default=50,
                   help="Reject clusters with fewer than this many points (default: 50)")
    p.add_argument("--no-statistical", action="store_true",
                   help="Skip statistical outlier removal")
    p.add_argument("--keep-table", action="store_true",
                   help="Keep table points in output (don't remove plane)")
    return p.parse_args()


def remove_table_plane(
    pcd,
    distance_threshold: float,
    n_iterations: int,
) -> tuple:
    """Remove the dominant plane (tabletop) from the point cloud."""
    import open3d as o3d

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=n_iterations,
    )
    a, b, c, d = plane_model  # ax + by + cz + d = 0
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    n_total = len(pcd.points)
    n_plane = len(inliers)
    n_above = len(outlier_cloud.points)
    print(f"  Plane removed: {n_plane}/{n_total} pts inliers "
          f"({n_plane/n_total*100:.1f}%)")
    print(f"  Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}] "
          f"offset={d:.4f}m")

    return outlier_cloud, plane_model, normal, n_plane


def statistical_outlier_removal(pcd, nb_neighbours: int = 20, std_ratio: float = 2.0):
    """Remove statistical outliers (sensor noise)."""
    import open3d as o3d

    n_before = len(pcd.points)
    clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbours,
        std_ratio=std_ratio,
    )
    n_removed = n_before - len(clean.points)
    if n_removed > 0:
        print(f"  Statistical: removed {n_removed}/{n_before} noise pts "
              f"({n_removed/n_before*100:.1f}%)")
    return clean


def cluster_fragments(
    pcd,
    eps: float,
    min_points: int,
    max_points: int,
    min_cluster_size: int,
) -> tuple[list, np.ndarray]:
    """
    Cluster points using scipy-based DBSCAN (more reliable than Open3D).

    Returns (list_of_cluster_pcds, labels_array).
    """
    import open3d as o3d
    from sklearn.cluster import DBSCAN  # noqa: E402

    points = np.asarray(pcd.points, dtype=np.float64)

    # scipy DBSCAN uses eps in same units as points
    clustering = DBSCAN(eps=eps, min_samples=min_points, metric="euclidean", n_jobs=-1)
    labels = clustering.fit_predict(points)

    n_clusters = labels.max() + 1
    n_noise = int(np.sum(labels == -1))
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise pts "
          f"(eps={eps}m, min_pts={min_points})")

    clusters = []
    for cid in range(n_clusters):
        mask = labels == cid
        n_pts = mask.sum()
        if n_pts < min_cluster_size:
            labels[mask] = -1  # re-label as noise
            continue
        if n_pts > max_points:
            # Re-cluster oversized cluster with tighter eps
            sub_pcd = pcd.select_by_index(np.where(mask)[0])
            sub_clusters, _ = cluster_fragments(
                sub_pcd, eps * 0.7, min_points, max_points, min_cluster_size,
            )
            clusters.extend(sub_clusters)
            print(f"    Re-clustered oversized cluster {cid} ({n_pts} pts) "
                  f"→ {len(sub_clusters)} sub-clusters")
            continue
        cluster_pcd = pcd.select_by_index(np.where(mask)[0])
        clusters.append((cid, n_pts, cluster_pcd))

    return clusters, labels


def save_clusters(
    clusters: list[tuple[int, int]],
    output_dir: Path,
    stem: str,
) -> list[dict]:
    """Save each cluster as a PLY file and return manifest entries."""
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for rank, (cid, n_pts, cluster_pcd) in enumerate(clusters):
        fname = f"{stem}_cluster_{rank:03d}.ply"
        path = output_dir / fname
        o3d.io.write_point_cloud(str(path), cluster_pcd)

        pts = np.asarray(cluster_pcd.points)
        bbox = pts.max(axis=0) - pts.min(axis=0)
        centroid = pts.mean(axis=0)

        entry = {
            "file": str(path),
            "cluster_id": int(cid),
            "rank": rank,
            "num_points": int(n_pts),
            "bbox_m": [float(b) for b in bbox],
            "bbox_diag_m": float(np.linalg.norm(bbox)),
            "centroid": [float(c) for c in centroid],
        }
        manifest.append(entry)
        print(f"    Cluster {rank}: {n_pts:>6d} pts  "
              f"bbox=[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}]m  →  {fname}")

    return manifest


def main() -> None:
    args = parse_args()
    import open3d as o3d

    stem = Path(args.scene).stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──
    print(f"=== Loading scene: {args.scene}")
    t0 = time.perf_counter()
    pcd = o3d.io.read_point_cloud(args.scene)
    if not pcd.has_points():
        raise SystemExit(f"No points in '{args.scene}'")
    n_original = len(pcd.points)
    print(f"  {n_original:,} points loaded")

    # ── 2. Table removal ──
    if not args.keep_table:
        print("\n=== Table-plane removal (RANSAC)")
        pcd, plane_model, plane_normal, n_plane = remove_table_plane(
            pcd, args.plane_distance, args.plane_n_iterations,
        )
        if len(pcd.points) == 0:
            raise SystemExit("All points classified as table — nothing to segment.")

    # ── 3. Statistical outlier removal ──
    if not args.no_statistical:
        print("\n=== Statistical outlier removal")
        pcd = statistical_outlier_removal(pcd)

    if len(pcd.points) < args.min_points:
        raise SystemExit(
            f"Only {len(pcd.points)} points after cleaning — "
            "not enough for clustering.  Try smaller --plane-distance."
        )

    # ── Auto-scale eps if not overridden ──
    eps = args.cluster_eps
    if eps <= 0.01:  # likely a default, auto-detect
        pts_for_scale = np.asarray(pcd.points, dtype=np.float64)
        # Sample for speed (max 5000)
        if len(pts_for_scale) > 5000:
            idx = np.random.default_rng(42).choice(len(pts_for_scale), 5000, replace=False)
            pts_for_scale = pts_for_scale[idx]
        from scipy.spatial import cKDTree as _KD
        tree = _KD(pts_for_scale)
        dists, _ = tree.query(pts_for_scale, k=2)
        med_nn = float(np.median(dists[:, 1]))
        eps = max(eps, med_nn * 3.0)
        print(f"  Auto-scaled eps: {eps:.6f} (median NN = {med_nn:.6f})")

    # ── 4. Clustering ──
    print("\n=== DBSCAN clustering")
    clusters, labels = cluster_fragments(
        pcd, eps, args.min_points,
        args.max_points, args.min_cluster_size,
    )

    if not clusters:
        raise SystemExit(
            "No clusters found.  Check --cluster-eps and --min-points."
        )

    # Sort by size descending
    clusters.sort(key=lambda c: c[1], reverse=True)
    print(f"\n  {len(clusters)} clusters after filtering")

    # ── 5. Save ──
    print("\n=== Saving clusters")
    manifest = save_clusters(clusters, output_dir, stem)

    # ── 6. Manifest ──
    elapsed = time.perf_counter() - t0
    manifest_path = output_dir / f"{stem}_manifest.json"
    summary = {
        "scene": args.scene,
        "num_points_original": n_original,
        "num_clusters": len(clusters),
        "parameters": {
            "plane_distance": args.plane_distance,
            "cluster_eps": eps,
            "min_points": args.min_points,
            "max_points": args.max_points,
            "min_cluster_size": args.min_cluster_size,
        },
        "clusters": manifest,
        "runtime_sec": elapsed,
    }
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"  {len(clusters)} fragments saved to {output_dir}/")
    print(f"  Manifest: {manifest_path}")
    print(f"\nNext — batch-register each cluster against CAD models:")
    print(f"  python scripts/batch_evaluate.py {output_dir}/{stem}_cluster_*.ply "
          f"--output results/segmented")


if __name__ == "__main__":
    main()
