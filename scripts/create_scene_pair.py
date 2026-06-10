#!/usr/bin/env python3
"""
Generate a random SE(3)-perturbed scene point cloud from a CAD/object cloud.

Applies a bounded random rotation (Rodrigues on uniform S² axis) and
translation to the input point cloud, then saves the perturbed copy.
The ground-truth SE(3) matrix is printed for downstream ADD-S / Chamfer
validation against recovered transforms.

Usage:
    python scripts/create_scene_pair.py RPf_00577_ds.ply --angle 20 --translation 0.03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from registration.se3_utils import random_se3, transform_points_np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create random SE(3)-perturbed scene cloud for registration testing",
    )
    p.add_argument("input", type=str, help="Input point cloud (PLY, PCD, XYZ)")
    p.add_argument(
        "--angle",
        type=float,
        default=25.0,
        help="Maximum rotation angle in degrees (default: 25)",
    )
    p.add_argument(
        "--translation",
        type=float,
        default=0.03,
        help="Maximum translation magnitude in metres (default: 0.03)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the scene PLY (default: <input_stem>_scene.ply)",
    )
    p.add_argument(
        "--gt-output",
        type=str,
        default=None,
        help="Path to save the 4x4 ground-truth SE(3) matrix as .npy "
        "(default: <output_stem>_gt.npy)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load
    pcd = o3d.io.read_point_cloud(args.input)
    if not pcd.has_points():
        raise ValueError(f"No points loaded from '{args.input}'")
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points from {Path(args.input).name}")

    # Generate random SE(3)
    T_gt, rot_deg, t_norm = random_se3(
        max_angle_deg=args.angle,
        max_translation=args.translation,
        seed=args.seed,
    )

    # Apply transform
    scene_points = transform_points_np(T_gt, points)
    scene_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(scene_points.astype(np.float64))
    )
    if pcd.has_normals():
        orig_normals = np.asarray(pcd.normals)
        rotated_normals = orig_normals @ T_gt[:3, :3].T
        scene_pcd.normals = o3d.utility.Vector3dVector(
            rotated_normals.astype(np.float64)
        )

    # Save
    stem = Path(args.input).stem
    out_path = args.output or f"{stem}_scene.ply"
    o3d.io.write_point_cloud(out_path, scene_pcd)

    # Persist ground-truth SE(3) matrix for downstream evaluation
    gt_out = args.gt_output
    if gt_out is None:
        gt_out = f"{stem}_gt.npy"
    np.save(gt_out, T_gt)

    # Ground truth report
    print(f"\nGround-truth SE(3) — also saved to {gt_out}:")
    print(f"  Rotation angle : {rot_deg:.4f} deg")
    print(f"  Translation    : {t_norm:.6f} m")
    print(f"  SE(3) matrix   :")
    for row in T_gt:
        print(f"    [{row[0]:.8f}  {row[1]:.8f}  {row[2]:.8f}  {row[3]:.8f}]")
    print(f"\nScene cloud saved to: {out_path}")


if __name__ == "__main__":
    main()
