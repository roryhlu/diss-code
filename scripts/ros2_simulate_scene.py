#!/usr/bin/env python3
"""
RePAIR robot simulation scene — tabletop pick-and-place as coloured PLY.

Generates a top-down visualisation of the complete manipulation scenario:
  - Table plane (dark grey)
  - Fragment seated on table (beige)
  - Camera position above and in front (cyan sphere)
  - Camera FOV cone (wireframe lines from camera to fragment)
  - Pre-grasp point 8 cm above fragment (green)
  - Grasp contact point (red)
  - Approach vector arrow (yellow, top-down)

All output as individual coloured ASCII PLY files — import into Blender
via File → Import → Stanford PLY.  Top-down view: press Numpad 7.

Usage
-----
    python scripts/ros2_simulate_scene.py RPf_00577.obj \\
        --output results/simulation/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_ply_ascii(path: str, points: np.ndarray, colours: np.ndarray) -> None:
    n = len(points)
    pts = points.astype(np.float64)
    cls = colours.astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                    f"{cls[i,0]} {cls[i,1]} {cls[i,2]}\n")


def make_table(width=0.4, depth=0.3, z=0.0, n_points=8000):
    rng = np.random.default_rng(0)
    x = rng.uniform(-width/2, width/2, n_points)
    y = rng.uniform(-depth/2, depth/2, n_points)
    return np.column_stack([x, y, np.full(n_points, z)])


def make_cone_lines(apex, base_center, radius=0.06, n_ring=16, n_line=50):
    """Wireframe cone from apex to a circle at base_center."""
    pts, cols = [], []
    # Ring at base
    for i in range(n_ring):
        angle = 2*np.pi*i/n_ring
        ring_pt = base_center + np.array([radius*np.cos(angle), radius*np.sin(angle), 0])
        # Line from apex to ring point
        t = np.linspace(0, 1, n_line)
        line_pts = apex + t[:,None]*(ring_pt - apex)
        pts.append(line_pts)
    return np.vstack(pts)


def load_obj(path):
    verts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float64)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading fragment ===")
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        frag = load_obj(args.fragment)
    elif ext == ".ply":
        # Fast binary PLY reader — avoids Open3D
        with open(args.fragment, 'rb') as fh:
            for line in fh:
                line = line.decode('ascii','ignore')
                if 'element vertex' in line:
                    n = int(line.split()[-1]); break
            while fh.readline().strip() != b'end_header': pass
            raw = np.frombuffer(fh.read(), dtype=np.float64)
        stride = 6 if len(raw) % 6 == 0 else 3
        frag = raw.reshape(-1, stride)[:,:3].astype(np.float64)
    else:
        import open3d as o3d
        frag = np.asarray(o3d.io.read_point_cloud(args.fragment).points, dtype=np.float64)
    centroid = frag.mean(axis=0)
    frag -= centroid
    frag[:,2] -= frag[:,2].min()  # sit on Z=0
    print(f"  {len(frag):,} vertices, centred, seated on table")

    frag_center = frag.mean(axis=0)

    # Positions
    table_z = -0.002
    table = make_table(z=table_z) + np.array([frag_center[0], frag_center[1], 0])

    pre_grasp = np.array([frag_center[0], frag_center[1], 0.08])
    grasp_pos = np.array([frag_center[0], frag_center[1], frag[:,2].max() + 0.005])
    camera_pos = np.array([frag_center[0], frag_center[1], 0.30])

    # Build line for approach arrow (pre-grasp → grasp)
    t = np.linspace(0, 1, 100)
    approach_line = pre_grasp + t[:, None] * (grasp_pos - pre_grasp)
    # Small arrowhead at grasp end
    arrow_dir = grasp_pos - pre_grasp; arrow_dir /= np.linalg.norm(arrow_dir)

    # Cone lines from camera to fragment
    cone = make_cone_lines(camera_pos, grasp_pos, radius=0.05)

    # Write files
    print("\n=== Writing PLY files ===")

    write_ply_ascii(str(output_dir / "00_table.ply"), table,
                    np.full((len(table),3), [80,80,90], np.uint8))
    print("  00_table.ply — dark grey table surface")

    write_ply_ascii(str(output_dir / "01_fragment.ply"), frag,
                    np.full((len(frag),3), [210,180,140], np.uint8))
    print("  01_fragment.ply — beige fragment seated on table")

    write_ply_ascii(str(output_dir / "02_camera.ply"),
                    np.array([camera_pos]),
                    np.full((1,3), [0,220,255], np.uint8))
    print("  02_camera.ply — cyan camera position (30cm above)")

    write_ply_ascii(str(output_dir / "03_camera_cone.ply"), cone,
                    np.full((len(cone),3), [100,180,255], np.uint8))
    print("  03_camera_cone.ply — blue camera FOV cone")

    write_ply_ascii(str(output_dir / "04_pre_grasp.ply"),
                    np.array([pre_grasp]),
                    np.full((1,3), [0,255,80], np.uint8))
    print("  04_pre_grasp.ply — green pre-grasp point (8cm above)")

    write_ply_ascii(str(output_dir / "05_approach_arrow.ply"), approach_line,
                    np.full((100,3), [255,200,50], np.uint8))
    print("  05_approach_arrow.ply — yellow approach vector")

    write_ply_ascii(str(output_dir / "06_grasp_point.ply"),
                    np.array([grasp_pos]),
                    np.full((1,3), [255,40,40], np.uint8))
    print("  06_grasp_point.ply — red grasp contact point")

    (output_dir / "README.txt").write_text("""ROBOT SIMULATION SCENE

Blender: File → Import → Stanford PLY → import all 7 files
Press Numpad 7 for TOP-DOWN view (camera looking down at table)
Press Numpad 1 for FRONT view
Press Numpad 3 for SIDE view

Scene layout (top-down):
  ┌─────────────────────────┐
  │     grey table          │
  │   ┌───────────┐         │
  │   │  beige    │         │  ← fragment on table
  │   │ fragment  │         │
  │   │   ● red   │         │  ← grasp point
  │   │   │ green │         │  ← pre-grasp (above)
  │   └───────────┘         │
  │          ┌ cyan         │  ← camera (30cm up)
  └─────────────────────────┘

The yellow arrow shows the robot approach direction (top-down).
The blue cone shows the camera field of view.
""")

    print(f"\n{'='*60}")
    print(f"  Done — 7 PLY files in {output_dir}/")
    print(f"  Blender → Import → Stanford PLY → select all files")
    print(f"  Press Numpad 7 for TOP-DOWN view")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR robot simulation scene (ASCII PLY)")
    p.add_argument("fragment", help="OBJ or PLY file")
    p.add_argument("--output", default="results/simulation", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    main()
