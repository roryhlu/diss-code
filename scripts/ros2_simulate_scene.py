#!/usr/bin/env python3
"""
RePAIR robot simulation visualisation — tabletop scene for Blender.

Generates a GLB file showing the complete pick-and-place scenario:
  - Table plane (grey, semi-transparent suggestion)
  - Fragment (your TEASER++ registered pose or centred mesh)
  - Camera position (small sphere above the scene)
  - Grasp approach vectors (arrows from pre-grasp to grasp)
  - Grasp contact points with large coloured spheres
  - Robot base marker

This runs WITHOUT ROS2 or MoveIt2 — pure Python, outputs one GLB
for Blender.  Use it to validate the setup BEFORE running physical
hardware experiments.

Usage
-----
    python scripts/ros2_simulate_scene.py RPf_00577.obj \\
        --output results/simulation/
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ── GLB writer (inline — same as export_pipeline_glb) ───────────────


def _write_glb(path: str, meshes: list[dict]) -> None:
    buffer_views, accessors, meshes_gltf, nodes, materials, bin_data = [], [], [], [], [], bytearray()
    byte_offset = 0
    mat_cache = {}

    for m in meshes:
        pts = m['points'].astype(np.float32)
        cols = m['colors']
        if np.issubdtype(cols.dtype, np.floating):
            cols = np.clip(cols*255,0,255).astype(np.uint8)
        cols = cols.astype(np.uint8)
        n = len(pts)
        if n == 0:
            continue

        row = np.zeros(n, dtype=[('x','f4'),('y','f4'),('z','f4'),
                                  ('r','u1'),('g','u1'),('b','u1'),('pad','u1')])
        row['x']=pts[:,0]; row['y']=pts[:,1]; row['z']=pts[:,2]
        row['r']=cols[:,0]; row['g']=cols[:,1]; row['b']=cols[:,2]
        row_bytes = row.tobytes()

        acc_pos = len(accessors)
        accessors.append({"bufferView":len(buffer_views),"componentType":5126,"count":n,
            "type":"VEC3","max":[float(pts[:,0].max()),float(pts[:,1].max()),float(pts[:,2].max())],
            "min":[float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,2].min())]})
        buffer_views.append({"buffer":0,"byteOffset":byte_offset,"byteLength":len(row_bytes),
                             "byteStride":16,"target":34962})
        acc_col = len(accessors)
        accessors.append({"bufferView":len(buffer_views),"componentType":5121,"count":n,
                          "type":"VEC3","normalized":True})
        buffer_views.append({"buffer":0,"byteOffset":byte_offset+12,"byteLength":len(row_bytes),
                             "byteStride":16,"target":34962})
        bin_data.extend(row_bytes); byte_offset += len(row_bytes)

        mk = m.get('emat','')
        if mk not in mat_cache:
            r,g,b = m.get('material_color',[0.5,0.5,0.5])
            mat = {"pbrMetallicRoughness":{"baseColorFactor":[r,g,b,1.0],
                    "roughnessFactor":0.4,"metallicFactor":0.0},
                   "extensions":{"KHR_materials_unlit":{}}}
            if m.get('emissive'):
                mat["emissiveFactor"] = m['emissive']; del mat["extensions"]
            mat_cache[mk] = len(materials)
            materials.append(mat)
        mesh_idx = len(meshes_gltf)
        meshes_gltf.append({"primitives":[{"attributes":{"POSITION":acc_pos,"COLOR_0":acc_col},
                            "mode":0,"material":mat_cache[mk]}]})
        nodes.append({"mesh":mesh_idx,"name":m['name']})

    gltf = {"asset":{"version":"2.0","generator":"RePAIR simulation"},
            "scene":0,"scenes":[{"nodes":list(range(len(nodes)))}],
            "nodes":nodes,"meshes":meshes_gltf,"accessors":accessors,
            "bufferViews":buffer_views,"buffers":[{"byteLength":len(bin_data)}],
            "materials":materials,"extensionsUsed":["KHR_materials_unlit"],
            "extensionsRequired":["KHR_materials_unlit"]}
    js = json.dumps(gltf, separators=(',',':'))
    while len(js)%4!=0: js+=' '
    jb = js.encode('utf-8')
    total = 12+8+len(jb)+8+len(bin_data)
    hdr = struct.pack('<III',0x46546C67,2,total)
    with open(path,'wb') as f:
        f.write(hdr); f.write(struct.pack('<II',len(jb),0x4E4F534A)+jb)
        f.write(struct.pack('<II',len(bin_data),0x004E4942)+bin_data)
    print(f"  GLB: {path} ({total/(1024*1024):.1f} MB, {len(nodes)} meshes)")


# ── Scene builders ───────────────────────────────────────────────────


def make_table(width=0.4, depth=0.3, z=0.0, n=5000):
    rng = np.random.default_rng(0)
    x = rng.uniform(-width/2, width/2, n)
    y = rng.uniform(-depth/2, depth/2, n)
    z_pts = np.full(n, z)
    return np.column_stack([x, y, z_pts])


def make_arrow(start, direction, length=0.05, n=100):
    t = np.linspace(0, 1, n)
    points = start + t[:, None] * direction * length
    return points


def load_fragment(path):
    if Path(path).suffix.lower() == ".obj":
        verts = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
        return np.array(verts, dtype=np.float64)
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fragment
    print("=== Loading fragment ===")
    frag_pts = load_fragment(args.fragment)
    centroid = frag_pts.mean(axis=0)
    frag_pts -= centroid
    frag_z_min = frag_pts[:,2].min()
    print(f"  {len(frag_pts):,} vertices, centred")

    # Place fragment on table
    table_z = 0.0
    table_pts = make_table(z=table_z)
    frag_on_table = frag_pts.copy()
    frag_on_table[:,2] -= frag_z_min  # sit on surface

    # Pre-grasp and grasp poses (top-down)
    frag_center = frag_on_table.mean(axis=0)
    approach_z = 0.08  # 8cm above table
    grasp_z = frag_on_table[:,2].max()  # just above fragment top

    pre_grasp = np.array([frag_center[0], frag_center[1], approach_z])
    grasp_pos = np.array([frag_center[0], frag_center[1], grasp_z + 0.005])

    # Camera position (above, looking down)
    camera_pos = np.array([frag_center[0], frag_center[1] + 0.15, 0.35])

    # Build all scene elements
    print("\n=== Building simulation scene ===")
    meshes = [
        {'name':'Table',       'points':table_pts,      'colors':np.full((len(table_pts),3),[100,100,110],dtype=np.uint8),  'material_color':[0.40,0.40,0.45], 'emat':'table'},
        {'name':'Fragment',    'points':frag_on_table,   'colors':np.full((len(frag_on_table),3),[210,180,140],dtype=np.uint8),'material_color':[0.82,0.70,0.55], 'emat':'frag'},
        {'name':'Grasp_Point', 'points':np.array([grasp_pos]),'colors':np.full((1,3),[255,50,50],dtype=np.uint8),            'material_color':[1.0,0.2,0.2],     'emat':'grasp', 'emissive':[0.5,0.0,0.0]},
        {'name':'Pre_Grasp',   'points':np.array([pre_grasp]),'colors':np.full((1,3),[50,255,50],dtype=np.uint8),            'material_color':[0.2,1.0,0.2],     'emat':'pre',   'emissive':[0.0,0.5,0.0]},
        {'name':'ApproachArrow', 'points':make_arrow(pre_grasp, grasp_pos-pre_grasp, length=1.0, n=200),
         'colors':np.full((200,3),[255,200,50],dtype=np.uint8), 'material_color':[1.0,0.8,0.2], 'emat':'arrow'},
        {'name':'Camera',      'points':np.array([camera_pos]),'colors':np.full((1,3),[0,200,255],dtype=np.uint8),            'material_color':[0.0,0.8,1.0],     'emat':'cam',   'emissive':[0.0,0.3,0.5]},
        {'name':'CameraFOV',   'points':make_arrow(camera_pos, grasp_pos-camera_pos, length=1.0, n=100),
         'colors':np.full((100,3),[200,200,255],dtype=np.uint8), 'material_color':[0.8,0.8,1.0], 'emat':'fov'},
    ]

    glb_path = output_dir / "robot_simulation.glb"
    _write_glb(str(glb_path), meshes)

    print(f"\n{'='*60}")
    print(f"  Done → {glb_path}")
    print(f"  Blender: File → Import → glTF 2.0 → robot_simulation.glb")
    print(f"\n  Scene shows:")
    print(f"    Table (grey), Fragment (beige) sitting on top")
    print(f"    Pre-grasp point (green, 8cm above)")
    print(f"    Grasp point (red, just above fragment)")
    print(f"    Approach arrow (yellow, from pre-grasp down to grasp)")
    print(f"    Camera position (cyan, above and in front)")
    print(f"    Camera FOV line (light blue, camera→fragment)")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR robot simulation scene for Blender")
    p.add_argument("fragment", help="OBJ or PLY file")
    p.add_argument("--output", default="results/simulation", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    main()
