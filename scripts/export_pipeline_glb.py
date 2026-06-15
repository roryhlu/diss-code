#!/usr/bin/env python3
"""
RePAIR pipeline → single GLB file for Blender import.

Runs the full perception pipeline and exports all stages as a single
glTF 2.0 binary (.glb) file with per-vertex colours baked in.
Zero Blender scripting — just File → Import → glTF 2.0.

Pipeline stages exported
------------------------
  CAD_Model     — Original fragment mesh (grey)
  Scene_Cloud   — Noisy scene with ground-truth perturbation (blue)
  Aligned_Cloud — TEASER++ registration result (green)
  Variance      — MC Dropout epistemic uncertainty (blue→red)
  Grasps_OK     — CVaR-accepted grasp contacts (emissive green)
  Grasps_FAIL   — CVaR-rejected grasp contacts (emissive red)

Usage
-----
    python scripts/export_pipeline_glb.py RPf_00577.obj \\
        --model checkpoints_146/geotransformer_best.pt \\
        --output results/blender_scene/

    Then in Blender: File → Import → glTF 2.0 → repair_pipeline.glb
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


# ── Inline SE(3) utilities ───────────────────────────────────────────


def _random_se3(max_angle_deg=25.0, max_translation=0.03, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1); th = rng.uniform(0, 2*np.pi)
    s = np.sqrt(max(0, 1-z*z)); axis = np.array([s*np.cos(th), s*np.sin(th), z])
    angle = rng.uniform(0, np.deg2rad(max_angle_deg))
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    z2 = rng.uniform(-1,1); th2 = rng.uniform(0,2*np.pi)
    s2 = np.sqrt(max(0, 1-z2*z2)); d = np.array([s2*np.cos(th2), s2*np.sin(th2), z2])
    tn = rng.uniform(0, max_translation); t = d*tn
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    return T, float(np.rad2deg(angle)), float(tn)


def _transform_points(T, pts):
    return pts @ T[:3,:3].T + T[:3,3]


# ── GLB writer (pure Python, no dependencies) ───────────────────────


def write_glb(output_path: str, meshes: list[dict]) -> None:
    """
    Write a glTF 2.0 binary (.glb) file containing multiple point-cloud meshes.

    Each mesh dict: {'name': str, 'points': (N,3) float64, 'colors': (N,3) uint8}

    Uses GLTF POINTS primitive mode so Blender renders each vertex as a dot.
    """
    # Build binary buffer: interleave position (float32) + color (uint8) per vertex
    mesh_offsets = []  # byte offset of each mesh in the buffer
    bin_data = bytearray()
    accessors = []
    buffer_views = []
    meshes_gltf = []
    nodes = []

    total_offset = 0
    for m_idx, m in enumerate(meshes):
        pts = m['points'].astype(np.float32)
        colours = m['colors']
        if np.issubdtype(colours.dtype, np.floating):
            colours = np.clip(colours * 255, 0, 255).astype(np.uint8)
        colours = colours.astype(np.uint8)
        n = len(pts)

        # Interleave: position (12 bytes) + color (3 bytes) + 1 pad byte = 16 bytes aligned
        row = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('pad', 'u1')])
        row['x'] = pts[:, 0]; row['y'] = pts[:, 1]; row['z'] = pts[:, 2]
        row['r'] = colours[:, 0]; row['g'] = colours[:, 1]; row['b'] = colours[:, 2]
        row_bytes = row.tobytes()

        byte_offset = total_offset
        bin_data.extend(row_bytes)
        mesh_offsets.append(byte_offset)
        total_offset += len(row_bytes)

        byte_length = len(row_bytes)
        # Accessors: position and color
        acc_pos_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views),
            "componentType": 5126,  # FLOAT
            "count": n,
            "type": "VEC3",
            "max": [float(pts[:,0].max()), float(pts[:,1].max()), float(pts[:,2].max())],
            "min": [float(pts[:,0].min()), float(pts[:,1].min()), float(pts[:,2].min())],
        })
        # Buffer view for positions (12 bytes per vertex, stride 16)
        bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
            "byteStride": 16,
            "target": 34962,  # ARRAY_BUFFER
        })

        acc_col_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views),
            "componentType": 5121,  # UNSIGNED_BYTE
            "count": n,
            "type": "VEC3",
            "normalized": True,
        })
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset + 12,
            "byteLength": byte_length,
            "byteStride": 16,
            "target": 34962,
        })

        # Mesh
        mesh_idx = len(meshes_gltf)
        meshes_gltf.append({
            "primitives": [{
                "attributes": {"POSITION": acc_pos_idx, "COLOR_0": acc_col_idx},
                "mode": 0,  # POINTS
                "material": m_idx,
            }]
        })

        nodes.append({"mesh": mesh_idx, "name": m['name']})

        # Material
        r, g, b = m.get('material_color', [0.5, 0.5, 0.5])
        base_col = [r, g, b, 1.0]
        # Check for special materials (grasps get emission)
        extras = {}
        if 'OK' in m['name']:
            extras = {"emissiveFactor": [0.0, 0.3, 0.05]}
        elif 'FAIL' in m['name']:
            extras = {"emissiveFactor": [0.3, 0.0, 0.0]}

        accessors.append({
            # Dummy accessor for material (not strictly needed but keeps indices aligned)
            "bufferView": 0, "componentType": 5126, "count": 0, "type": "SCALAR",
        })

    # Materials
    materials = []
    for m in meshes:
        r, g, b = m.get('material_color', [0.5, 0.5, 0.5])
        mat = {"pbrMetallicRoughness": {"baseColorFactor": [r, g, b, 1.0], "roughnessFactor": 0.4, "metallicFactor": 0.0}}
        if 'OK' in m['name']:
            mat["emissiveFactor"] = [0.0, 0.3, 0.05]
        elif 'FAIL' in m['name']:
            mat["emissiveFactor"] = [0.3, 0.0, 0.0]
        materials.append(mat)

    # GLTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "RePAIR pipeline"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes_gltf,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_data)}],
        "materials": materials,
    }
    json_str = json.dumps(gltf, separators=(',', ':'))

    # Pad JSON to 4-byte alignment with spaces
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')

    # GLB header
    glb_header = struct.pack('<I', 0x46546C67)  # magic 'glTF'
    glb_header += struct.pack('<I', 2)           # version 2
    glb_header += struct.pack('<I', 12 + 8 + len(json_bytes) + 8 + len(bin_data))  # total length

    # JSON chunk
    json_chunk = struct.pack('<I', len(json_bytes))
    json_chunk += struct.pack('<I', 0x4E4F534A)  # chunk type 'JSON'
    json_chunk += json_bytes

    # BIN chunk
    bin_chunk = struct.pack('<I', len(bin_data))
    bin_chunk += struct.pack('<I', 0x004E4942)  # chunk type 'BIN\0'
    bin_chunk += bin_data

    with open(output_path, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk)
        f.write(bin_chunk)

    print(f"  GLB written: {output_path} ({len(glb_header)+len(json_chunk)+len(bin_chunk)} bytes)")


# ── Pipeline ─────────────────────────────────────────────────────────


def main() -> None:
    import open3d as o3d

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # ── 1. Load fragment ──
    print("=== 1. Loading fragment ===")
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split()
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
        model_pts = np.array(verts, dtype=np.float64)
    else:
        pcd = o3d.io.read_point_cloud(args.fragment)
        model_pts = np.asarray(pcd.points, dtype=np.float64)
    centroid = model_pts.mean(axis=0)
    model_pts -= centroid
    print(f"  {len(model_pts):,} points, centred")

    # ── 2. Scene cloud ──
    print("\n=== 2. Generating scene cloud ===")
    T_gt, rot_deg, t_norm = _random_se3(args.max_angle, args.max_translation, seed)
    print(f"  Rotation: {rot_deg:.1f}°, Translation: {t_norm:.4f}m")
    scene_pts = _transform_points(T_gt, model_pts)

    # ── 3. TEASER++ ──
    print("\n=== 3. TEASER++ registration ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import register_teaser, TeaserParams

    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts))
    model_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts))
    t0 = time.perf_counter()
    result = register_teaser(scene_pcd, model_pcd, TeaserParams(
        c_threshold=0.005, noise_bound=0.001, fpfh_radius=0.035, ratio_threshold=0.9))
    elapsed = time.perf_counter() - t0
    T_est = np.asarray(result.T, dtype=np.float64)
    aligned_pts = _transform_points(T_est, scene_pts)
    print(f"  {result.rotation_angle_deg:.2f}° rot, {result.translation_norm*1000:.1f}mm trans ({elapsed:.1f}s)")

    # ── 4. MC Dropout variance ──
    print("\n=== 4. MC Dropout variance ===")
    variance_pts = model_pts.copy()
    variance_colors = np.full((len(model_pts), 3), [128, 128, 128], dtype=np.uint8)

    if args.model and Path(args.model).exists():
        from uncertainty.geotransformer import GeoTransformer
        from uncertainty.mc_inference import run_mc_passes
        from uncertainty.pose_covariance import variance_to_rgb
        import torch

        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        model = GeoTransformer(in_channels=6, embed_dim=128, num_heads=4,
                               num_layers=4, bottleneck_dropout=args.dropout_rate)
        model.load_state_dict(ckpt["model_state_dict"])
        model.set_mc_mode(True); model.eval()
        cent = np.array(ckpt.get("centroids", [[0,0,0]])[0])
        scl = float(ckpt.get("scales", [1.0])[0])
        norms = np.zeros_like(model_pts)
        data = np.column_stack([(model_pts - cent) / scl, norms])
        data_t = torch.from_numpy(data.astype(np.float32))
        mean_t, var_t = run_mc_passes(model, data_t, T=args.mc_passes, batch_size=4096, device="cpu", verbose=True)
        variance_pts = mean_t.numpy() * scl + cent
        var_np = var_t.numpy() * (scl**2)
        variance_colors = (variance_to_rgb(var_np) * 255).astype(np.uint8)
        print(f"  σ² mean={var_np.mean():.1f}, range=[{var_np.min():.1f}, {var_np.max():.1f}]")
    else:
        print("  Model not found — using grey placeholder")

    # ── 5. CVaR grasp candidates ──
    print("\n=== 5. Generating grasp spheres ===")
    from scipy.spatial import cKDTree

    # Estimate normals
    k = min(30, len(model_pts))
    tree = cKDTree(model_pts); _, idx = tree.query(model_pts, k=k)
    neighbours = model_pts[idx]; mu_n = neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", neighbours-mu_n, neighbours-mu_n)/(k-1)
    _, eigvecs = np.linalg.eigh(cov)
    mesh_n = eigvecs[:,:,0].copy()
    dot = np.sum(mesh_n * (-model_pts), axis=1)  # inward toward origin-centred centroid
    mesh_n[dot<0] *= -1.0
    ns = np.linalg.norm(mesh_n, axis=1, keepdims=True); ns[ns<1e-12]=1.0; mesh_n/=ns

    # Find antipodal pairs
    rng = np.random.default_rng(seed)
    cos_a = np.cos(np.arctan(args.mu)); cos_min = cos_a*0.5
    accepted, rejected = [], []
    for _ in range(15*100):
        if len(accepted)+len(rejected) >= 15: break
        i=rng.integers(0,len(model_pts)); j=rng.integers(0,len(model_pts))
        if i==j: continue
        d=model_pts[j]-model_pts[i]; dist=np.linalg.norm(d)
        if dist<1e-9: continue
        dh=d/dist; s1=float(np.dot(dh,mesh_n[i])); s2=float(np.dot(-dh,mesh_n[j]))
        if s1>=cos_min and s2>=cos_min:
            (accepted if s1>=cos_a-1e-9 and s2>=cos_a-1e-9 else rejected).append((model_pts[i],model_pts[j]))

    print(f"  {len(accepted)} accepted, {len(rejected)} rejected")

    # Generate grasp sphere point clouds
    def _sphere_clouds(pairs, n_pts=200, radius=0.003):
        pts, cols = [], []
        for c1,c2 in pairs:
            for c in [c1,c2]:
                s = rng.normal(0,radius,(n_pts,3))+c
                pts.append(s)
        return np.vstack(pts) if pts else np.empty((0,3))

    ok_pts = _sphere_clouds(accepted)
    fail_pts = _sphere_clouds(rejected[:10])

    # ── 6. Build GLB ──
    print("\n=== 6. Writing GLB ===")
    meshes = [
        {'name':'CAD_Model',      'points':model_pts,     'colors':np.full((len(model_pts),3),[180,180,180],dtype=np.uint8), 'material_color':[0.7,0.7,0.7]},
        {'name':'Scene_Cloud',    'points':scene_pts,     'colors':np.full((len(scene_pts),3),[25,75,230],dtype=np.uint8),   'material_color':[0.1,0.3,0.9]},
        {'name':'Aligned_Cloud',  'points':aligned_pts,   'colors':np.full((len(aligned_pts),3),[0,200,50],dtype=np.uint8),  'material_color':[0.0,0.8,0.2]},
        {'name':'Variance',       'points':variance_pts,  'colors':variance_colors,                                           'material_color':[0.6,0.6,0.6]},
    ]
    if len(ok_pts) > 0:
        meshes.append({'name':'Grasps_OK',   'points':ok_pts,   'colors':np.full((len(ok_pts),3),[0,230,50],dtype=np.uint8),  'material_color':[0.0,0.9,0.2]})
    if len(fail_pts) > 0:
        meshes.append({'name':'Grasps_FAIL', 'points':fail_pts, 'colors':np.full((len(fail_pts),3),[230,0,0],dtype=np.uint8),  'material_color':[0.9,0.0,0.0]})

    glb_path = output_dir / "repair_pipeline.glb"
    write_glb(str(glb_path), meshes)

    print(f"\n{'='*60}")
    print(f"  Done → {glb_path}")
    print(f"  Blender: File → Import → glTF 2.0 → repair_pipeline.glb")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR pipeline → GLB for Blender")
    p.add_argument("fragment", help="OBJ or PLY file")
    p.add_argument("--output", default="results/blender_scene", help="Output directory")
    p.add_argument("--model", default="checkpoints_146/geotransformer_best.pt", help="Checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-angle", type=float, default=25.0)
    p.add_argument("--max-translation", type=float, default=0.03)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--mc-passes", type=int, default=30)
    p.add_argument("--dropout-rate", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    main()
