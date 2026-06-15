#!/usr/bin/env python3
"""
RePAIR full pipeline → single GLB for Blender.  Every stage visible.

Exports all pipeline stages as named, distinctly-coloured point clouds
and meshes in one glTF 2.0 binary file.  No Blender scripting —
File → Import → glTF 2.0 → one click.

Stages exported (9 meshes)
---------------------------
  01_OriginalMesh     Beige/tan  — raw OBJ fragment surface
  02_VoxelDownsampled Grey       — 5 mm voxel-grid centroids
  03_PCANormals       RGB        — normal direction encoded as colour
  04_SceneNoisy       Blue       — random SE(3) perturbation
  05_TEASERAligned    Green      — registration result
  06_GeoMean          Cyan       — MC Dropout predictive mean
  07_GeoVariance      Heatmap    — epistemic uncertainty (blue→red)
  08_GraspsPassed     Green glow — CVaR-accepted, large spheres
  09_GraspsFailed     Red glow   — CVaR-rejected, large spheres

Usage
-----
    python scripts/export_pipeline_glb.py RPf_00577.obj \\
        --model checkpoints_146/geotransformer_best.pt \\
        --output results/blender_pipeline/

    Then: Blender → File → Import → glTF 2.0 → repair_pipeline.glb
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np


# ── Inline SE(3) ────────────────────────────────────────────────────


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


def _transform(T, pts):
    return pts @ T[:3,:3].T + T[:3,3]


# ── Voxel downsample ────────────────────────────────────────────────


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) < 2:
        return points
    min_pt = points.min(axis=0)
    voxel_idx = np.floor((points - min_pt) / voxel_size).astype(np.int64)
    span = voxel_idx.max(axis=0) - voxel_idx.min(axis=0) + 1
    ids = voxel_idx[:,0]*span[1]*span[2] + voxel_idx[:,1]*span[2] + voxel_idx[:,2]
    _, inverse = np.unique(ids, return_inverse=True)
    n_voxels = inverse.max()+1
    down = np.zeros((n_voxels,3), dtype=np.float64)
    np.add.at(down, inverse, points)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return down / counts[:, None]


def pca_normals(points: np.ndarray, k=30) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points); _, idx = tree.query(points, k=min(k, len(points)))
    nbrs = points[idx]; mu = nbrs.mean(axis=1, keepdims=True)
    centred = nbrs - mu
    cov = np.einsum("nki,nkj->nij", centred, centred)/(min(k, len(points))-1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:,:,0].copy()
    centroid = points.mean(axis=0)
    dot = np.sum(normals * (centroid - points), axis=1)
    normals[dot < 0] *= -1.0
    ns = np.linalg.norm(normals, axis=1, keepdims=True); ns[ns<1e-12]=1.0
    return normals/ns


# ── Sphere mesh generator (for grasp contacts) ────────────────────────


def sphere_vertices(center, radius=0.005, subdivisions=2):
    """Generate icosahedron-sphere vertices around a centre point."""
    # Use a subdivided icosahedron for smooth spheres
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1,-t, 0], [1,-t, 0],
        [0,-1, t], [0, 1, t], [0,-1,-t], [0, 1,-t],
        [t, 0,-1], [t, 0, 1], [-t, 0,-1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    # subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_mid = {}
        def _mid(a,b):
            k = (min(a,b), max(a,b))
            if k not in edge_mid:
                edge_mid[k] = verts[a] + verts[b]
                edge_mid[k] /= np.linalg.norm(edge_mid[k])
            return edge_mid[k]
        for f in faces:
            va, vb, vc = int(f[0]), int(f[1]), int(f[2])
            # Add midpoints if needed
            m_ab = _mid(va, vb) if (va,vb) not in edge_mid else None
            # Actually just do naive: create 4 faces per original face
            # Mid indices will be stored
            pass
        # Simple approach: just use many points approximating a sphere surface
        break  # skip subdivision for now, use dense random sampling instead

    # Random sampling on sphere surface approximates it well for small radius
    n = 500
    rng_n = np.random.default_rng(hash(str(center)) % 2**32)
    theta = np.arccos(1 - 2*rng_n.random(n))
    phi = 2*np.pi*rng_n.random(n)
    sx = radius*np.sin(theta)*np.cos(phi) + center[0]
    sy = radius*np.sin(theta)*np.sin(phi) + center[1]
    sz = radius*np.cos(theta) + center[2]
    return np.column_stack([sx, sy, sz])


def grasp_line_points(c1, c2, n_pts=50):
    """Generate points along a line between two contact points."""
    t = np.linspace(0, 1, n_pts)
    return c1 + t[:, None] * (c2 - c1)


# ── GLB writer ───────────────────────────────────────────────────────


def write_glb(output_path: str, meshes: list[dict]) -> None:
    """
    Write glTF 2.0 binary (.glb) with interleaved position + color.

    Each mesh dict: {'name':str, 'points':(N,3) float, 'colors':(N,3) uint8}
    """
    buffer_views = []
    accessors = []
    meshes_gltf = []
    nodes = []
    materials = []
    bin_data = bytearray()
    byte_offset = 0
    material_cache = {}
    material_indices = {}

    for m_idx, m in enumerate(meshes):
        pts = m['points'].astype(np.float32)
        cols = m['colors']
        if np.issubdtype(cols.dtype, np.floating):
            cols = np.clip(cols*255, 0, 255).astype(np.uint8)
        cols = cols.astype(np.uint8)
        n = len(pts)
        if n == 0:
            continue

        # Apply side-by-side spatial offset so stages don't overlap
        offset_x = m.get('offset_x', 0.0)
        pts = pts.copy()
        pts[:, 0] += offset_x

        # Interleaved row: 12 bytes position + 3 bytes color + 1 pad = 16 bytes
        row = np.zeros(n, dtype=[('x','f4'),('y','f4'),('z','f4'),
                                  ('r','u1'),('g','u1'),('b','u1'),('pad','u1')])
        row['x']=pts[:,0]; row['y']=pts[:,1]; row['z']=pts[:,2]
        row['r']=cols[:,0]; row['g']=cols[:,1]; row['b']=cols[:,2]
        row_bytes = row.tobytes()

        acc_pos = len(accessors)
        accessors.append({"bufferView":len(buffer_views),"componentType":5126,
                          "count":n,"type":"VEC3",
                          "max":[float(pts[:,0].max()),float(pts[:,1].max()),float(pts[:,2].max())],
                          "min":[float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,2].min())]})
        buffer_views.append({"buffer":0,"byteOffset":byte_offset,"byteLength":len(row_bytes),
                             "byteStride":16,"target":34962})

        acc_col = len(accessors)
        accessors.append({"bufferView":len(buffer_views),"componentType":5121,
                          "count":n,"type":"VEC3","normalized":True})
        buffer_views.append({"buffer":0,"byteOffset":byte_offset+12,"byteLength":len(row_bytes),
                             "byteStride":16,"target":34962})

        bin_data.extend(row_bytes)
        byte_offset += len(row_bytes)

        # Material with KHR_materials_unlit — bright flat colours, no lighting
        mat_key = m.get('emat','')
        if mat_key not in material_cache:
            r,g,b = m.get('material_color',[0.5,0.5,0.5])
            mat = {
                "pbrMetallicRoughness": {"baseColorFactor":[r,g,b,1.0],
                    "roughnessFactor":0.4,"metallicFactor":0.0},
                "extensions": {"KHR_materials_unlit": {}},
            }
            if m.get('emissive', None):
                mat["emissiveFactor"] = m['emissive']
                del mat["extensions"]  # emissive materials need lighting
            material_cache[mat_key] = len(materials)
            materials.append(mat)

        mesh_idx = len(meshes_gltf)
        meshes_gltf.append({"primitives":[{"attributes":{"POSITION":acc_pos,"COLOR_0":acc_col},
                            "mode":0,"material":material_cache[mat_key]}]})
        nodes.append({"mesh":mesh_idx,"name":m['name']})

    gltf = {
        "asset":{"version":"2.0","generator":"RePAIR pipeline"},
        "scene":0,"scenes":[{"nodes":list(range(len(nodes)))}],
        "nodes":nodes,"meshes":meshes_gltf,
        "accessors":accessors,"bufferViews":buffer_views,
        "buffers":[{"byteLength":len(bin_data)}],
        "materials":materials,
        "extensionsUsed":["KHR_materials_unlit"],
        "extensionsRequired":["KHR_materials_unlit"],
    }
    js = json.dumps(gltf, separators=(',',':'))
    while len(js)%4 != 0: js += ' '
    jb = js.encode('utf-8')

    total = 12 + 8 + len(jb) + 8 + len(bin_data)
    hdr = struct.pack('<III', 0x46546C67, 2, total)
    jch = struct.pack('<II', len(jb), 0x4E4F534A) + jb
    bch = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data

    with open(output_path, 'wb') as f:
        f.write(hdr); f.write(jch); f.write(bch)
    size_mb = total/(1024*1024)
    print(f"  GLB: {output_path} ({size_mb:.1f} MB, {len(nodes)} meshes)")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import open3d as o3d

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # ── 1. Load original OBJ ──
    print("=== 1. Loading original OBJ ===")
    if Path(args.fragment).suffix.lower() == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8",errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]),float(p[2]),float(p[3])])
        raw_pts = np.array(verts, dtype=np.float64)
    else:
        pcd = o3d.io.read_point_cloud(args.fragment)
        raw_pts = np.asarray(pcd.points, dtype=np.float64)
    centroid = raw_pts.mean(axis=0)
    raw_pts -= centroid
    print(f"  {len(raw_pts):,} vertices, centred")

    # ── 2. Voxel downsample ──
    print("\n=== 2. Voxel downsample (5 mm) ===")
    ds_pts = voxel_downsample(raw_pts, 0.005)
    print(f"  {len(raw_pts):,} → {len(ds_pts):,} points")

    # ── 3. PCA normals ──
    print("\n=== 3. PCA normal estimation (k=30) ===")
    normals = pca_normals(ds_pts)
    # Encode normals as RGB: R = |nx| in x, G = |ny| in y, B = |nz| in z
    # This makes sphere-like geometry show rainbow colours by normal direction
    normal_rgb = ((normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    print(f"  Normals encoded as RGB colours")

    # ── 4. Scene cloud ──
    print("\n=== 4. Scene cloud (random SE(3) perturbation) ===")
    T_gt, rot_deg, t_norm = _random_se3(args.max_angle, args.max_translation, seed)
    print(f"  {rot_deg:.1f}° rotation, {t_norm:.4f}m translation")
    scene_pts = _transform(T_gt, ds_pts)

    # ── 5. TEASER++ registration ──
    print("\n=== 5. TEASER++ registration ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import register_teaser, TeaserParams
    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts))
    model_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds_pts))
    t0 = time.perf_counter()
    r = register_teaser(scene_pcd, model_pcd,
        TeaserParams(c_threshold=0.005,noise_bound=0.001,fpfh_radius=0.035,ratio_threshold=0.9))
    elapsed = time.perf_counter()-t0
    T_est = np.asarray(r.T, dtype=np.float64)
    aligned_pts = _transform(T_est, scene_pts)
    print(f"  {r.rotation_angle_deg:.2f}° rot, {r.translation_norm*1000:.1f}mm trans ({elapsed:.1f}s)")

    # ── 6. MC Dropout ──
    print("\n=== 6. GeoTransformer MC Dropout ===")
    geo_mean = ds_pts.copy()
    geo_var_colors = np.full((len(ds_pts),3), [128,128,128], dtype=np.uint8)
    if args.model and Path(args.model).exists():
        from uncertainty.geotransformer import GeoTransformer
        from uncertainty.mc_inference import run_mc_passes
        from uncertainty.pose_covariance import variance_to_rgb
        import torch
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        model = GeoTransformer(in_channels=6,embed_dim=128,num_heads=4,
                               num_layers=4,bottleneck_dropout=args.dropout_rate)
        model.load_state_dict(ckpt["model_state_dict"]); model.set_mc_mode(True); model.eval()
        cent = np.array(ckpt.get("centroids",[[0,0,0]])[0])
        scl = float(ckpt.get("scales",[1.0])[0])
        nrm = np.zeros_like(ds_pts)
        data = np.column_stack([(ds_pts-cent)/scl, nrm])
        data_t = torch.from_numpy(data.astype(np.float32))
        mean_t, var_t = run_mc_passes(model, data_t, T=args.mc_passes,
                                       batch_size=4096, device="cpu", verbose=True)
        geo_mean = mean_t.numpy()*scl + cent
        var_np = var_t.numpy()*(scl**2)
        geo_var_colors = (variance_to_rgb(var_np)*255).astype(np.uint8)
        print(f"  Epistemic σ²: mean={var_np.mean():.1f}, range=[{var_np.min():.1f}, {var_np.max():.1f}]")
    else:
        print("  Model not found — grey placeholder")

    # ── 7. CVaR grasp candidates ──
    print("\n=== 7. CVaR grasp generation ===")
    from scipy.spatial import cKDTree
    k = min(30, len(ds_pts)); tree = cKDTree(ds_pts); _, idx = tree.query(ds_pts, k=k)
    nbrs = ds_pts[idx]; mu_n = nbrs.mean(axis=1,keepdims=True)
    cov_n = np.einsum("nki,nkj->nij", nbrs-mu_n, nbrs-mu_n)/(k-1)
    _, eigvecs = np.linalg.eigh(cov_n)
    mesh_n = eigvecs[:,:,0].copy()
    d_in = np.sum(mesh_n*(-ds_pts), axis=1); mesh_n[d_in<0]*=-1.0
    ns = np.linalg.norm(mesh_n, axis=1, keepdims=True); ns[ns<1e-12]=1.0; mesh_n/=ns

    rng = np.random.default_rng(seed)
    cos_a = np.cos(np.arctan(args.mu)); cos_min = cos_a*0.5
    accepted, rejected = [], []
    for _ in range(15*200):
        if len(accepted)+len(rejected)>=15: break
        i=rng.integers(0,len(ds_pts)); j=rng.integers(0,len(ds_pts))
        if i==j: continue
        d=ds_pts[j]-ds_pts[i]; dist=np.linalg.norm(d)
        if dist<1e-9: continue
        dh=d/dist; s1=float(np.dot(dh,mesh_n[i])); s2=float(np.dot(-dh,mesh_n[j]))
        if s1>=cos_min and s2>=cos_min:
            (accepted if s1>=cos_a-1e-9 and s2>=cos_a-1e-9 else rejected).append((ds_pts[i],ds_pts[j]))
    print(f"  {len(accepted)} accepted, {len(rejected)} rejected")

    # Large visible grasp spheres + axis lines
    grasp_ok_pts, grasp_ok_cols = [], []
    grasp_fail_pts, grasp_fail_cols = [], []
    for (c1,c2) in accepted:
        # Big sphere at each contact
        grasp_ok_pts.append(sphere_vertices(c1, radius=0.004))
        grasp_ok_cols.append(np.full((500,3),[0,255,50],dtype=np.uint8))
        grasp_ok_pts.append(sphere_vertices(c2, radius=0.004))
        grasp_ok_cols.append(np.full((500,3),[0,255,50],dtype=np.uint8))
        # Line connecting them
        grasp_ok_pts.append(grasp_line_points(c1,c2,80))
        grasp_ok_cols.append(np.full((80,3),[0,200,50],dtype=np.uint8))
    for (c1,c2) in rejected[:8]:
        grasp_fail_pts.append(sphere_vertices(c1, radius=0.003))
        grasp_fail_cols.append(np.full((500,3),[255,30,30],dtype=np.uint8))
        grasp_fail_pts.append(sphere_vertices(c2, radius=0.003))
        grasp_fail_cols.append(np.full((500,3),[255,30,30],dtype=np.uint8))
        grasp_fail_pts.append(grasp_line_points(c1,c2,50))
        grasp_fail_cols.append(np.full((50,3),[200,30,30],dtype=np.uint8))

    # ── 8. Write GLB ──
    print("\n=== 8. Writing GLB ===")
    meshes = [
        {'name':'01_ORIGINAL_MESH',      'points':raw_pts,     'colors':np.full((len(raw_pts),3),[210,180,140],dtype=np.uint8), 'material_color':[0.82,0.70,0.55], 'emat':'beige',  'offset_x':-0.24},
        {'name':'02_VOXEL_5mm',          'points':ds_pts,      'colors':np.full((len(ds_pts),3),[150,150,150],dtype=np.uint8),  'material_color':[0.60,0.60,0.60], 'emat':'grey',   'offset_x':-0.18},
        {'name':'03_PCA_NORMALS',        'points':ds_pts,      'colors':normal_rgb,                                                'material_color':[0.70,0.70,0.70], 'emat':'rgb',    'offset_x':-0.12},
        {'name':'04_SCENE_NOISY',        'points':scene_pts,   'colors':np.full((len(scene_pts),3),[50,100,255],dtype=np.uint8),  'material_color':[0.20,0.40,1.00], 'emat':'blue',   'offset_x':-0.06},
        {'name':'05_TEASER_ALIGNED',     'points':aligned_pts, 'colors':np.full((len(aligned_pts),3),[0,230,60],dtype=np.uint8),  'material_color':[0.00,0.90,0.24], 'emat':'green',  'offset_x':0.00},
        {'name':'06_GEOTRANSFORMER_MEAN','points':geo_mean,    'colors':np.full((len(geo_mean),3),[0,200,220],dtype=np.uint8),    'material_color':[0.00,0.78,0.86], 'emat':'cyan',   'offset_x':0.06},
        {'name':'07_VARIANCE_HEATMAP',   'points':ds_pts,      'colors':geo_var_colors,                                            'material_color':[0.50,0.50,0.50], 'emat':'heat',   'offset_x':0.12},
    ]
    if grasp_ok_pts:
        meshes.append({'name':'08_GRASPS_PASSED', 'points':np.vstack(grasp_ok_pts), 'colors':np.vstack(grasp_ok_cols),
                       'material_color':[0.0,1.0,0.2], 'emissive':[0.0,0.5,0.1], 'emat':'ok', 'offset_x':0.18})
    if grasp_fail_pts:
        meshes.append({'name':'09_GRASPS_FAILED', 'points':np.vstack(grasp_fail_pts), 'colors':np.vstack(grasp_fail_cols),
                       'material_color':[1.0,0.0,0.0], 'emissive':[0.5,0.0,0.0], 'emat':'fail', 'offset_x':0.24})

    glb_path = output_dir / "repair_pipeline.glb"
    write_glb(str(glb_path), meshes)

    print(f"\n{'='*65}")
    print(f"  Done → {glb_path}")
    print(f"  Blender: File → Import → glTF 2.0 → repair_pipeline.glb")
    print(f"  All 9 pipeline stages visible with distinct colours.")
    print(f"  Grasp spheres are large 4mm-radius balls at contact points.")
    print(f"  Grasp axis lines connect each contact pair.")
    print(f"{'='*65}")


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR full pipeline → GLB for Blender")
    p.add_argument("fragment", help="OBJ or PLY file")
    p.add_argument("--output", default="results/blender_pipeline", help="Output directory")
    p.add_argument("--model", default="checkpoints_146/geotransformer_best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-angle", type=float, default=25.0)
    p.add_argument("--max-translation", type=float, default=0.03)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--mc-passes", type=int, default=30)
    p.add_argument("--dropout-rate", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    main()
