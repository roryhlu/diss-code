#!/usr/bin/env python3
"""
RePAIR full pipeline → coloured PLY files for Blender.

Exports every pipeline stage as an individual ASCII PLY file with
per-vertex colours and baked-in spatial offsets so all 9 stages
are visible side-by-side.  No GLB, no Blender scripting —
File → Import → Stanford PLY for each file.

Stages (9 files in output directory)
-------------------------------------
  01_original.ply         Beige  — raw OBJ surface
  02_voxel_5mm.ply        Grey   — 5 mm voxel centroids
  03_pca_normals.ply      RGB    — normal direction as colour
  04_scene_noisy.ply      Blue   — random SE(3) perturbation
  05_teaser_aligned.ply   Green  — registration result
  06_geotransformer.ply   Cyan   — MC Dropout predicted surface
  07_variance.ply         Heat   — epistemic uncertainty (blue→red)
  08_grasps_pass.ply      GrnGlo — CVaR accepted, large spheres
  09_grasps_fail.ply      RedGlo — CVaR rejected, large spheres

Usage
-----
    python scripts/export_pipeline_glb.py RPf_00577.obj \\
        --model checkpoints_146/geotransformer_best.pt \\
        --output results/blender_pipeline/

    Then: Blender → File → Import → Stanford PLY (for each file)
"""

from __future__ import annotations

import argparse
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


# ── Voxel / PCA ─────────────────────────────────────────────────────


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


# ── ASCII PLY writer — guaranteed to work in any Blender version ────


def write_ply_ascii(path: str, points: np.ndarray, colours_uint8: np.ndarray) -> None:
    """Write ASCII PLY with float x,y,z and uchar r,g,b per vertex."""
    n = len(points)
    pts = points.astype(np.float64)
    cls = colours_uint8.astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                    f"{cls[i,0]} {cls[i,1]} {cls[i,2]}\n")


def sphere_cloud(center, radius=0.005, n=500):
    """Random points on a sphere surface."""
    rng = np.random.default_rng(hash(str(center)) % 2**32)
    theta = np.arccos(1 - 2*rng.random(n))
    phi = 2*np.pi*rng.random(n)
    return np.column_stack([
        radius*np.sin(theta)*np.cos(phi)+center[0],
        radius*np.sin(theta)*np.sin(phi)+center[1],
        radius*np.cos(theta)+center[2]])


def line_points(c1, c2, n=80):
    """Points along a line segment."""
    t = np.linspace(0, 1, n)
    return c1 + t[:, None] * (c2 - c1)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import open3d as o3d

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # ── 1. Load original OBJ ──
    print("=== 1. Loading original OBJ ===")
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
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
    norms = pca_normals(ds_pts)
    normal_rgb = ((norms + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # ── 4. Scene cloud ──
    print("\n=== 4. Scene cloud (random SE(3) perturbation) ===")
    T_gt, rot_deg, t_norm = _random_se3(args.max_angle, args.max_translation, seed)
    print(f"  {rot_deg:.1f}° rotation, {t_norm:.4f}m translation")
    scene_pts = _transform(T_gt, ds_pts)

    # ── 5. TEASER++ ──
    print("\n=== 5. TEASER++ registration ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import register_teaser, TeaserParams
    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts))
    model_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds_pts))
    t0 = time.perf_counter()
    r = register_teaser(scene_pcd, model_pcd,
        TeaserParams(c_threshold=0.005, noise_bound=0.001, fpfh_radius=0.035, ratio_threshold=0.9))
    elapsed = time.perf_counter()-t0
    T_est = np.asarray(r.T, dtype=np.float64)
    aligned_pts = _transform(T_est, scene_pts)
    print(f"  {r.rotation_angle_deg:.2f}° rot, {r.translation_norm*1000:.1f}mm trans ({elapsed:.1f}s)")

    # ── 6. MC Dropout ──
    print("\n=== 6. GeoTransformer MC Dropout ===")
    geo_mean = ds_pts.copy()
    var_colors = np.full((len(ds_pts),3), [128,128,128], dtype=np.uint8)
    if args.model and Path(args.model).exists():
        from uncertainty.geotransformer import GeoTransformer
        from uncertainty.mc_inference import run_mc_passes
        from uncertainty.pose_covariance import variance_to_rgb
        import torch
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        model = GeoTransformer(in_channels=6, embed_dim=128, num_heads=4,
                               num_layers=4, bottleneck_dropout=args.dropout_rate)
        model.load_state_dict(ckpt["model_state_dict"]); model.set_mc_mode(True); model.eval()
        cent = np.array(ckpt.get("centroids", [[0,0,0]])[0])
        scl = float(ckpt.get("scales", [1.0])[0])
        nrm_data = np.zeros_like(ds_pts)
        data = np.column_stack([(ds_pts-cent)/scl, nrm_data])
        data_t = torch.from_numpy(data.astype(np.float32))
        mean_t, var_t = run_mc_passes(model, data_t, T=args.mc_passes,
                                       batch_size=4096, device="cpu", verbose=True)
        geo_mean = mean_t.numpy()*scl + cent
        var_np = var_t.numpy()*(scl**2)
        var_colors = (variance_to_rgb(var_np)*255).astype(np.uint8)
        print(f"  σ² mean={var_np.mean():.1f}, range=[{var_np.min():.1f},{var_np.max():.1f}]")
    else:
        print("  Model not found — grey placeholder")

    # ── 7. CVaR grasp candidates ──
    print("\n=== 7. CVaR grasp generation ===")
    from scipy.spatial import cKDTree
    k = min(30, len(ds_pts)); tree = cKDTree(ds_pts); _, idx = tree.query(ds_pts, k=k)
    nbrs = ds_pts[idx]; mu_n = nbrs.mean(axis=1, keepdims=True)
    cov_n = np.einsum("nki,nkj->nij", nbrs-mu_n, nbrs-mu_n)/(k-1)
    _, eigv = np.linalg.eigh(cov_n)
    mesh_n = eigv[:,:,0].copy()
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

    # ── Build large visible grasp meshes ──
    grasp_ok_pts, grasp_ok_col = [], []
    for (c1,c2) in accepted:
        for c in [c1,c2]:
            grasp_ok_pts.append(sphere_cloud(c, 0.005))
            grasp_ok_col.append(np.full((500,3),[0,255,50],np.uint8))
        grasp_ok_pts.append(line_points(c1,c2,80))
        grasp_ok_col.append(np.full((80,3),[0,200,50],np.uint8))

    grasp_fail_pts, grasp_fail_col = [], []
    for (c1,c2) in rejected[:8]:
        for c in [c1,c2]:
            grasp_fail_pts.append(sphere_cloud(c, 0.004))
            grasp_fail_col.append(np.full((500,3),[255,30,30],np.uint8))
        grasp_fail_pts.append(line_points(c1,c2,60))
        grasp_fail_col.append(np.full((60,3),[200,30,30],np.uint8))

    # ── Side-by-side offsets ──
    # Each stage shifted 0.07m in X so they don't overlap.
    SPACING = 0.07

    # ── 8. Write PLY files ──
    print("\n=== 8. Writing coloured PLY files ===")
    stages = [
        ("01_original.ply",           raw_pts,     np.full((len(raw_pts),3),[210,180,140],np.uint8)),
        ("02_voxel_5mm.ply",          ds_pts,      np.full((len(ds_pts),3),[150,150,150],np.uint8)),
        ("03_pca_normals.ply",        ds_pts,      normal_rgb),
        ("04_scene_noisy.ply",        scene_pts,   np.full((len(scene_pts),3),[50,100,255],np.uint8)),
        ("05_teaser_aligned.ply",     aligned_pts, np.full((len(aligned_pts),3),[0,230,60],np.uint8)),
        ("06_geotransformer.ply",     geo_mean,    np.full((len(geo_mean),3),[0,200,220],np.uint8)),
        ("07_variance.ply",           ds_pts,      var_colors),
    ]

    for idx, (name, pts, cols) in enumerate(stages):
        pts_offset = pts.copy()
        pts_offset[:, 0] += idx * SPACING
        write_ply_ascii(str(output_dir / name), pts_offset, cols)
        print(f"  {name}")

    if grasp_ok_pts:
        go = np.vstack(grasp_ok_pts); go[:,0] += 7*SPACING
        write_ply_ascii(str(output_dir / "08_grasps_pass.ply"), go, np.vstack(grasp_ok_col))
        print(f"  08_grasps_pass.ply")
    if grasp_fail_pts:
        gf = np.vstack(grasp_fail_pts); gf[:,0] += 8*SPACING
        write_ply_ascii(str(output_dir / "09_grasps_fail.ply"), gf, np.vstack(grasp_fail_col))
        print(f"  09_grasps_fail.ply")

    # ── README ──
    readme = f"""REPAIR PIPELINE VISUALISATION — 9 stage files

Blender instructions:
  1. File → Import → Stanford PLY (.ply)
  2. Import each file listed below
  3. Press Numpad 7 for TOP-DOWN view
  4. Press Numpad 1 for FRONT view  
  5. Press Numpad 3 for SIDE view
  6. Outliner (top-right panel) shows all 9 objects by name
  7. Click the eye icon to toggle visibility per object

Files and colours:
  01_original.ply          BEIGE   — raw OBJ mesh ({len(raw_pts):,} verts)
  02_voxel_5mm.ply         GREY    — 5mm voxel grid ({len(ds_pts):,} pts)
  03_pca_normals.ply       RAINBOW — PCA normals as RGB ({len(ds_pts):,} pts)
  04_scene_noisy.ply       BLUE    — random SE(3) perturbation
  05_teaser_aligned.ply    GREEN   — TEASER++ registration result
  06_geotransformer.ply    CYAN    — GeoTransformer predicted surface
  07_variance.ply          HEATMAP — epistemic uncertainty (blue=low, red=high)
  08_grasps_pass.ply       GREEN   — CVaR accepted grasp spheres + axis
  09_grasps_fail.ply       RED     — CVaR rejected grasp spheres + axis

Each file is offset along X by 0.07m so stages don't overlap.
"""
    (output_dir / "README.txt").write_text(readme)

    print(f"\n{'='*60}")
    print(f"  Done — {len(stages)+2} PLY files in {output_dir}/")
    print(f"  Blender: File → Import → Stanford PLY → select each .ply")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR full pipeline → coloured PLY for Blender")
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
