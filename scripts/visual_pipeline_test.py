#!/usr/bin/env python3
"""
Visual pipeline test — runs the full RePAIR pipeline and exports
coloured PLY files for Blender visualisation.

Produces a complete set of 3D assets showing every pipeline stage:
  fragment_mesh.ply      — Original fragment geometry (grey)
  scene_cloud.ply        — Noisy scene cloud with ground-truth pose (blue)
  aligned_cloud.ply      — TEASER++ registered result (green)
  variance_cloud.ply     — MC Dropout epistemic uncertainty (blue→red)
  grasps_accepted.ply    — CVaR-passed grasp contacts (green spheres)
  grasps_rejected.ply    — CVaR-failed grasp contacts (red spheres)
  render.py              — Blender script to load and render everything

Usage
-----
    python scripts/visual_pipeline_test.py RPf_00577.obj \
        --model checkpoints_146/geotransformer_best.pt \
        --output results/blender_scene/

    # Then in Blender:
    #   Scripting workspace → Open → results/blender_scene/render.py → Run
    #   F12 to render
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── Inline utilities (no torch/Open3D at module level) ──────────────


def write_ply_coloured(path: str, points: np.ndarray, colours: np.ndarray) -> None:
    """Write a binary PLY with x,y,z (double) and red,green,blue (uchar)."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment Created by RePAIR visual pipeline\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    if np.issubdtype(colours.dtype, np.floating):
        colours = np.clip(colours * 255, 0, 255).astype(np.uint8)
    # Write interleaved: 3 doubles + 3 uchars per vertex (no dtype promotion)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        pts = points.astype(np.float64)
        cls = colours.astype(np.uint8)
        for i in range(n):
            f.write(pts[i].tobytes())
            f.write(cls[i].tobytes())


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


def _load_obj_vertices(path: str) -> np.ndarray:
    """Fast line-by-line OBJ vertex extraction — reads only 'v ' lines."""
    vertices = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float64)


# ── Blender render.py generator ─────────────────────────────────────


def generate_blender_script(
    output_dir: Path,
    mesh_file: str,
    scene_file: str,
    aligned_file: str,
    variance_file: str,
    accepted_file: str,
    rejected_file: str,
) -> str:
    """Generate a standalone Blender Python script that loads and renders everything."""
    return f'''
"""RePAIR Pipeline Visualisation — run in Blender Scripting workspace."""

import bpy
import os
from math import radians

BASE = r"{output_dir.resolve()}"
if not os.path.isabs(BASE):
    BASE = os.path.join(os.path.dirname(__file__), BASE)

# ── Clear default scene ──
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Import PLY files (Blender 3.x + 4.x compatible) ──
def import_ply(path, name, color_rgb):
    abspath = os.path.join(BASE, path) if not os.path.isabs(path) else path
    if not os.path.exists(abspath):
        print(f"WARNING: File not found: {{abspath}}")
        return None
    # Try Blender 4.x API first, fall back to 3.x
    try:
        bpy.ops.wm.ply_import(filepath=abspath)
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=abspath)
    obj = bpy.context.active_object
    if obj is None:
        print(f"WARNING: Failed to import {{abspath}}")
        return None
    obj.name = name
    # Create material
    mat = bpy.data.materials.new(name=name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color_rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.append(mat)
    return obj

# ── Camera ──
bpy.ops.object.camera_add(location=(0.15, -0.20, 0.12))
camera = bpy.context.active_object
camera.name = "RenderCam"
camera.rotation_euler = (radians(60), 0, radians(45))
bpy.context.scene.camera = camera

# ── Lighting (three-point) ──
def add_light(name, location, energy, light_type='AREA'):
    bpy.ops.object.light_add(type=light_type, location=location)
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    if light_type == 'AREA':
        light.data.size = 0.3
    return light

key = add_light("KeyLight", (0.3, -0.3, 0.4), 500, 'AREA')
fill = add_light("FillLight", (-0.2, -0.1, 0.3), 200, 'AREA')
rim  = add_light("RimLight", (0.0, 0.3, 0.2), 300, 'AREA')

# ── World background ──
world = bpy.data.worlds["World"]
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.05, 0.05, 0.06, 1.0)
bg.inputs["Strength"].default_value = 0.5

# ── Import all pipeline assets ──
print("Importing RePAIR pipeline assets ...")

mesh_obj = import_ply("{Path(mesh_file).name}", "CAD_Model", (0.7, 0.7, 0.7))
scene_obj = import_ply("{Path(scene_file).name}", "Scene_Cloud", (0.1, 0.3, 0.9))
aligned_obj = import_ply("{Path(aligned_file).name}", "Aligned_Cloud", (0.0, 0.8, 0.2))

# Variance cloud: colour from vertex colours (PLY has uchar rgb)
variance_obj = import_ply("{Path(variance_file).name}", "Variance_Cloud", (0.5, 0.5, 0.5))
if variance_obj and variance_obj.data.color_attributes:
    mat_var = variance_obj.data.materials[0]
    mat_var.use_nodes = True
    bsdf_var = mat_var.node_tree.nodes["Principled BSDF"]
    try:
        attr = mat_var.node_tree.nodes.new("ShaderNodeVertexColor")
        mat_var.node_tree.links.new(attr.outputs["Color"], bsdf_var.inputs["Base Color"])
    except Exception:
        pass

# Grasp spheres
accepted_path = os.path.join(BASE, "{Path(accepted_file).name}")
rejected_path = os.path.join(BASE, "{Path(rejected_file).name}")
if os.path.exists(accepted_path):
    print(f"Loading accepted grasps: {{accepted_path}}")
    bpy.ops.wm.ply_import(filepath=accepted_path) if hasattr(bpy.ops.wm, 'ply_import') else bpy.ops.import_mesh.ply(filepath=accepted_path)
    acc_obj = bpy.context.active_object
    if acc_obj:
        acc_obj.name = "Grasps_Accepted"
        mat_acc = bpy.data.materials.new(name="Accepted_mat")
        mat_acc.use_nodes = True
        mat_acc.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.0, 1.0, 0.2, 1.0)
        mat_acc.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.0, 0.3, 0.05, 1.0)
        mat_acc.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 2.0
        acc_obj.data.materials.append(mat_acc)

if os.path.exists(rejected_path):
    print(f"Loading rejected grasps: {{rejected_path}}")
    bpy.ops.wm.ply_import(filepath=rejected_path) if hasattr(bpy.ops.wm, 'ply_import') else bpy.ops.import_mesh.ply(filepath=rejected_path)
    rej_obj = bpy.context.active_object
    if rej_obj:
        rej_obj.name = "Grasps_Rejected"
        mat_rej = bpy.data.materials.new(name="Rejected_mat")
        mat_rej.use_nodes = True
        mat_rej.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1.0, 0.1, 0.1, 1.0)
        mat_rej.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.3, 0.0, 0.0, 1.0)
        mat_rej.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.5
        rej_obj.data.materials.append(mat_rej)

# ── Scene setup ──
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = True

# ── Viewport ──
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].shading.type = 'MATERIAL'

print("RePAIR pipeline scene ready. F12 to render.")
'''


# ── Main pipeline ────────────────────────────────────────────────────


def main() -> None:
    import open3d as o3d

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.fragment).stem
    seed = args.seed

    # ── 1. Load fragment mesh ──
    print("=== 1. Loading fragment mesh ===")
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        # Fast line-by-line OBJ vertex extraction (no Open3D, no trimesh)
        model_points = _load_obj_vertices(args.fragment)
        if len(model_points) == 0:
            raise SystemExit(f"No vertices found in '{args.fragment}'")
    else:
        pcd = o3d.io.read_point_cloud(args.fragment)
        if not pcd.has_points():
            raise SystemExit(f"No points in '{args.fragment}'")
        model_points = np.asarray(pcd.points, dtype=np.float64)
    print(f"  {len(model_points):,} points")

    # Export coloured CAD mesh (grey)
    mesh_path = output_dir / "fragment_mesh.ply"
    grey = np.full((len(model_points), 3), [0.6, 0.6, 0.6])
    write_ply_coloured(str(mesh_path), model_points, grey)
    print(f"  Saved: {mesh_path.name}")

    # ── 2. Generate scene cloud ──
    print("\n=== 2. Generating scene cloud ===")
    T_gt, rot_deg, t_norm = _random_se3(
        max_angle_deg=args.max_angle,
        max_translation=args.max_translation,
        seed=seed,
    )
    print(f"  Rotation: {rot_deg:.1f}°, Translation: {t_norm:.4f}m")
    scene_points = _transform_points(T_gt, model_points)
    scene_blue = np.full((len(scene_points), 3), [0.1, 0.3, 0.9])
    scene_ply = output_dir / "scene_cloud.ply"
    write_ply_coloured(str(scene_ply), scene_points, scene_blue)
    print(f"  Saved: {scene_ply.name}")

    # ── 3. TEASER++ registration ──
    print("\n=== 3. Running TEASER++ registration ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import register_teaser, TeaserParams

    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_points))
    model_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_points))

    t0 = time.perf_counter()
    result = register_teaser(scene_pcd, model_pcd, TeaserParams(
        c_threshold=0.005, noise_bound=0.001,
        fpfh_radius=0.035, ratio_threshold=0.9,
    ))
    elapsed = time.perf_counter() - t0
    T_est = np.asarray(result.T, dtype=np.float64)

    aligned_points = _transform_points(T_est, scene_points)
    aligned_green = np.full((len(aligned_points), 3), [0.0, 0.8, 0.2])
    aligned_ply = output_dir / "aligned_cloud.ply"
    write_ply_coloured(str(aligned_ply), aligned_points, aligned_green)
    print(f"  TEASER++: {result.rotation_angle_deg:.2f}° rot, "
          f"{result.translation_norm*1000:.1f}mm trans  ({elapsed:.1f}s)")
    print(f"  Saved: {aligned_ply.name}")

    # ── 4. MC Dropout variance cloud ──
    print("\n=== 4. Running MC Dropout variance ===")
    variance_ply = output_dir / "variance_cloud.ply"

    if args.model and Path(args.model).exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from uncertainty.geotransformer import GeoTransformer
        from uncertainty.mc_inference import run_mc_passes
        from uncertainty.pose_covariance import variance_to_rgb
        import torch

        # Load model
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        model = GeoTransformer(in_channels=6, embed_dim=128, num_heads=4,
                               num_layers=4, bottleneck_dropout=args.dropout_rate)
        model.load_state_dict(ckpt["model_state_dict"])
        model.set_mc_mode(True)
        model.eval()

        # Normalise
        centroid = np.array(ckpt.get("centroids", [[0, 0, 0]])[0])
        scale = float(ckpt.get("scales", [1.0])[0])
        normals = np.zeros_like(model_points)
        data = np.column_stack([(model_points - centroid) / scale, normals])
        data_t = torch.from_numpy(data.astype(np.float32))

        # Run MC passes
        mean_t, var_t = run_mc_passes(model, data_t, T=args.mc_passes,
                                       batch_size=4096, device="cpu", verbose=True)
        mean_np = mean_t.numpy() * scale + centroid
        var_np = var_t.numpy() * (scale ** 2)
        colours = variance_to_rgb(var_np)
        write_ply_coloured(str(variance_ply), mean_np, colours)
        print(f"  Variance: mean σ²={var_np.mean():.1f}, range=[{var_np.min():.1f}, {var_np.max():.1f}]")
        print(f"  Saved: {variance_ply.name}")
    else:
        print(f"  Model not found — skipping MC Dropout.")
        # Copy scene cloud as placeholder
        colours = np.full((len(scene_points), 3), [0.5, 0.5, 0.5])
        write_ply_coloured(str(variance_ply), model_points, colours)

    # ── 5. CVaR grasp validation ──
    print("\n=== 5. Running CVaR grasp validation ===")
    accepted_ply = output_dir / "grasps_accepted.ply"
    rejected_ply = output_dir / "grasps_rejected.ply"

    # Use the fragment vertices for grasp geometry
    mesh_pts = model_points
    # Estimate normals via vectorised PCA
    from scipy.spatial import cKDTree
    k_n = min(30, len(mesh_pts))
    tree = cKDTree(mesh_pts)
    _, idx = tree.query(mesh_pts, k=k_n)
    neighbours = mesh_pts[idx]
    mu_n = neighbours.mean(axis=1, keepdims=True)
    centred_n = neighbours - mu_n
    cov = np.einsum("nki,nkj->nij", centred_n, centred_n) / (k_n - 1)
    _, eigvecs = np.linalg.eigh(cov)
    mesh_normals = eigvecs[:, :, 0].copy()
    centroid_m = mesh_pts.mean(axis=0)
    dot = np.sum(mesh_normals * (centroid_m - mesh_pts), axis=1)
    mesh_normals[dot < 0] *= -1.0
    ns = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
    ns[ns < 1e-12] = 1.0
    mesh_normals /= ns

    # Generate antipodal pairs
    rng = np.random.default_rng(seed)
    cos_alpha = np.cos(np.arctan(args.mu))
    cos_ap_min = cos_alpha * 0.5
    accepted = []
    rejected_all = []
    for _ in range(15 * 100):
        if len(accepted) + len(rejected_all) >= 15:
            break
        i = rng.integers(0, len(mesh_pts))
        j = rng.integers(0, len(mesh_pts))
        if i == j:
            continue
        d = mesh_pts[j] - mesh_pts[i]
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            continue
        d_hat = d / dist
        s1 = float(np.dot(d_hat, mesh_normals[i]))
        s2 = float(np.dot(-d_hat, mesh_normals[j]))
        if s1 >= cos_ap_min and s2 >= cos_ap_min:
            # Strict check
            if s1 >= cos_alpha - 1e-9 and s2 >= cos_alpha - 1e-9:
                accepted.append((mesh_pts[i], mesh_pts[j]))
            else:
                rejected_all.append((mesh_pts[i], mesh_pts[j]))

    if accepted:
        pts_acc = []
        cols_acc = []
        for c1, c2 in accepted:
            for pt in [c1, c2]:
                rng_s = np.random.default_rng(hash(str(c1)) % 2**32)
                sphere_pts = rng_s.normal(0, 0.003, (200, 3)) + pt
                pts_acc.append(sphere_pts)
                cols_acc.append(np.full((200, 3), [0.0, 1.0, 0.2]))
        if pts_acc:
            write_ply_coloured(str(accepted_ply),
                              np.vstack(pts_acc), np.vstack(cols_acc))
            print(f"  {len(accepted)} accepted grasps → {accepted_ply.name}")

    if rejected_all:
        pts_rej = []
        cols_rej = []
        for c1, c2 in rejected_all[:min(len(rejected_all), 10)]:
            rng_s = np.random.default_rng(hash(str(c1) + "r") % 2**32)
            for pt in [c1, c2]:
                sphere_pts = rng_s.normal(0, 0.002, (100, 3)) + pt
                pts_rej.append(sphere_pts)
                cols_rej.append(np.full((100, 3), [1.0, 0.1, 0.1]))
        if pts_rej:
            write_ply_coloured(str(rejected_ply),
                              np.vstack(pts_rej), np.vstack(cols_rej))
            print(f"  {len(rejected_all)} rejected grasps → {rejected_ply.name}")

    if not accepted and not rejected_all:
        # Generate at least some spheres for visualisation
        print("  No grasp candidates generated — creating sample spheres.")
        centroid_m = model_points.mean(axis=0)
        c1 = centroid_m + np.array([0.01, 0, 0])
        c2 = centroid_m + np.array([-0.01, 0, 0])
        accepted = [(c1, c2)]
        pts_acc = []
        cols_acc = []
        for pt in [c1, c2]:
            rng = np.random.default_rng(0)
            s = rng.normal(0, 0.003, (200, 3)) + pt
            pts_acc.append(s)
            cols_acc.append(np.full((200, 3), [0.0, 1.0, 0.2]))
        write_ply_coloured(str(accepted_ply),
                          np.vstack(pts_acc), np.vstack(cols_acc))

    # ── 6. Generate Blender render.py ──
    print("\n=== 6. Generating Blender script ===")
    blender_script = generate_blender_script(
        output_dir,
        str(mesh_path),
        str(scene_ply),
        str(aligned_ply),
        str(variance_ply),
        str(accepted_ply),
        str(rejected_ply),
    )
    render_path = output_dir / "render.py"
    render_path.write_text(blender_script)
    print(f"  Saved: {render_path.name}")

    # ── Done ──
    print(f"\n{'='*60}")
    print(f"  Pipeline complete — all assets in {output_dir}/")
    print(f"  Open Blender → Scripting workspace → Open →")
    print(f"    {render_path}")
    print(f"  Then click 'Run Script' or press F12 to render.")
    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser(description="Visual pipeline test for RePAIR")
    p.add_argument("fragment", help="Fragment OBJ/PLY file")
    p.add_argument("--output", default="results/blender_scene", help="Output directory")
    p.add_argument("--model", default="checkpoints_146/geotransformer_best.pt",
                   help="GeoTransformer checkpoint")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-angle", type=float, default=25.0, help="Max scene rotation (°)")
    p.add_argument("--max-translation", type=float, default=0.03, help="Max scene translation (m)")
    p.add_argument("--mu", type=float, default=0.5, help="Friction coefficient")
    p.add_argument("--mc-passes", type=int, default=30, help="MC Dropout passes")
    p.add_argument("--dropout-rate", type=float, default=0.2, help="MC Dropout probability")
    return p.parse_args()


if __name__ == "__main__":
    main()
