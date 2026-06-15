#!/usr/bin/env python3
"""
RePAIR pipeline → interactive 3D HTML via Plotly.

Runs the full perception pipeline and generates a self-contained HTML
file with a 3×3 subplot grid showing every pipeline stage in colour.
Double-click the HTML — opens in any browser with orbit/zoom/pan.

Stages (3×3 grid)
------------------
  01 Original     02 Voxel 5mm    03 PCA Normals
  04 Scene Noisy  05 TEASER++     06 GeoTransformer
  07 Variance     08 Grasps Pass  09 Grasps Fail

Simulation mode (--simulation)
------------------------------
  Single-scene HTML showing table, fragment, camera, grasp approach.

Usage
-----
    # Quick pipeline (skip MC Dropout, ~10 sec)
    python scripts/visualize_pipeline.py RPf_00577.obj --quick

    # Full pipeline with MC Dropout variance
    python scripts/visualize_pipeline.py RPf_00577.obj \\
        --model checkpoints_146/geotransformer_best.pt

    # Robot simulation scene
    python scripts/visualize_pipeline.py RPf_00577.obj --simulation

    # Custom output path
    python scripts/visualize_pipeline.py RPf_00577.obj --quick \\
        --output results/my_pipeline.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# ── Voxel + PCA ─────────────────────────────────────────────────────


def voxel_ds(points, vs=0.005):
    if vs <= 0 or len(points) < 2: return points
    min_pt = points.min(axis=0)
    vi = np.floor((points - min_pt) / vs).astype(np.int64)
    sp = vi.max(axis=0) - vi.min(axis=0) + 1
    ids = vi[:,0]*sp[1]*sp[2] + vi[:,1]*sp[2] + vi[:,2]
    _, inv = np.unique(ids, return_inverse=True)
    nv = inv.max()+1; down = np.zeros((nv,3),np.float64)
    np.add.at(down, inv, points)
    cnt = np.bincount(inv, minlength=nv).astype(np.float64)
    return down / np.maximum(cnt,1)[:,None]


def pca_normals(pts, k=30):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts); _, idx = tree.query(pts, k=min(k,len(pts)))
    nbrs = pts[idx]; mu = nbrs.mean(axis=1, keepdims=True)
    c = nbrs - mu
    cv = np.einsum("nki,nkj->nij", c, c)/(min(k,len(pts))-1)
    _, eig = np.linalg.eigh(cv)
    n = eig[:,:,0].copy()
    cent = pts.mean(axis=0)
    d = np.sum(n*(cent-pts), axis=1); n[d<0]*=-1
    ns = np.linalg.norm(n, axis=1, keepdims=True); ns[ns<1e-12]=1
    return n/ns


# ── Grasp sphere/line generators ────────────────────────────────────


def sphere_cloud(center, radius=0.005, n=300):
    rng = np.random.default_rng(abs(hash(str(center))) % 2**32)
    th = np.arccos(1 - 2*rng.random(n))
    ph = 2*np.pi*rng.random(n)
    return np.column_stack([radius*np.sin(th)*np.cos(ph)+center[0],
                            radius*np.sin(th)*np.sin(ph)+center[1],
                            radius*np.cos(th)+center[2]])


def line_pts(c1, c2, n=60):
    t = np.linspace(0,1,n)
    return c1 + t[:,None]*(c2-c1)


# ── Plotly helpers ──────────────────────────────────────────────────


def _scatter3d(pts, color_rgba_str, size=2):
    """Single 3D scatter trace."""
    return go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers', marker=dict(size=size, color=color_rgba_str),
        showlegend=False,
    )


def _subplot_scene(fig, row, col, pts, color, size=2):
    """Add a 3D scatter subplot to the figure."""
    if len(pts) > 30000:
        idx = np.random.default_rng(0).choice(len(pts), 30000, replace=False)
        pts = pts[idx]
    fig.add_trace(_scatter3d(pts, color, size), row=row, col=col)
    fig.update_scenes(
        dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            camera=dict(eye=dict(x=0, y=0, z=2)),
        ),
        row=row, col=col,
    )


# ── Main pipeline ───────────────────────────────────────────────────


def run_pipeline(args):
    import open3d as o3d

    output = args.output or "results/pipeline.html"

    # ── 1. Load ──
    print("=== 1. Loading fragment ===")
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]),float(p[2]),float(p[3])])
        raw = np.array(verts, np.float64)
    else:
        pcd = o3d.io.read_point_cloud(args.fragment)
        raw = np.asarray(pcd.points, np.float64)
    cent = raw.mean(axis=0); raw -= cent
    print(f"  {len(raw):,} points, centred")

    # ── 2. Voxel ──
    print("\n=== 2. Voxel downsample ===")
    ds = voxel_ds(raw, 0.005)
    print(f"  {len(raw):,} → {len(ds):,}")

    # ── 3. PCA normals ──
    print("\n=== 3. PCA normals ===")
    norms = pca_normals(ds)
    normal_rgb = ((norms+1)*127.5).clip(0,255).astype(np.uint8)

    # ── 4. Scene ──
    print("\n=== 4. Scene cloud ===")
    T_gt, rot_deg, t_norm = _random_se3(args.max_angle, args.max_translation, args.seed)
    scene = _transform(T_gt, ds)
    print(f"  {rot_deg:.1f}° rot, {t_norm:.4f}m trans")

    # ── 5. TEASER++ ──
    print("\n=== 5. TEASER++ ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import register_teaser, TeaserParams
    sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene))
    mp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds))
    t0 = time.perf_counter()
    r = register_teaser(sp, mp, TeaserParams(c_threshold=0.005,noise_bound=0.001,fpfh_radius=0.035,ratio_threshold=0.9))
    T_est = np.asarray(r.T, np.float64)
    aligned = _transform(T_est, scene)
    print(f"  {r.rotation_angle_deg:.2f}° rot {r.translation_norm*1000:.1f}mm ({time.perf_counter()-t0:.1f}s)")

    # ── 6. MC Dropout ──
    geo_mean = ds.copy()
    var_colors_u8 = np.full((len(ds),3), [128,128,128], np.uint8)
    if not args.quick and args.model and Path(args.model).exists():
        print("\n=== 6. MC Dropout ===")
        from uncertainty.geotransformer import GeoTransformer
        from uncertainty.mc_inference import run_mc_passes
        from uncertainty.pose_covariance import variance_to_rgb
        import torch
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        model = GeoTransformer(in_channels=6,embed_dim=128,num_heads=4,num_layers=4,bottleneck_dropout=0.2)
        model.load_state_dict(ckpt["model_state_dict"]); model.set_mc_mode(True); model.eval()
        cnt = np.array(ckpt.get("centroids",[[0,0,0]])[0])
        scl = float(ckpt.get("scales",[1.])[0])
        nrm = np.zeros_like(ds)
        data = np.column_stack([(ds-cnt)/scl, nrm])
        dt = torch.from_numpy(data.astype(np.float32))
        mt, vt = run_mc_passes(model, dt, T=args.mc_passes, batch_size=4096, device="cpu", verbose=True)
        geo_mean = mt.numpy()*scl + cnt
        vn = vt.numpy()*(scl**2)
        var_colors_u8 = (variance_to_rgb(vn)*255).astype(np.uint8)
        print(f"  σ² mean={vn.mean():.1f}")
    elif args.quick:
        print("\n=== 6. MC Dropout skipped (--quick) ===")
    else:
        print("\n=== 6. MC Dropout skipped (no model) ===")

    # ── 7. CVaR grasps ──
    print("\n=== 7. CVaR grasps ===")
    from scipy.spatial import cKDTree
    k = min(30, len(ds)); tree = cKDTree(ds); _, idx = tree.query(ds, k=k)
    nbrs = ds[idx]; mu_n = nbrs.mean(axis=1, keepdims=True)
    cov_n = np.einsum("nki,nkj->nij",nbrs-mu_n,nbrs-mu_n)/(k-1)
    _, eigv = np.linalg.eigh(cov_n)
    mn = eigv[:,:,0].copy()
    d_in = np.sum(mn*(-ds), axis=1); mn[d_in<0]*=-1
    ns = np.linalg.norm(mn, axis=1, keepdims=True); ns[ns<1e-12]=1; mn/=ns
    rng = np.random.default_rng(args.seed)
    ca = np.cos(np.arctan(args.mu)); cm = ca*0.5
    acc, rej = [], []
    for _ in range(15*200):
        if len(acc)+len(rej)>=15: break
        i=rng.integers(0,len(ds)); j=rng.integers(0,len(ds))
        if i==j: continue
        d=ds[j]-ds[i]; dist=np.linalg.norm(d)
        if dist<1e-9: continue
        dh=d/dist; s1=float(np.dot(dh,mn[i])); s2=float(np.dot(-dh,mn[j]))
        if s1>=cm and s2>=cm:
            (acc if s1>=ca-1e-9 and s2>=ca-1e-9 else rej).append((ds[i],ds[j]))
    print(f"  {len(acc)} accepted, {len(rej)} rejected")

    ok_pts, ok_col = [], []
    for (c1,c2) in acc:
        for c in [c1,c2]:
            ok_pts.append(sphere_cloud(c,0.005))
        ok_pts.append(line_pts(c1,c2,60))
    fail_pts, fail_col = [], []
    for (c1,c2) in rej[:6]:
        for c in [c1,c2]:
            fail_pts.append(sphere_cloud(c,0.004))
        fail_pts.append(line_pts(c1,c2,50))

    # ── Build Plotly 3x3 grid ──
    print("\n=== 8. Building interactive HTML ===")
    fig = make_subplots(rows=3, cols=3,
        subplot_titles=[
            '01 Original',     '02 Voxel 5mm',    '03 PCA Normals',
            '04 Scene Noisy',  '05 TEASER++',     '06 GeoTransformer',
            '07 Variance',     '08 Grasps Pass',   '09 Grasps Fail',
        ],
        specs=[[{'type':'scatter3d'}]*3]*3,
        horizontal_spacing=0.01, vertical_spacing=0.02,
    )

    # Row 1
    _subplot_scene(fig,1,1, raw, 'rgb(210,180,140)', size=1)
    # Row 1 col 2
    _subplot_scene(fig,1,2, ds,  'rgb(150,150,150)', size=2)
    # Row 1 col 3 — PCA normals (per-vertex colors)
    if len(ds) > 10000:
        si = np.random.default_rng(0).choice(len(ds), 10000, replace=False)
    else:
        si = slice(None)
    fig.add_trace(go.Scatter3d(x=ds[si,0],y=ds[si,1],z=ds[si,2],
        mode='markers', marker=dict(size=2,
            color=[f'rgb({normal_rgb[i,0]},{normal_rgb[i,1]},{normal_rgb[i,2]})' for i in si]),
        showlegend=False), row=1, col=3)
    fig.update_scenes(dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',
        camera=dict(eye=dict(x=0,y=0,z=2))), row=1, col=3)

    # Row 2
    _subplot_scene(fig,2,1, scene,   'rgb(50,100,255)', size=2)
    _subplot_scene(fig,2,2, aligned, 'rgb(0,230,60)',   size=2)
    _subplot_scene(fig,2,3, geo_mean,'rgb(0,200,220)',  size=2)

    # Row 3
    # Row 3 col 1 — Variance heatmap (per-vertex colors)
    if len(ds) > 10000:
        si = np.random.default_rng(0).choice(len(ds), 10000, replace=False)
    else:
        si = slice(None)
    fig.add_trace(go.Scatter3d(x=ds[si,0],y=ds[si,1],z=ds[si,2],
        mode='markers', marker=dict(size=2,
            color=[f'rgb({var_colors_u8[i,0]},{var_colors_u8[i,1]},{var_colors_u8[i,2]})' for i in si]),
        showlegend=False), row=3, col=1)
    fig.update_scenes(dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',
        camera=dict(eye=dict(x=0,y=0,z=2))), row=3, col=1)

    if ok_pts:
        _subplot_scene(fig,3,2, np.vstack(ok_pts), 'rgb(0,255,50)', size=3)
    else:
        _subplot_scene(fig,3,2, np.array([[0,0,0]]), 'rgb(0,0,0)', size=1)
    if fail_pts:
        _subplot_scene(fig,3,3, np.vstack(fail_pts), 'rgb(255,30,30)', size=3)
    else:
        _subplot_scene(fig,3,3, np.array([[0,0,0]]), 'rgb(0,0,0)', size=1)

    # Global layout
    fig.update_layout(
        title=dict(text="<b>RePAIR Pipeline — All Stages</b>", font=dict(size=20), x=0.5),
        height=1000, width=1400,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f"\n  Done → {output}")
    print(f"  Double-click to open in browser.")


# ── Simulation mode ─────────────────────────────────────────────────


def run_simulation(args):
    output = args.output or "results/simulation.html"

    print("=== Loading fragment ===")
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]),float(p[2]),float(p[3])])
        frag = np.array(verts, np.float64)
    else:
        import open3d as o3d
        frag = np.asarray(o3d.io.read_point_cloud(args.fragment).points, np.float64)
    cent = frag.mean(axis=0); frag -= cent
    frag[:,2] -= frag[:,2].min()  # sit on table
    fc = frag.mean(axis=0)
    print(f"  {len(frag):,} points, centred, seated")

    # Table
    rng = np.random.default_rng(0)
    tw, td = 0.8, 0.6
    tx = rng.uniform(-tw/2,tw/2,8000)
    ty = rng.uniform(-td/2,td/2,8000)
    table = np.column_stack([tx, ty, np.full(8000,-0.002)])

    # Grasp positions
    pre_grasp = np.array([fc[0], fc[1], 0.15])
    grasp_pos = np.array([fc[0], fc[1], frag[:,2].max()+0.005])
    camera_pos = np.array([fc[0], fc[1], 0.45])

    # Approach arrow + camera cone
    t = np.linspace(0,1,100)
    arrow = pre_grasp + t[:,None]*(grasp_pos-pre_grasp)
    n_ring, n_line = 16, 40
    cone_pts = []
    for i in range(n_ring):
        a = 2*np.pi*i/n_ring
        rp = grasp_pos + np.array([0.10*np.cos(a), 0.10*np.sin(a), 0])
        tl = np.linspace(0,1,n_line)
        cone_pts.append(camera_pos + tl[:,None]*(rp-camera_pos))
    cone = np.vstack(cone_pts)

    # Build single-scene Plotly
    fig = go.Figure()
    traces = [
        (table,      'rgb(80,80,90)',   'Table',        1),
        (frag,       'rgb(210,180,140)','Fragment',      2),
        (np.array([camera_pos]), 'rgb(0,220,255)', 'Camera', 4),
        (cone,       'rgb(100,180,255)','Camera FOV',    1),
        (np.array([pre_grasp]), 'rgb(0,255,80)', 'Pre-Grasp', 5),
        (arrow,      'rgb(255,200,50)','Approach',      2),
        (np.array([grasp_pos]), 'rgb(255,40,40)', 'Grasp', 5),
    ]
    for pts, col, name, sz in traces:
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers', marker=dict(size=sz, color=col),
            name=name,
        ))

    fig.update_layout(
        title=dict(text="<b>Robot Simulation — Tabletop Pick-and-Place</b>", font=dict(size=20), x=0.5),
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            camera=dict(eye=dict(x=0, y=0, z=2)),
        ),
        height=800, width=1100,
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f"\n  Done → {output}")
    print(f"  Double-click to open in browser.  Orbit: drag.  Zoom: scroll.")


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR pipeline → interactive 3D HTML")
    p.add_argument("fragment", help="OBJ or PLY file")
    p.add_argument("--output", default=None, help="Output HTML path")
    p.add_argument("--simulation", action="store_true", help="Robot simulation mode")
    p.add_argument("--model", default="checkpoints_146/geotransformer_best.pt")
    p.add_argument("--quick", action="store_true", help="Skip MC Dropout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-angle", type=float, default=25.0)
    p.add_argument("--max-translation", type=float, default=0.03)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--mc-passes", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    if args.simulation:
        run_simulation(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
