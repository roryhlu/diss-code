#!/usr/bin/env python3
"""
RePAIR pipeline → interactive 3D HTML with Three.js.

Runs the full perception pipeline and generates a self-contained
HTML file with proper 3D rendering: spheres for grasps, overlay mode
(all stages in one scene), checkbox toggles, and preset cameras.

Double-click the HTML — opens in any browser.

Usage
-----
    python scripts/visualize_pipeline.py RPf_00577.obj --quick
    python scripts/visualize_pipeline.py RPf_00577.obj --simulation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ── SE(3) + geometry helpers ─────────────────────────────────────────


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


# ── Three.js HTML generator ──────────────────────────────────────────


def points_to_json(pts, cols, max_n=30000):
    """Convert points + colours to JSON-serialisable arrays, downsampling if needed."""
    if len(pts) > max_n:
        idx = np.random.default_rng(0).choice(len(pts), max_n, replace=False)
        pts, cols = pts[idx], cols[idx]
    return {
        'x': pts[:,0].tolist(), 'y': pts[:,1].tolist(), 'z': pts[:,2].tolist(),
        'cr': cols[:,0].tolist() if cols.ndim>1 else [cols[0]]*len(pts),
        'cg': cols[:,1].tolist() if cols.ndim>1 else [cols[1]]*len(pts),
        'cb': cols[:,2].tolist() if cols.ndim>1 else [cols[2]]*len(pts),
    }


def sphere_json(centers, color_rgb, radius=0.006, n=300):
    """Generate sphere point cloud data for each center."""
    all_pts, all_col = [], []
    rng = np.random.default_rng(42)
    for c in centers:
        th = np.arccos(1 - 2*rng.random(n))
        ph = 2*np.pi*rng.random(n)
        pt = np.column_stack([
            radius*np.sin(th)*np.cos(ph)+c[0],
            radius*np.sin(th)*np.sin(ph)+c[1],
            radius*np.cos(th)+c[2]])
        all_pts.append(pt)
    if not all_pts: return {'x':[0],'y':[0],'z':[0],'cr':[0],'cg':[0],'cb':[0]}
    pts = np.vstack(all_pts)
    cols = np.full((len(pts),3), color_rgb, np.uint8)
    return points_to_json(pts, cols, max_n=99999)  # keep all sphere points


def line_json(pairs, color_rgb, n=40):
    """Generate line point cloud data for each pair."""
    all_pts = []
    for (c1,c2) in pairs:
        t = np.linspace(0,1,n)
        all_pts.append(c1 + t[:,None]*(c2-c1))
    if not all_pts: return {'x':[0],'y':[0],'z':[0],'cr':[0],'cg':[0],'cb':[0]}
    pts = np.vstack(all_pts)
    cols = np.full((len(pts),3), color_rgb, np.uint8)
    return points_to_json(pts, cols, max_n=99999)


def build_html(pipeline_data: list[dict]) -> str:
    """Generate complete self-contained HTML with Three.js."""
    stages_json = json.dumps(pipeline_data, separators=(',', ':'))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RePAIR Pipeline</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#1a1a2e;color:#eee;font-family:system-ui;overflow:hidden}}
#panel{{position:absolute;top:10px;right:10px;z-index:10;background:#16213e;border-radius:8px;padding:12px;max-width:280px}}
#panel h3{{margin:0 0 8px;font-size:14px;color:#e94560}}
#panel label{{display:block;font-size:12px;margin:2px 0;cursor:pointer;user-select:none}}
#panel input{{margin-right:6px;accent-color:#e94560}}
#panel button{{display:block;width:100%;margin:3px 0;padding:6px;background:#0f3460;color:#eee;border:none;border-radius:4px;cursor:pointer;font-size:12px}}
#panel button:hover{{background:#e94560}}
#panel button.cam-active{{background:#e94560}}
#view3d{{width:100vw;height:100vh}}
</style>
</head>
<body>
<div id="panel">
<h3>RePAIR Pipeline</h3>
<div id="toggles"></div>
<div style="margin-top:8px;border-top:1px solid #333;padding-top:8px">
<b style="font-size:12px">Camera</b>
<button id="btn-top" class="active">Top-Down</button>
<button id="btn-front">Front</button>
<button id="btn-side">Side</button>
<button id="btn-iso">Isometric</button>
<button id="btn-overlay" class="active">Overlay</button>
</div>
</div>
<div id="view3d"></div>

<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"OrbitControls":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js"}}}}
</script>

<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'OrbitControls';

const STAGES = {stages_json};
const SPACING = 0.06;
let mode = 'overlay'; // 'overlay' | 'side'

// Scene
const container = document.getElementById('view3d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(50, container.clientWidth/container.clientHeight, 0.0001, 1000);
camera.position.set(0, 0, 0.30);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,0,0);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.zoomSpeed = 1.5;
controls.update();

// Lights
scene.add(new THREE.AmbientLight(0x404060, 3));
const dl = new THREE.DirectionalLight(0xffffff, 4);
dl.position.set(0.1,-0.2,0.15);
scene.add(dl);

// Axes
const ax = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0,0,0),new THREE.Vector3(0.02,0,0),
    new THREE.Vector3(0,0,0),new THREE.Vector3(0,0.02,0),
    new THREE.Vector3(0,0,0),new THREE.Vector3(0,0,0.02)]);
scene.add(new THREE.LineSegments(ax, new THREE.LineBasicMaterial({{color:0xffffff}})));

// Build point clouds
const groups = [];
const panelToggles = document.getElementById('toggles');

STAGES.forEach((sd, idx) => {{
    const g = new THREE.Group();
    g.name = sd.name;
    g.visible = sd.visible !== false;
    groups.push(g);

    // Points
    if (sd.x && sd.x.length > 0) {{
        const n = sd.x.length;
        const geom = new THREE.BufferGeometry();
        const pos = new Float32Array(n*3);
        const col = new Float32Array(n*3);
        for (let i=0;i<n;i++) {{
            pos[i*3]=sd.x[i]; pos[i*3+1]=sd.y[i]; pos[i*3+2]=sd.z[i];
            col[i*3]=sd.cr[i]/255; col[i*3+1]=sd.cg[i]/255; col[i*3+2]=sd.cb[i]/255;
        }}
        geom.setAttribute('position', new THREE.BufferAttribute(pos,3));
        geom.setAttribute('color', new THREE.BufferAttribute(col,3));
        const mat = new THREE.PointsMaterial({{size:sd.size||0.005, vertexColors:true, sizeAttenuation:true}});
        g.add(new THREE.Points(geom, mat));
    }}

    // Apply offset
    g.position.x = sd.offset_x || 0;
    g.position.y = sd.offset_y || 0;
    g.position.z = sd.offset_z || 0;

    scene.add(g);

    // Checkbox
    const label = document.createElement('label');
    label.innerHTML = `<input type="checkbox" checked id="cb_${{idx}}"> ${{sd.name}}`;
    label.querySelector('input').onchange = (e) => {{ g.visible = e.target.checked; }};
    panelToggles.appendChild(label);
}});

function setView(dir) {{
    const d = 0.30;
    if (dir==='top') {{ camera.position.set(0,0,d); controls.target.set(0,0,0); }}
    else if (dir==='front') {{ camera.position.set(0,-d,0); controls.target.set(0,0,0); }}
    else if (dir==='side') {{ camera.position.set(d,0,0); controls.target.set(0,0,0); }}
    else {{ camera.position.set(d*0.7,-d*0.7,d*0.5); controls.target.set(0,0,0); }}
    controls.update();
    document.querySelectorAll('#panel button.cam-active').forEach(b=>b.classList.remove('cam-active'));
    document.getElementById('btn-'+dir).classList.add('cam-active');
}}

// Wire up camera buttons (module scope — use addEventListener, not onclick)
document.getElementById('btn-top').addEventListener('click', () => setView('top'));
document.getElementById('btn-front').addEventListener('click', () => setView('front'));
document.getElementById('btn-side').addEventListener('click', () => setView('side'));
document.getElementById('btn-iso').addEventListener('click', () => setView('iso'));
document.getElementById('btn-overlay').addEventListener('click', toggleMode);

function toggleMode() {{
    mode = mode === 'overlay' ? 'side' : 'overlay';
    const btn = document.getElementById('btn-overlay');
    btn.textContent = mode === 'overlay' ? 'Overlay' : 'Grid';
    btn.classList.toggle('active', mode==='overlay');
    groups.forEach((g, i) => {{
        if (mode === 'overlay') {{
            g.position.x = 0; g.position.y = 0;
        }} else {{
            g.position.x = i * SPACING;
            g.position.y = 0;
        }}
    }});
}}

// Render loop
function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
    camera.aspect = container.clientWidth/container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}});

// Top-down view on load
window.addEventListener('DOMContentLoaded', () => {{ setView('top'); }});
</script>
</body>
</html>'''


# ── Pipeline runner ──────────────────────────────────────────────────


def run_pipeline(args):
    import open3d as o3d

    output = Path(args.output or "results/pipeline.html")
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
        raw = np.array(verts, np.float64)
    else:
        pcd = o3d.io.read_point_cloud(args.fragment)
        raw = np.asarray(pcd.points, np.float64)
    cent = raw.mean(axis=0); raw -= cent
    print(f"  {len(raw):,} points, centred, extent={raw.max(axis=0)-raw.min(axis=0)}")

    # ── Voxel ──
    ds = voxel_ds(raw, 0.005)

    # ── PCA normals ──
    norms = pca_normals(ds)
    normal_rgb = ((norms+1)*127.5).clip(0,255).astype(np.uint8)

    # ── Scene ──
    T_gt, rot_deg, t_norm = _random_se3(args.max_angle, args.max_translation, args.seed)
    scene = _transform(T_gt, ds)

    # ── TEASER++ ──
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import register_teaser, TeaserParams
    sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene))
    mp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds))
    r = register_teaser(sp, mp, TeaserParams(c_threshold=0.005, noise_bound=0.001, fpfh_radius=0.035, ratio_threshold=0.9))
    T_est = np.asarray(r.T, np.float64)
    aligned = _transform(T_est, scene)

    # ── MC Dropout ──
    geo_mean = ds.copy()
    var_colors_u8 = np.full((len(ds),3), [255,20,147], np.uint8)  # hot pink — clearly distinct
    if not args.quick and args.model and Path(args.model).exists():
        from uncertainty.geotransformer import GeoTransformer
        from uncertainty.mc_inference import run_mc_passes
        from uncertainty.pose_covariance import variance_to_rgb
        import torch
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
        model = GeoTransformer(in_channels=6, embed_dim=128, num_heads=4, num_layers=4, bottleneck_dropout=0.2)
        model.load_state_dict(ckpt["model_state_dict"]); model.set_mc_mode(True); model.eval()
        cnt = np.array(ckpt.get("centroids", [[0,0,0]])[0])
        scl = float(ckpt.get("scales", [1.])[0])
        nrm = np.zeros_like(ds)
        data = np.column_stack([(ds-cnt)/scl, nrm])
        dt = torch.from_numpy(data.astype(np.float32))
        mt, vt = run_mc_passes(model, dt, T=args.mc_passes, batch_size=4096, device="cpu", verbose=True)
        geo_mean = mt.numpy()*scl + cnt
        vn = vt.numpy()*(scl**2)
        var_colors_u8 = (variance_to_rgb(vn)*255).astype(np.uint8)

    # ── CVaR grasps ──
    from scipy.spatial import cKDTree
    k = min(30, len(ds)); tree = cKDTree(ds); _, idx = tree.query(ds, k=k)
    nbrs = ds[idx]; mu_n = nbrs.mean(axis=1, keepdims=True)
    cov_n = np.einsum("nki,nkj->nij", nbrs-mu_n, nbrs-mu_n)/(k-1)
    _, eigv = np.linalg.eigh(cov_n)
    mn = eigv[:,:,0].copy()
    d_in = np.sum(mn*(-ds), axis=1); mn[d_in<0]*=-1
    ns = np.linalg.norm(mn, axis=1, keepdims=True); ns[ns<1e-12]=1; mn/=ns
    rng_n = np.random.default_rng(args.seed)
    ca = np.cos(np.arctan(args.mu)); cm = ca*0.5
    z_top = float(np.percentile(ds[:,2], 70))
    acc, rej = [], []
    for _ in range(15*800):
        if len(acc)+len(rej)>=15: break
        i=rng_n.integers(0,len(ds)); j=rng_n.integers(0,len(ds))
        if i==j: continue
        d=ds[j]-ds[i]; dist=np.linalg.norm(d)
        if dist<1e-9: continue
        dh=d/dist
        s1=float(np.dot(dh,mn[i])); s2=float(np.dot(-dh,mn[j]))
        if s1>=cm and s2>=cm:
            # ── Practical graspability (top-down robot approach) ──
            # Gripper width: 2–50mm
            if dist < 0.002 or dist > 0.050: continue
            # At least one contact accessible from above
            if ds[i,2] < z_top and ds[j,2] < z_top: continue
            # Grasp axis vaguely vertical (within 70°)
            if abs(dh[2]) < np.cos(np.deg2rad(70)): continue
            (acc if s1>=ca-1e-9 and s2>=ca-1e-9 else rej).append((ds[i],ds[j]))

    # ── Build JSON data for each stage ──
    pipeline_data = [
        {'name':'01_Original',      'size':0.003,  'offset_x':0, **points_to_json(raw, np.full((len(raw),3),[210,180,140],np.uint8))},
        {'name':'02_Voxel',         'size':0.005,  'offset_x':0, **points_to_json(ds,  np.full((len(ds),3),[150,150,150],np.uint8))},
        {'name':'03_PCA_Normals',   'size':0.005,  'offset_x':0, **points_to_json(ds,  normal_rgb)},
        {'name':'04_Scene_Noisy',   'size':0.005,  'offset_x':0, **points_to_json(scene, np.full((len(scene),3),[50,100,255],np.uint8))},
        {'name':'05_TEASER_Aligned','size':0.005,  'offset_x':0, **points_to_json(aligned, np.full((len(aligned),3),[0,230,60],np.uint8))},
        {'name':'06_GeoTransformer','size':0.005,  'offset_x':0, **points_to_json(geo_mean, np.full((len(geo_mean),3),[0,200,220],np.uint8))},
        {'name':'07_Variance',      'size':0.005,  'offset_x':0, **points_to_json(ds, var_colors_u8)},
        {'name':'08_Grasps_Pass',   'size':0.150,  'offset_x':0, **sphere_json([c for (c,_) in acc]+[c for (_,c) in acc], [0,255,50], 0.300)},
        {'name':'09_Grasps_Fail',   'size':0.120,  'offset_x':0, **sphere_json([c for (c,_) in rej[:6]]+[c for (_,c) in rej[:6]], [255,30,30], 0.250)},
    ]
    # Add grasp axis lines as extra entries with sphere data layered on same offset
    if acc:
        pipeline_data.append({'name':'08b_Grasp_Axis_Pass','size':0.080,'offset_x':0,
            **line_json(acc, [0,200,50], 40)})
    if rej:
        pipeline_data.append({'name':'09b_Grasp_Axis_Fail','size':0.070,'offset_x':0,
            **line_json(rej[:6], [200,30,30], 30)})

    html = build_html(pipeline_data)
    output.write_text(html, encoding="utf-8")
    print(f"Done → {output}")


# ── Simulation mode ──────────────────────────────────────────────────


def run_simulation(args):
    output = Path(args.output or "results/simulation.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = Path(args.fragment).suffix.lower()
    if ext == ".obj":
        verts = []
        with open(args.fragment, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
        frag = np.array(verts, np.float64)
    else:
        import open3d as o3d
        frag = np.asarray(o3d.io.read_point_cloud(args.fragment).points, np.float64)
    cent = frag.mean(axis=0); frag -= cent
    frag[:,2] -= frag[:,2].min()
    fc = frag.mean(axis=0)

    rng_t = np.random.default_rng(0)
    tw, td = 0.8, 0.6
    table = np.column_stack([rng_t.uniform(-tw/2,tw/2,4000), rng_t.uniform(-td/2,td/2,4000), np.full(4000,-0.002)])

    pre = np.array([fc[0], fc[1], 0.15])
    grasp = np.array([fc[0], fc[1], frag[:,2].max()+0.005])
    cam = np.array([fc[0], fc[1], 0.45])

    t = np.linspace(0,1,80)
    arrow = pre + t[:,None]*(grasp-pre)

    cone = []
    for i in range(16):
        a = 2*np.pi*i/16
        rp = grasp + np.array([0.10*np.cos(a), 0.10*np.sin(a), 0])
        tl = np.linspace(0,1,30)
        cone.append(cam + tl[:,None]*(rp-cam))
    cone_pts = np.vstack(cone)

    sim_data = [
        {'name':'Table',     'size':0.005, 'offset_x':0, **points_to_json(table, np.full((len(table),3),[80,80,90],np.uint8))},
        {'name':'Fragment',  'size':0.004, 'offset_x':0, **points_to_json(frag,  np.full((len(frag),3),[210,180,140],np.uint8))},
        {'name':'Approach',  'size':0.006, 'offset_x':0, **points_to_json(arrow, np.full((len(arrow),3),[255,200,50],np.uint8))},
        {'name':'Camera_FOV','size':0.004, 'offset_x':0, **points_to_json(cone_pts, np.full((len(cone_pts),3),[100,180,255],np.uint8))},
        {'name':'Pre_Grasp', 'size':0.012, 'offset_x':0, **sphere_json([pre], [0,255,80], 0.008)},
        {'name':'Grasp_Pt',  'size':0.012, 'offset_x':0, **sphere_json([grasp], [255,40,40], 0.008)},
        {'name':'Camera',    'size':0.012, 'offset_x':0, **sphere_json([cam], [0,220,255], 0.006)},
    ]

    html = build_html(sim_data)
    output.write_text(html, encoding="utf-8")
    print(f"Done → {output}")


def parse_args():
    p = argparse.ArgumentParser(description="RePAIR pipeline → Three.js 3D HTML")
    p.add_argument("fragment", help="OBJ or PLY file")
    p.add_argument("--output", default=None)
    p.add_argument("--simulation", action="store_true")
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
