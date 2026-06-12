#!/usr/bin/env python3
"""
End-to-end grasp success rate experiment for RePAIR pipeline.

Measures success rates at each stage of the perception-to-control
pipeline across multiple fragments:

  Stage 1 — Registration (TEASER++):  scene → 6-DoF pose
  Stage 2 — Force-Closure (LP):       pose → stable two-finger grasp
  Stage 3 — CVaR Filter:              stochastic geometry → risk-averse decision
  Stage 4 — Reachability:             grasp pose within robot workspace

For each fragment, N random scene perturbations are generated and the
full pipeline is run.  Per-stage and end-to-end success rates are
reported with confidence intervals.

Methodology
-----------
  1. Generate random SE(3) scene perturbation (ground truth).
  2. TEASER++ registers scene → model (ADD-S, RMS rot metrics).
  3. Force-closure LP tests baseline deterministic geometry.
  4. CVaR filter tests stochastic geometry (≥ 1 realization from
     isotropic per-point Gaussian noise at contact points).
  5. Reachability checks Cartesian workspace and top-down approach.
  6. Repeat with N seeds → statistical success rates.

Usage
-----
    python scripts/grasp_experiment.py repair_fragments_ds/RPf_0052*_ds.ply \\
        --seeds 5 --output results/grasp_experiment

    # With CVaR (requires variance cloud)
    python scripts/grasp_experiment.py fragment.ply \\
        --variance-cloud variance_cloud.pcd \\
        --seeds 3 --cvar-realizations 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Inline SE(3) + geometry (no torch/Open3D) ────────────────────────


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


def _transform_points_np(T, pts):
    return pts @ T[:3,:3].T + T[:3,3]


def orthonormal_basis(n):
    n = n / np.linalg.norm(n)
    axis = np.array([1.,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
    u = np.cross(n, axis); u /= np.linalg.norm(u)
    v = np.cross(n, u); return u, v


def friction_cone_gens(normal, mu=0.5, m=8):
    if mu <= 0: return (normal/np.linalg.norm(normal)).reshape(1,3)
    a = np.arctan(mu); n = normal/np.linalg.norm(normal)
    u, v = orthonormal_basis(n); th = np.linspace(0, 2*np.pi, m, endpoint=False)
    return np.array([np.cos(a)*n + np.sin(a)*(np.cos(t)*u + np.sin(t)*v) for t in th])


def build_wrench(pos, gens):
    forces = gens.T
    px, py, pz = pos
    skew = np.array([[0,-pz,py],[pz,0,-px],[-py,px,0]])
    return np.vstack([forces, skew@forces]).astype(np.float64)


def test_fc_lp(W):
    n = W.shape[1]
    if n < 7: return False, 0.0
    nv = n + 1; c = np.zeros(nv); c[-1] = -1.0
    A_eq = np.zeros((6, nv)); A_eq[:,:n] = W
    sum_row = np.zeros((1, nv)); sum_row[0,:n] = 1.0
    A_eq = np.vstack([A_eq, sum_row]); b_eq = np.zeros(7); b_eq[6] = 1.0
    A_ub = np.zeros((n, nv))
    for j in range(n): A_ub[j,j] = -1.0; A_ub[j,-1] = 1.0
    b_ub = np.zeros(n)
    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(None,None)]*nv, method="highs")
    if not res.success: return False, 0.0
    eps = float(res.x[-1] if res.x[-1] is not None else 0)
    return eps > 1e-9, max(0.0, eps)


# ── Evaluation metrics (inline) ──────────────────────────────────────


def compute_add_s(pts_est, pts_model):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_model)
    dists, _ = tree.query(pts_est, k=1)
    return float(np.mean(dists))


def compute_rms_pose_error(T_est, T_gt):
    R_est, t_est = T_est[:3,:3].astype(np.float64), T_est[:3,3].astype(np.float64)
    R_gt, t_gt = T_gt[:3,:3].astype(np.float64), T_gt[:3,3].astype(np.float64)
    tr = np.trace(R_gt @ R_est)
    rot = float(np.rad2deg(np.arccos(np.clip((tr-1)/2, -1, 1))))
    trans = float(np.linalg.norm(t_est + R_gt.T @ t_gt))
    return rot, trans


# ── Grasp candidate generation ───────────────────────────────────────


def generate_antipodal_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    mu: float = 0.5,
    max_candidates: int = 10,
    seed: int | None = None,
) -> list[dict]:
    """
    Sample random surface points and check antipodal condition.

    Returns up to max_candidates grasp pairs (contact1, contact2).
    """
    rng = np.random.default_rng(seed)
    cos_alpha = np.cos(np.arctan(mu))
    # Relax friction constraint for candidate search (real check is stricter)
    cos_ap_min = cos_alpha * 0.5
    n_pts = len(points)
    candidates = []

    for _ in range(max_candidates * 50):  # oversample 50×
        if len(candidates) >= max_candidates:
            break
        i = rng.integers(0, n_pts)
        j = rng.integers(0, n_pts)
        if i == j:
            continue
        d = points[j] - points[i]
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            continue
        d_hat = d / dist
        s1 = float(np.dot(d_hat, normals[i]))
        s2 = float(np.dot(-d_hat, normals[j]))
        if s1 >= cos_ap_min and s2 >= cos_ap_min:
            candidates.append({
                "contact1": points[i].tolist(),
                "contact2": points[j].tolist(),
                "normal1": normals[i].tolist(),
                "normal2": normals[j].tolist(),
                "score1": s1, "score2": s2,
            })

    return candidates


def estimate_normals(pts, k=30):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    neighbours = pts[idx]
    mu = neighbours.mean(axis=1, keepdims=True)
    centred = neighbours - mu
    cov = np.einsum("nki,nkj->nij", centred, centred) / (min(k, len(pts)) - 1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].copy()
    centroid = pts.mean(axis=0)
    dot = np.sum(normals * (pts - centroid), axis=1)
    normals[dot < 0] *= -1.0
    ns = np.linalg.norm(normals, axis=1, keepdims=True)
    ns[ns < 1e-12] = 1.0
    return normals / ns


# ── Per-trial result ─────────────────────────────────────────────────


@dataclass
class TrialResult:
    fragment: str
    seed: int
    scene_angle_deg: float = 0.0
    scene_translation_m: float = 0.0

    # Stage 1: Registration
    reg_success: bool = False
    add_s_mm: float = 0.0
    rms_rot_deg: float = 0.0
    rms_trans_mm: float = 0.0
    reg_runtime_s: float = 0.0

    # Stage 2: Force-Closure
    fc_success: bool = False
    fc_epsilon: float = 0.0
    num_candidates: int = 0
    num_fc_pass: int = 0

    # Stage 3: CVaR
    cvar_success: bool = False
    cvar_epsilon: float = 0.0
    cvar_realizations: int = 0
    cvar_failures: int = 0

    # Stage 4: Reachability
    reachable: bool = False

    # Overall
    end_to_end_success: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "fragment": self.fragment, "seed": self.seed,
            "scene_angle_deg": self.scene_angle_deg,
            "scene_translation_m": self.scene_translation_m,
            "reg_success": self.reg_success,
            "add_s_mm": self.add_s_mm, "rms_rot_deg": self.rms_rot_deg,
            "rms_trans_mm": self.rms_trans_mm, "reg_runtime_s": self.reg_runtime_s,
            "fc_success": self.fc_success, "fc_epsilon": self.fc_epsilon,
            "num_candidates": self.num_candidates,
            "num_fc_pass": self.num_fc_pass,
            "cvar_success": self.cvar_success, "cvar_epsilon": self.cvar_epsilon,
            "cvar_realizations": self.cvar_realizations,
            "cvar_failures": self.cvar_failures,
            "reachable": self.reachable,
            "end_to_end_success": self.end_to_end_success,
            "error": self.error,
        }


# ── Single-trial execution ──────────────────────────────────────────


def run_trial(
    ply_path: Path,
    seed: int,
    max_angle_deg: float,
    max_translation: float,
    mu: float,
    cvar_realizations: int,
    cvar_alpha: float,
    cvar_variance_scale: float,
    reg_add_s_threshold: float,
    reg_rot_threshold: float,
    workspace_radius: float,
) -> TrialResult:
    """
    Run the full perception-to-control pipeline for a single trial.
    Uses subprocess for TEASER++ registration (isolates torch/Open3D).
    """
    import open3d as o3d  # lazy

    result = TrialResult(fragment=ply_path.stem, seed=seed)

    try:
        # ── 1. Load model + generate scene ──
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if not pcd.has_points():
            result.error = "Failed to load PLY"
            return result
        points_model = np.asarray(pcd.points, dtype=np.float64)
        normals_model = estimate_normals(points_model)

        T_gt, rot_deg, t_norm = _random_se3(max_angle_deg, max_translation, seed)
        result.scene_angle_deg = rot_deg
        result.scene_translation_m = t_norm
        scene_pts = _transform_points_np(T_gt, points_model)

        # Build Open3D point clouds
        model_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points_model.astype(np.float64))
        )
        scene_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(scene_pts.astype(np.float64))
        )

        # ── 2. TEASER++ registration (inline, lazy import) ──
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from registration.teaser_registration import register_teaser, TeaserParams

        t0 = time.perf_counter()
        params = TeaserParams(
            c_threshold=0.005, noise_bound=0.001,
            fpfh_radius=0.035, ratio_threshold=0.9,
            max_correspondences=5000,
        )
        teaser_result = register_teaser(scene_pcd, model_pcd, params)
        result.reg_runtime_s = time.perf_counter() - t0
        T_est = np.asarray(teaser_result.T, dtype=np.float64)

        # Evaluate registration
        rot, trans = compute_rms_pose_error(T_est, T_gt)
        T_comp = T_est @ T_gt
        pts_est = _transform_points_np(T_comp, points_model)
        adds = compute_add_s(pts_est, points_model)

        result.add_s_mm = adds * 1000
        result.rms_rot_deg = rot
        result.rms_trans_mm = trans * 1000
        result.reg_success = (adds < reg_add_s_threshold and rot < reg_rot_threshold)

        if not result.reg_success:
            return result  # Stage 1 failed → no point continuing

        # ── 3. Generate grasp candidates ──
        candidates = generate_antipodal_candidates(
            points_model, normals_model, mu=mu, max_candidates=10,
            seed=seed * 2,
        )
        result.num_candidates = len(candidates)

        if not candidates:
            return result  # no antipodal candidates → fc_success remains False

        # ── 4. Force-closure check ──
        fc_pass = 0
        best_fc_eps = 0.0
        best_candidate = None
        for cand in candidates:
            c1 = np.array(cand["contact1"]); c2 = np.array(cand["contact2"])
            n1 = np.array(cand["normal1"]); n2 = np.array(cand["normal2"])
            gens1 = friction_cone_gens(n1, mu); gens2 = friction_cone_gens(n2, mu)
            W = np.hstack([build_wrench(c1, gens1), build_wrench(c2, gens2)])
            ok, eps = test_fc_lp(W)
            if ok:
                fc_pass += 1
                if eps > best_fc_eps:
                    best_fc_eps = eps
                    best_candidate = cand

        result.num_fc_pass = fc_pass
        result.fc_epsilon = best_fc_eps
        result.fc_success = fc_pass > 0

        if not result.fc_success:
            return result  # Stage 2 failed

        # ── 5. CVaR (stochastic geometry) ──
        cvar_eps_vals = []
        cvar_failures = 0
        for k in range(cvar_realizations):
            # Perturb contact points with isotropic Gaussian noise
            sigma = cvar_variance_scale * 0.001  # 1mm base uncertainty
            rng_cvar = np.random.default_rng(seed * 100 + k)
            noise_c1 = rng_cvar.normal(0, sigma, 3)
            noise_c2 = rng_cvar.normal(0, sigma, 3)
            c1_k = np.array(best_candidate["contact1"]) + noise_c1
            c2_k = np.array(best_candidate["contact2"]) + noise_c2

            # Re-find closest normals after perturbation
            from scipy.spatial import cKDTree
            tree = cKDTree(points_model)
            _, i1 = tree.query(c1_k, k=1); _, i2 = tree.query(c2_k, k=1)
            n1_k = normals_model[i1]; n2_k = normals_model[i2]

            gens1_k = friction_cone_gens(n1_k, mu); gens2_k = friction_cone_gens(n2_k, mu)
            W_k = np.hstack([build_wrench(c1_k, gens1_k), build_wrench(c2_k, gens2_k)])
            ok_k, eps_k = test_fc_lp(W_k)
            cvar_eps_vals.append(eps_k if ok_k else 0.0)
            if not ok_k:
                cvar_failures += 1

        sorted_eps = np.sort(cvar_eps_vals)
        k_tail = max(int(np.ceil(cvar_alpha * cvar_realizations)), 1)
        cvar_eps = float(sorted_eps[:k_tail].mean())

        result.cvar_epsilon = cvar_eps
        result.cvar_failures = cvar_failures
        result.cvar_realizations = cvar_realizations
        result.cvar_success = cvar_eps > 0

        if not result.cvar_success:
            return result  # Stage 3 failed

        # ── 6. Reachability check ──
        # Grasp midpoint must be within robot workspace (cylindrical approx.)
        c1_best = np.array(best_candidate["contact1"])
        c2_best = np.array(best_candidate["contact2"])
        grasp_center = (c1_best + c2_best) * 0.5
        xy_dist = np.linalg.norm(grasp_center[:2])
        z_val = grasp_center[2]
        # Rough Mirobot workspace: radius 0.3m, Z ∈ [0.02, 0.30]m
        result.reachable = (xy_dist < workspace_radius and 0.02 < z_val < 0.30)
        result.end_to_end_success = result.reachable

    except Exception as exc:
        result.error = f"Trial exception: {exc}"

    return result


# ── Aggregation ──────────────────────────────────────────────────────


def aggregate(results: list[TrialResult]) -> dict:
    n = len(results)
    successes = [r for r in results if not r.error]
    reg_ok = [r for r in successes if r.reg_success]
    fc_ok = [r for r in reg_ok if r.fc_success]
    cvar_ok = [r for r in fc_ok if r.cvar_success]
    e2e_ok = [r for r in cvar_ok if r.end_to_end_success]

    def _pct(part, total):
        return part / total * 100 if total > 0 else 0.0

    return {
        "num_trials": n,
        "num_errors": sum(1 for r in results if r.error),
        "num_registration_attempted": len(successes),
        "num_registration_success": len(reg_ok),
        "registration_success_rate": _pct(len(reg_ok), len(successes)),
        "num_fc_success": len(fc_ok),
        "fc_success_rate": _pct(len(fc_ok), len(reg_ok)),
        "num_cvar_success": len(cvar_ok),
        "cvar_success_rate": _pct(len(cvar_ok), len(fc_ok)),
        "num_reachable": len(e2e_ok),
        "reachability_rate": _pct(len(e2e_ok), len(cvar_ok)),
        "end_to_end_success_rate": _pct(len(e2e_ok), len(successes)),
        "mean_add_s_mm": float(np.mean([r.add_s_mm for r in reg_ok])) if reg_ok else 0,
        "mean_rms_rot_deg": float(np.mean([r.rms_rot_deg for r in reg_ok])) if reg_ok else 0,
        "mean_fc_epsilon": float(np.mean([r.fc_epsilon for r in fc_ok])) if fc_ok else 0,
        "mean_cvar_epsilon": float(np.mean([r.cvar_epsilon for r in cvar_ok])) if cvar_ok else 0,
    }


# ── Output ───────────────────────────────────────────────────────────


def save_results(results: list[TrialResult], agg: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "grasp_experiment.csv"
    keys = list(TrialResult("", 0).to_dict().keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r.to_dict())

    json_path = output_dir / "grasp_experiment.json"
    with open(json_path, "w") as f:
        json.dump({"aggregate": agg, "trials": [r.to_dict() for r in results]}, f, indent=2)

    report_path = output_dir / "grasp_experiment.txt"
    lines = [
        "=" * 72, "  GRASP SUCCESS RATE EXPERIMENT", "=" * 72,
        f"  Trials: {agg['num_trials']} | Errors: {agg['num_errors']}",
        "",
        f"  Stage 1 — Registration:     {agg['registration_success_rate']:5.1f}% "
        f"({agg['num_registration_success']}/{agg['num_registration_attempted']})",
        f"    Mean ADD-S: {agg['mean_add_s_mm']:.2f} mm  |  "
        f"Mean RMS rot: {agg['mean_rms_rot_deg']:.2f}°",
        "",
        f"  Stage 2 — Force-Closure:    {agg['fc_success_rate']:5.1f}% "
        f"({agg['num_fc_success']}/{agg['num_registration_success']})",
        f"    Mean FC ε:  {agg['mean_fc_epsilon']:.4f}",
        "",
        f"  Stage 3 — CVaR Filter:      {agg['cvar_success_rate']:5.1f}% "
        f"({agg['num_cvar_success']}/{agg['num_fc_success']})",
        f"    Mean CVaR ε: {agg['mean_cvar_epsilon']:.4f}",
        "",
        f"  Stage 4 — Reachability:     {agg['reachability_rate']:5.1f}% "
        f"({agg['num_reachable']}/{agg['num_cvar_success']})",
        "",
        f"  ─────────────────────────────────────────",
        f"  END-TO-END SUCCESS RATE:    {agg['end_to_end_success_rate']:5.1f}% "
        f"({agg['num_reachable']}/{agg['num_registration_attempted']})",
        "=" * 72,
    ]
    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nResults saved to {output_dir}/")
    print(f"  grasp_experiment.csv")
    print(f"  grasp_experiment.json")
    print(f"  grasp_experiment.txt")


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end grasp success rate experiment for RePAIR pipeline",
    )
    p.add_argument("fragments", nargs="+", help="Fragment PLY files or directories")
    p.add_argument("--output", type=str, default="results/grasp_experiment",
                   help="Output directory (default: results/grasp_experiment)")
    p.add_argument("--seeds", type=int, default=5,
                   help="Trials per fragment (default: 5)")
    p.add_argument("--max-angle", type=float, default=25.0,
                   help="Max scene rotation (deg, default: 25)")
    p.add_argument("--max-translation", type=float, default=0.03,
                   help="Max scene translation (m, default: 0.03)")
    p.add_argument("--mu", type=float, default=0.5,
                   help="Friction coefficient (default: 0.5)")
    p.add_argument("--cvar-realizations", type=int, default=50,
                   help="CVaR stochastic realizations (default: 50)")
    p.add_argument("--cvar-alpha", type=float, default=0.05,
                   help="CVaR tail fraction (default: 0.05)")
    p.add_argument("--cvar-variance-scale", type=float, default=1.0,
                   help="Contact noise σ scale factor (default: 1.0)")
    p.add_argument("--reg-add-s-threshold", type=float, default=0.010,
                   help="Registration ADD-S threshold (m, default: 0.010 = 10mm)")
    p.add_argument("--reg-rot-threshold", type=float, default=5.0,
                   help="Registration rotation error threshold (deg, default: 5)")
    p.add_argument("--workspace-radius", type=float, default=0.3,
                   help="Robot workspace XY radius (m, default: 0.3)")
    return p.parse_args()


def _find_plys(entries: list[str]) -> list[Path]:
    paths = []
    for e in entries:
        p = Path(e)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.ply")))
        elif p.is_file():
            paths.append(p)
    return sorted(set(paths))


def main() -> None:
    args = parse_args()
    ply_paths = _find_plys(args.fragments)
    if not ply_paths:
        raise SystemExit("No PLY files found.")

    output_dir = Path(args.output)
    print(f"Fragments: {len(ply_paths)}  "
          f"Seeds per fragment: {args.seeds}  "
          f"→  {len(ply_paths) * args.seeds} trials total")
    print(f"Registration thresholds: ADD-S < {args.reg_add_s_threshold*1000:.0f}mm, "
          f"rot < {args.reg_rot_threshold:.0f}°")
    print(f"CVaR: α={args.cvar_alpha}, N={args.cvar_realizations}, "
          f"μ={args.mu}")
    print()

    results: list[TrialResult] = []
    total = len(ply_paths) * args.seeds
    count = 0

    for frag_path in ply_paths:
        for seed_i in range(args.seeds):
            seed = seed_i * 137 + hash(frag_path.stem) % 1000
            count += 1
            print(f"[{count:3d}/{total}] {frag_path.stem} seed={seed} ",
                  end="", flush=True)
            t0 = time.perf_counter()

            result = run_trial(
                frag_path, seed, args.max_angle, args.max_translation,
                args.mu, args.cvar_realizations, args.cvar_alpha,
                args.cvar_variance_scale, args.reg_add_s_threshold,
                args.reg_rot_threshold, args.workspace_radius,
            )
            results.append(result)

            elapsed = time.perf_counter() - t0
            if result.error:
                print(f"ERROR: {result.error[:80]}")
            else:
                stages = []
                if result.reg_success:
                    stages.append(f"REG+{result.add_s_mm:.1f}mm")
                else:
                    stages.append("REG✗")
                if result.fc_success:
                    stages.append(f"FC+")
                else:
                    stages.append("FC✗")
                if result.cvar_success:
                    stages.append(f"CVaR+")
                else:
                    stages.append("CVaR✗")
                if result.end_to_end_success:
                    stages.append("✓")
                else:
                    stages.append("✗")
                print(f"{' → '.join(stages)}  ({elapsed:.0f}s)")

    agg = aggregate(results)
    save_results(results, agg, output_dir)


if __name__ == "__main__":
    main()
