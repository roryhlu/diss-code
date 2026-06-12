#!/usr/bin/env python3
"""
TEASER++ (TLS) vs Open3D ICP (L2) benchmark — outlier-robustness comparison.

Core thesis claim: TEASER++'s Truncated Least Squares cost rejects outliers
that catastrophically degrade standard ICP.  This script proves it with
controlled noise injection.

Noise types
-----------
  outliers    — Random ghost points added to the scene cloud (mimics
                spurious reflections, clutter, false FPFH matches).
                Swept from 0% to 99% of model point count.

  gaussian    — Isotropic Gaussian jitter on scene point positions
                (mimics D405 depth sensor noise).  σ swept 0–10 mm.

Methodology
-----------
  For each (fragment, noise_level, seed):
    1. Generate ground-truth SE(3) scene perturbation.
    2. Inject controlled noise into the scene cloud.
    3. Run TEASER++ registration (TLS cost, FPFH + GNC-TLS).
    4. Run Open3D point-to-point ICP (L2 cost, iterative closest point).
    5. Evaluate both: ADD-S, Chamfer, RMS Pose Error.
    6. Repeat with N seeds for statistical power.

  Aggregate across seeds → mean ± std per noise level.
  Plot ADD-S and RMS rotation vs. outlier ratio / Gaussian σ.

Usage
-----
    python scripts/benchmark_teaser_vs_icp.py RPf_00577_ds.ply \\
        --outlier-ratios 0,0.1,0.2,0.3,0.5,0.7,0.9 \\
        --seeds 5 --output results/benchmark
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
from scipy.spatial import cKDTree


# ── Inline SE(3) utilities (no torch dependency) ──────────────────────


def _random_so3(rng: np.random.Generator, max_angle_deg: float = 30.0) -> np.ndarray:
    """Generate a random SO(3) matrix with bounded angle."""
    z = rng.uniform(-1.0, 1.0)
    th = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    axis = np.array([s * np.cos(th), s * np.sin(th), z])
    angle = rng.uniform(0.0, np.deg2rad(max_angle_deg))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _random_se3(
    max_angle_deg: float = 25.0,
    max_translation: float = 0.03,
    seed: int | None = None,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0)
    th = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    axis = np.array([s * np.cos(th), s * np.sin(th), z])
    angle = rng.uniform(0.0, np.deg2rad(max_angle_deg))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    z2 = rng.uniform(-1.0, 1.0)
    th2 = rng.uniform(0.0, 2.0 * np.pi)
    s2 = np.sqrt(max(0.0, 1.0 - z2 * z2))
    direction = np.array([s2 * np.cos(th2), s2 * np.sin(th2), z2])
    tn = rng.uniform(0.0, max_translation)
    t_vec = direction * tn
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_vec
    return T, float(np.rad2deg(angle)), float(tn)


def _transform_points_np(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return points @ R.T + t


# ── Noise injection ───────────────────────────────────────────────────


def inject_outlier_ghosts(
    points: np.ndarray,
    ratio: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add random ghost points to the point cloud.

    Ghosts are uniformly distributed in the bounding box of the original
    cloud, mimicking spurious reflections and background clutter.

    Args:
        points: (N, 3) clean point cloud.
        ratio:  Fraction of ghost points relative to N.
        seed:   RNG seed.

    Returns:
        (N + ⌈ratio·N⌉, 3) corrupted points (original + ghosts).
    """
    if ratio <= 0:
        return points.copy()
    rng = np.random.default_rng(seed)
    n_ghosts = max(1, int(len(points) * ratio))
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    ghosts = rng.uniform(min_pt, max_pt, (n_ghosts, 3))
    return np.vstack([points, ghosts])


def inject_gaussian_noise(
    points: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add isotropic Gaussian noise to point positions.

    Args:
        points: (N, 3) point cloud.
        sigma:  Standard deviation in metres.
        seed:   RNG seed.

    Returns:
        (N, 3) points with added noise.
    """
    if sigma <= 0:
        return points.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, points.shape)
    return points + noise.astype(points.dtype)


# ── Evaluation metrics (pure numpy) ───────────────────────────────────


def compute_add_s(points_est: np.ndarray, points_model: np.ndarray) -> dict:
    tree = cKDTree(points_model)
    dists, _ = tree.query(points_est, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p95": float(np.percentile(dists, 95)) if len(dists) > 0 else float("nan"),
    }


def compute_chamfer(points_est, points_model) -> dict:
    t_m, t_e = cKDTree(points_model), cKDTree(points_est)
    fwd = float(np.mean(t_m.query(points_est, k=1)[0]))
    bwd = float(np.mean(t_e.query(points_model, k=1)[0]))
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd}


def compute_rms_pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    R_est, t_est = T_est[:3, :3].astype(np.float64), T_est[:3, 3].astype(np.float64)
    R_gt, t_gt = T_gt[:3, :3].astype(np.float64), T_gt[:3, 3].astype(np.float64)
    tr = np.trace(R_gt @ R_est)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.rad2deg(np.arccos(cos_theta)))
    t_gt_inv = -R_gt.T @ t_gt
    trans_err = float(np.linalg.norm(t_est - t_gt_inv))
    return {"rotation_deg": rot_err, "translation_m": trans_err}


def evaluate(T_est, T_gt, points_model):
    T_comp = T_est @ T_gt
    pts_est = _transform_points_np(T_comp, points_model)
    return {
        **compute_add_s(pts_est, points_model),
        **compute_chamfer(pts_est, points_model),
        **compute_rms_pose_error(T_est, T_gt),
    }


# ── Result dataclass ──────────────────────────────────────────────────


@dataclass
class TrialResult:
    fragment: str
    noise_type: str
    noise_level: float
    seed: int
    scene_angle_deg: float
    scene_translation_m: float
    teaser_add_s_mean: float = 0.0
    teaser_chamfer_total: float = 0.0
    teaser_rms_rot_deg: float = 0.0
    teaser_rms_trans_m: float = 0.0
    teaser_runtime_s: float = 0.0
    icp_add_s_mean: float = 0.0
    icp_chamfer_total: float = 0.0
    icp_rms_rot_deg: float = 0.0
    icp_rms_trans_m: float = 0.0
    icp_runtime_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "fragment": self.fragment, "noise_type": self.noise_type,
            "noise_level": self.noise_level, "seed": self.seed,
            "scene_angle_deg": self.scene_angle_deg,
            "scene_translation_m": self.scene_translation_m,
            "teaser_add_s_mean": self.teaser_add_s_mean,
            "teaser_chamfer_total": self.teaser_chamfer_total,
            "teaser_rms_rot_deg": self.teaser_rms_rot_deg,
            "teaser_rms_trans_m": self.teaser_rms_trans_m,
            "teaser_runtime_s": self.teaser_runtime_s,
            "icp_add_s_mean": self.icp_add_s_mean,
            "icp_chamfer_total": self.icp_chamfer_total,
            "icp_rms_rot_deg": self.icp_rms_rot_deg,
            "icp_rms_trans_m": self.icp_rms_trans_m,
            "icp_runtime_s": self.icp_runtime_s,
            "error": self.error,
        }


# ── Registration ─────────────────────────────────────────────────────


def _run_teaser_inline(
    scene_pcd,
    model_pcd,
) -> np.ndarray | None:
    """Run TEASER++ registration inline. Returns (4,4) T_est or None."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import (
        register_teaser, TeaserParams,
    )
    params = TeaserParams(
        c_threshold=0.005,
        noise_bound=0.001,
        fpfh_radius=0.035,
        ratio_threshold=0.9,
        max_correspondences=5000,
    )
    try:
        result = register_teaser(scene_pcd, model_pcd, params)
        return np.asarray(result.T, dtype=np.float64)
    except Exception:
        return None


def _run_icp_inline(
    scene_pcd,
    model_pcd,
    T_init: np.ndarray | None = None,
    max_distance: float = 0.05,
) -> np.ndarray | None:
    """Run Open3D point-to-point ICP inline. Returns (4,4) T_est or None."""
    import open3d as o3d
    if T_init is None:
        T_init = np.eye(4)
    try:
        result = o3d.pipelines.registration.registration_icp(
            scene_pcd, model_pcd, max_distance, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100,
            ),
        )
        return np.asarray(result.transformation, dtype=np.float64)
    except Exception:
        return None


# ── Single-trial runner ──────────────────────────────────────────────


def run_trial(
    points_model: np.ndarray,
    stem: str,
    noise_type: str,
    noise_level: float,
    seed: int,
    max_angle_deg: float,
    max_translation: float,
) -> TrialResult:
    import open3d as o3d

    result = TrialResult(
        fragment=stem, noise_type=noise_type,
        noise_level=noise_level, seed=seed,
        scene_angle_deg=0.0, scene_translation_m=0.0,
    )

    try:
        # 1. Generate scene with ground truth
        T_gt, rot_deg, t_norm = _random_se3(
            max_angle_deg=max_angle_deg,
            max_translation=max_translation,
            seed=seed,
        )
        result.scene_angle_deg = rot_deg
        result.scene_translation_m = t_norm
        scene_clean = _transform_points_np(T_gt, points_model)

        # 2. Inject noise
        scene_rng = seed * 137 + 1
        if noise_type == "outliers":
            scene_noisy = inject_outlier_ghosts(scene_clean, noise_level, seed=scene_rng)
        elif noise_type == "gaussian":
            scene_noisy = inject_gaussian_noise(scene_clean, noise_level, seed=scene_rng)
        else:
            scene_noisy = scene_clean

        # Build Open3D point clouds
        model_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points_model.astype(np.float64))
        )
        scene_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(scene_noisy.astype(np.float64))
        )

        # 4. Run TEASER++
        t0 = time.perf_counter()
        T_teaser = _run_teaser_inline(scene_pcd, model_pcd)
        result.teaser_runtime_s = time.perf_counter() - t0
        if T_teaser is not None:
            m = evaluate(T_teaser, T_gt, points_model)
            result.teaser_add_s_mean = m["mean"]
            result.teaser_chamfer_total = m["total"]
            result.teaser_rms_rot_deg = m["rotation_deg"]
            result.teaser_rms_trans_m = m["translation_m"]
        else:
            result.error = "TEASER++ failed"

        # 5. Run ICP with perturbed ground-truth as initial guess.
        #    TEASER++ does global search via FPFH; ICP needs good
        #    initialization but must refine.  We perturb T_gt_inv by
        #    ~5° + 5mm so ICP has to work, and noise degrades it.
        perturb_rng = np.random.default_rng(seed * 3 + 7)
        R_perturb = _random_so3(perturb_rng, 5.0)
        t_perturb = perturb_rng.uniform(-0.005, 0.005, 3)
        T_init = np.eye(4)
        T_init[:3, :3] = R_perturb @ T_gt[:3, :3].T
        T_init[:3, 3] = -R_perturb @ T_gt[:3, :3].T @ T_gt[:3, 3] + t_perturb
        t0 = time.perf_counter()
        T_icp = _run_icp_inline(scene_pcd, model_pcd, T_init=T_init)
        result.icp_runtime_s = time.perf_counter() - t0
        if T_icp is not None:
            m = evaluate(T_icp, T_gt, points_model)
            result.icp_add_s_mean = m["mean"]
            result.icp_chamfer_total = m["total"]
            result.icp_rms_rot_deg = m["rotation_deg"]
            result.icp_rms_trans_m = m["translation_m"]

    except Exception as exc:
        result.error = f"Trial failed: {exc}"

    return result


# ── Aggregation and plotting ──────────────────────────────────────────


def aggregate(results: list[TrialResult]) -> list[dict]:
    """Group by (noise_type, noise_level) and compute mean/std."""
    groups: dict[tuple, list[TrialResult]] = {}
    for r in results:
        if r.error:
            continue
        key = (r.noise_type, r.noise_level)
        groups.setdefault(key, []).append(r)

    rows = []
    for (ntype, level), trials in sorted(groups.items()):
        def _stats(values):
            arr = np.array(values)
            return {"mean": np.mean(arr), "std": np.std(arr), "n": len(arr)}

        rows.append({
            "noise_type": ntype,
            "noise_level": level,
            "num_trials": len(trials),
            "teaser_add_s": _stats([t.teaser_add_s_mean for t in trials]),
            "icp_add_s": _stats([t.icp_add_s_mean for t in trials]),
            "teaser_rot": _stats([t.teaser_rms_rot_deg for t in trials]),
            "icp_rot": _stats([t.icp_rms_rot_deg for t in trials]),
            "teaser_chamfer": _stats([t.teaser_chamfer_total for t in trials]),
            "icp_chamfer": _stats([t.icp_chamfer_total for t in trials]),
            "teaser_runtime": _stats([t.teaser_runtime_s for t in trials]),
            "icp_runtime": _stats([t.icp_runtime_s for t in trials]),
        })
    return rows


def save_results(
    results: list[TrialResult],
    agg: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "benchmark_trials.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(TrialResult(
            fragment="", noise_type="", noise_level=0, seed=0,
            scene_angle_deg=0, scene_translation_m=0,
        ).to_dict().keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.to_dict())

    # JSON
    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump({"aggregate": agg, "trials": [r.to_dict() for r in results]}, f, indent=2)

    # Text report
    report_path = output_dir / "benchmark_report.txt"
    lines = ["=" * 78, "  TEASER++ (TLS) vs Open3D ICP (L2) BENCHMARK", "=" * 78]
    for row in agg:
        nt, nl = row["noise_type"], row["noise_level"]
        lines.append(
            f"\n  {nt} = {nl}  ({row['num_trials']} trials)\n"
            f"  {'':>20} {'ADD-S (mm)':>12} {'RMS rot (°)':>12} {'Runtime (s)':>12}\n"
            f"  {'TEASER++':>20} {row['teaser_add_s']['mean']*1000:>12.3f} "
            f"{row['teaser_rot']['mean']:>12.2f} {row['teaser_runtime']['mean']:>12.1f}\n"
            f"  {'ICP':>20} {row['icp_add_s']['mean']*1000:>12.3f} "
            f"{row['icp_rot']['mean']:>12.2f} {row['icp_runtime']['mean']:>12.1f}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nResults saved to {output_dir}/")
    print(f"  benchmark_trials.csv   — per-trial raw data")
    print(f"  benchmark_summary.json  — aggregate + per-trial detail")
    print(f"  benchmark_report.txt    — human-readable report")

    # Plot
    _plot_results(agg, output_dir)


def _plot_results(agg: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
        return

    # Group by noise type
    by_type: dict[str, list[dict]] = {}
    for row in agg:
        by_type.setdefault(row["noise_type"], []).append(row)

    n_types = len(by_type)
    fig, axes = plt.subplots(2, n_types, figsize=(6 * n_types, 10))
    if n_types == 1:
        axes = axes.reshape(2, 1)

    for col, (ntype, rows) in enumerate(sorted(by_type.items())):
        rows.sort(key=lambda r: r["noise_level"])
        x = [r["noise_level"] for r in rows]

        # ADD-S
        ax = axes[0, col]
        ax.errorbar(x, [r["teaser_add_s"]["mean"] * 1000 for r in rows],
                    yerr=[r["teaser_add_s"]["std"] * 1000 for r in rows],
                    marker="o", label="TEASER++ (TLS)", capsize=4)
        ax.errorbar(x, [r["icp_add_s"]["mean"] * 1000 for r in rows],
                    yerr=[r["icp_add_s"]["std"] * 1000 for r in rows],
                    marker="s", label="ICP (L2)", capsize=4)
        ax.set_xlabel(f"{ntype} level" if ntype == "outliers" else f"{ntype} σ (m)")
        ax.set_ylabel("ADD-S (mm)")
        ax.set_title(f"ADD-S vs {ntype}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # RMS Rotation
        ax = axes[1, col]
        ax.errorbar(x, [r["teaser_rot"]["mean"] for r in rows],
                    yerr=[r["teaser_rot"]["std"] for r in rows],
                    marker="o", label="TEASER++ (TLS)", capsize=4)
        ax.errorbar(x, [r["icp_rot"]["mean"] for r in rows],
                    yerr=[r["icp_rot"]["std"] for r in rows],
                    marker="s", label="ICP (L2)", capsize=4)
        ax.set_xlabel(f"{ntype} level" if ntype == "outliers" else f"{ntype} σ (m)")
        ax.set_ylabel("RMS Rotation Error (°)")
        ax.set_title(f"Rotation Error vs {ntype}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "benchmark_plot.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"  benchmark_plot.png      — error-bar comparison chart")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TEASER++ (TLS) vs Open3D ICP (L2) benchmark",
    )
    p.add_argument("cad_ply", type=str, help="CAD model point cloud (PLY)")
    p.add_argument("--output", type=str, default="results/benchmark",
                   help="Output directory (default: results/benchmark)")
    p.add_argument("--outlier-ratios", type=str, default="0,0.1,0.2,0.3,0.5,0.7,0.9",
                   help="Comma-separated outlier ratios (default: 0,0.1,0.2,0.3,0.5,0.7,0.9)")
    p.add_argument("--gaussian-sigmas", type=str, default="0,0.001,0.002,0.005,0.01",
                   help="Comma-separated Gaussian σ in metres (default: 0,0.001,0.002,0.005,0.01)")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of random seeds per noise level (default: 5)")
    p.add_argument("--max-angle", type=float, default=25.0,
                   help="Max scene rotation in degrees (default: 25)")
    p.add_argument("--max-translation", type=float, default=0.03,
                   help="Max scene translation in metres (default: 0.03)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CAD model
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(args.cad_ply)
    if not pcd.has_points():
        raise SystemExit(f"No points in '{args.cad_ply}'")
    points_model = np.asarray(pcd.points, dtype=np.float64)
    stem = Path(args.cad_ply).stem
    print(f"Loaded {stem}: {len(points_model)} pts")
    print(f"Output: {output_dir}/\n")

    # Parse noise levels
    outlier_ratios = [float(x) for x in args.outlier_ratios.split(",")]
    gaussian_sigmas = [float(x) for x in args.gaussian_sigmas.split(",")]

    total_trials = len(outlier_ratios) * args.seeds + len(gaussian_sigmas) * args.seeds
    print(f"Noise levels: outliers={outlier_ratios}, gaussian={gaussian_sigmas}")
    print(f"Seeds per level: {args.seeds}  →  {total_trials} trials total\n")

    results: list[TrialResult] = []
    trial_num = 0

    for noise_type, levels in [("outliers", outlier_ratios), ("gaussian", gaussian_sigmas)]:
        for level in levels:
            for seed_i in range(args.seeds):
                seed = seed_i * 137 + int(level * 1000)
                trial_num += 1
                print(f"[{trial_num:3d}/{total_trials}] {noise_type}={level} seed={seed}",
                      end=" ", flush=True)
                t0 = time.perf_counter()

                result = run_trial(
                    points_model, stem, noise_type, level, seed,
                    args.max_angle, args.max_translation,
                )
                results.append(result)

                if result.error:
                    print(f"FAIL: {result.error}")
                else:
                    print(
                        f"TEASER ADD-S={result.teaser_add_s_mean*1000:.2f}mm "
                        f"rot={result.teaser_rms_rot_deg:.1f}°  |  "
                        f"ICP ADD-S={result.icp_add_s_mean*1000:.2f}mm "
                        f"rot={result.icp_rms_rot_deg:.1f}°  "
                        f"({time.perf_counter()-t0:.0f}s)"
                    )

    # Aggregate and save
    agg = aggregate(results)
    save_results(results, agg, output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
