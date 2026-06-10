#!/usr/bin/env python3
"""
Batch registration evaluation over multiple RePAIR fragments.

Generates scene pairs with known ground-truth SE(3) perturbations,
registers each scene back onto its CAD model via TEASER++, and computes
ADD-S, Chamfer Distance, and RMS Pose Error for every fragment.

Produces a summary CSV, per-fragment JSON, and a console report with
aggregate statistics (mean, std, min, max across all fragments).

Usage
-----
    # Evaluate all *_ds.ply files in a directory
    python scripts/batch_evaluate.py fragments/ --output results/batch_2026

    # Evaluate specific fragments with custom registration params
    python scripts/batch_evaluate.py RPf_00577_ds.ply RPf_00579_ds.ply \\
        --voxel-size 0.005 --c-threshold 0.005 --output results/batch

    # Generate scene pairs only (no registration) — useful for large batches
    python scripts/batch_evaluate.py fragments/ --output results/scenes \\
        --generate-only

    # Use specific seeds for reproducibility
    python scripts/batch_evaluate.py fragments/ --base-seed 42 --output results/batch

Output files
------------
    <output>_summary.csv      — One row per fragment, all metrics
    <output>_summary.json     — Summary + per-fragment details
    <output>_aggregate.txt    — Human-readable aggregate report
    For each fragment:
        <fragment>_gt.npy     — Ground-truth 4×4 SE(3) matrix
        <fragment>_est.npy    — Estimated 4×4 SE(3) matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

# ── Inline SE(3) utilities (no torch dependency) ──────────────────────


def _random_se3(
    max_angle_deg: float = 25.0,
    max_translation: float = 0.03,
    seed: int | None = None,
) -> tuple[np.ndarray, float, float]:
    r"""
    Generate a random 4×4 SE(3) matrix with bounded rotation + translation.

    Rotation: uniform axis on S², angle up to max_angle_deg (Rodrigues).
    Translation: uniform direction on S², magnitude up to max_translation.

    Returns:
        (T_4x4, angle_deg, t_norm) — SE(3) matrix, rotation in degrees,
        translation Euclidean norm in metres.
    """
    rng = np.random.default_rng(seed)

    # ── uniform axis on S² ──
    z = rng.uniform(-1.0, 1.0)
    theta_h = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    axis = np.array([s * np.cos(theta_h), s * np.sin(theta_h), z])

    # ── bounded angle ──
    angle = rng.uniform(0.0, np.deg2rad(max_angle_deg))

    # ── Rodrigues formula ──
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    # ── uniform direction + magnitude for translation ──
    z2 = rng.uniform(-1.0, 1.0)
    th2 = rng.uniform(0.0, 2.0 * np.pi)
    s2 = np.sqrt(max(0.0, 1.0 - z2 * z2))
    direction = np.array([s2 * np.cos(th2), s2 * np.sin(th2), z2])
    t_norm = rng.uniform(0.0, max_translation)
    t = direction * t_norm

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, float(np.rad2deg(angle)), float(t_norm)


def _transform_points_np(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply SE(3) transform: p' = R·p + t."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


# ── Metric computation ────────────────────────────────────────────────


def compute_add_s(
    points_est: np.ndarray,
    points_model: np.ndarray,
) -> dict:
    """ADD-S (mean, median, p95) in metres."""
    tree = cKDTree(points_model)
    dists, _ = tree.query(points_est, k=1)
    dists = np.asarray(dists, dtype=np.float64)
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p95": float(np.percentile(dists, 95)),
    }


def compute_chamfer(
    points_est: np.ndarray,
    points_model: np.ndarray,
) -> dict:
    """Bidirectional Chamfer distance in metres."""
    t_model = cKDTree(points_model)
    t_est = cKDTree(points_est)
    fwd = float(np.mean(t_model.query(points_est, k=1)[0]))
    bwd = float(np.mean(t_est.query(points_model, k=1)[0]))
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd}


def compute_rms_pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    r"""
    RMS rotation (°) and translation (m) error.

    T_gt: model → scene.  T_est: scene → model.
    Compares T_est with the ground-truth scene→model map  T_gt⁻¹.
    """
    R_est, t_est = T_est[:3, :3].astype(np.float64), T_est[:3, 3].astype(np.float64)
    R_gt, t_gt = T_gt[:3, :3].astype(np.float64), T_gt[:3, 3].astype(np.float64)

    tr = np.trace(R_gt @ R_est)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.rad2deg(np.arccos(cos_theta)))

    t_gt_inv = -R_gt.T @ t_gt
    trans_err = float(np.linalg.norm(t_est - t_gt_inv))

    return {"rotation_deg": rot_err_deg, "translation_m": trans_err}


# ── Per-fragment result ───────────────────────────────────────────────


@dataclass
class FragmentResult:
    fragment: str
    num_points: int
    scene_angle_deg: float
    scene_translation_m: float
    add_s_mean: float = 0.0
    add_s_median: float = 0.0
    add_s_p95: float = 0.0
    chamfer_forward: float = 0.0
    chamfer_backward: float = 0.0
    chamfer_total: float = 0.0
    rms_rotation_deg: float = 0.0
    rms_translation_m: float = 0.0
    registrations_correspondences: int = 0
    registration_runtime_sec: float = 0.0
    tls_certificate: Optional[float] = None
    converged: bool = True
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict:
        return {
            "fragment": self.fragment,
            "num_points": self.num_points,
            "scene_angle_deg": self.scene_angle_deg,
            "scene_translation_m": self.scene_translation_m,
            "add_s_mean": self.add_s_mean,
            "add_s_median": self.add_s_median,
            "add_s_p95": self.add_s_p95,
            "chamfer_forward": self.chamfer_forward,
            "chamfer_backward": self.chamfer_backward,
            "chamfer_total": self.chamfer_total,
            "rms_rotation_deg": self.rms_rotation_deg,
            "rms_translation_m": self.rms_translation_m,
            "registrations_correspondences": self.registrations_correspondences,
            "registration_runtime_sec": self.registration_runtime_sec,
            "tls_certificate": self.tls_certificate,
            "converged": self.converged,
            "error": self.error,
        }


# ── Process single fragment ───────────────────────────────────────────


def process_fragment(
    ply_path: Path,
    seed: int,
    voxel_size: float,
    c_threshold: float,
    noise_bound: float,
    fpfh_radius: float,
    ratio_threshold: float,
    max_angle_deg: float,
    max_translation: float,
    output_dir: Path,
    register: bool,
) -> FragmentResult:
    """
    Full evaluation pipeline for a single fragment.

    1. Load the PLY file
    2. Generate a random SE(3) scene perturbation (ground truth)
    3. Save scene cloud + GT .npy
    4. Run TEASER++ registration (scene → model)
    5. Compute ADD-S, Chamfer, RMS Pose Error
    6. Save estimated pose .npy
    """
    import open3d as o3d  # lazy — parent batch process never imports this

    stem = ply_path.stem
    result = FragmentResult(
        fragment=stem,
        num_points=0,
        scene_angle_deg=0.0,
        scene_translation_m=0.0,
    )

    # ── 1. Load ──
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        result.error = f"Failed to load '{ply_path}'"
        return result
    points_model = np.asarray(pcd.points, dtype=np.float64)
    result.num_points = len(points_model)

    # ── 2. Generate random SE(3) perturbation ──
    T_gt, rot_deg, t_norm = _random_se3(
        max_angle_deg=max_angle_deg,
        max_translation=max_translation,
        seed=seed,
    )
    result.scene_angle_deg = rot_deg
    result.scene_translation_m = t_norm

    gt_path = output_dir / f"{stem}_gt.npy"
    np.save(str(gt_path), T_gt)

    scene_points = _transform_points_np(T_gt, points_model)
    scene_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(scene_points.astype(np.float64))
    )
    scene_path = output_dir / f"{stem}_scene.ply"
    o3d.io.write_point_cloud(str(scene_path), scene_pcd)

    if not register:
        result.add_s_mean = -1.0  # sentinel: not computed
        return result

    # ── 3. Registration ──
    # Lazy import registration module to avoid torch hang on early exit
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from registration.teaser_registration import (  # noqa: E402
        TeaserParams,
        register_teaser,
    )

    pcd_scene_ds = scene_pcd
    pcd_model_ds = pcd
    if voxel_size > 0:
        pcd_scene_ds = pcd_scene_ds.voxel_down_sample(voxel_size=voxel_size)
        pcd_model_ds = pcd_model_ds.voxel_down_sample(voxel_size=voxel_size)

    params = TeaserParams(
        c_threshold=c_threshold,
        noise_bound=noise_bound,
        fpfh_radius=fpfh_radius,
        ratio_threshold=ratio_threshold,
    )

    t0 = time.perf_counter()
    try:
        se3_result = register_teaser(pcd_scene_ds, pcd_model_ds, params)
    except Exception as e:
        result.error = f"Registration failed: {e}"
        return result
    result.registration_runtime_sec = time.perf_counter() - t0
    T_est = se3_result.T
    result.registrations_correspondences = se3_result.num_correspondences
    result.tls_certificate = se3_result.certificate
    result.converged = se3_result.converged

    est_path = output_dir / f"{stem}_est.npy"
    np.save(str(est_path), T_est)

    # ── 4. Evaluate ──
    T_composed = T_est @ T_gt
    points_est = _transform_points_np(T_composed, points_model)

    adds = compute_add_s(points_est, points_model)
    result.add_s_mean = adds["mean"]
    result.add_s_median = adds["median"]
    result.add_s_p95 = adds["p95"]

    chamfer = compute_chamfer(points_est, points_model)
    result.chamfer_forward = chamfer["forward"]
    result.chamfer_backward = chamfer["backward"]
    result.chamfer_total = chamfer["total"]

    rms = compute_rms_pose_error(T_est, T_gt)
    result.rms_rotation_deg = rms["rotation_deg"]
    result.rms_translation_m = rms["translation_m"]

    return result


# ── Aggregate statistics ──────────────────────────────────────────────


def aggregate(results: list[FragmentResult]) -> dict:
    """Compute mean, std, min, max across successful fragments."""
    # Only include fragments where registration actually ran
    registered = [r for r in results if not r.failed and r.registrations_correspondences > 0]
    failed = [r for r in results if r.failed]
    generate_only = [r for r in results if not r.failed and r.registrations_correspondences == 0]

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "num_fragments": len(results),
        "num_registered": len(registered),
        "num_failed": len(failed),
        "num_generate_only": len(generate_only),
        "failed_fragments": [r.fragment for r in failed],
        "add_s_mean": _stats([r.add_s_mean for r in registered]),
        "add_s_p95": _stats([r.add_s_p95 for r in registered]),
        "chamfer_total": _stats([r.chamfer_total for r in registered]),
        "rms_rotation_deg": _stats([r.rms_rotation_deg for r in registered]),
        "rms_translation_m": _stats([r.rms_translation_m for r in registered]),
        "registrations_correspondences": _stats(
            [float(r.registrations_correspondences) for r in registered]
        ),
        "registration_runtime_sec": _stats([r.registration_runtime_sec for r in registered]),
    }


# ── Output ────────────────────────────────────────────────────────────


def save_results(
    results: list[FragmentResult],
    agg: dict,
    output_dir: Path,
) -> None:
    """Write summary CSV, JSON, and plain-text report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ──
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "fragment",
        "num_points",
        "scene_angle_deg",
        "scene_translation_m",
        "add_s_mean",
        "add_s_median",
        "add_s_p95",
        "chamfer_forward",
        "chamfer_backward",
        "chamfer_total",
        "rms_rotation_deg",
        "rms_translation_m",
        "registrations_correspondences",
        "registration_runtime_sec",
        "tls_certificate",
        "converged",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    # ── JSON ──
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "aggregate": agg,
                "fragments": [r.to_dict() for r in results],
            },
            f,
            indent=2,
        )

    # ── Plain-text report ──
    report_path = output_dir / "report.txt"
    lines = []
    lines.append("=" * 72)
    lines.append("  BATCH REGISTRATION EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append(f"  Fragments tested : {agg['num_fragments']}")
    lines.append(f"  Registered       : {agg['num_registered']}")
    lines.append(f"  Failed           : {agg['num_failed']}")
    if agg.get("num_generate_only", 0) > 0:
        lines.append(f"  Generate-only    : {agg['num_generate_only']}")
    if agg["failed_fragments"]:
        lines.append(f"  Failed IDs       : {', '.join(agg['failed_fragments'])}")
    lines.append("")

    # Aggregate table
    if agg["num_registered"] > 0:
        lines.append("-" * 72)
        lines.append(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        lines.append("-" * 72)
        for label, key, fmt in [
            ("ADD-S mean (mm)", "add_s_mean", ".3f"),
            ("ADD-S P95 (mm)", "add_s_p95", ".3f"),
            ("Chamfer total (mm)", "chamfer_total", ".3f"),
            ("RMS rotation (deg)", "rms_rotation_deg", ".2f"),
            ("RMS translation (mm)", "rms_translation_m", ".3f"),
            ("Correspondences", "registrations_correspondences", ".0f"),
            ("Runtime (s)", "registration_runtime_sec", ".2f"),
        ]:
            s = agg[key]
            if "mm" in label and "m" in key:
                scale = 1000.0
            else:
                scale = 1.0
            lines.append(
                f"  {label:<30} "
                f"{s['mean']*scale:>10{fmt}} "
                f"{s['std']*scale:>10{fmt}} "
                f"{s['min']*scale:>10{fmt}} "
                f"{s['max']*scale:>10{fmt}}"
            )
        lines.append("-" * 72)
    lines.append("")

    # Per-fragment table
    lines.append("-" * 72)
    header = (
        f"  {'Fragment':<22} {'ADD-S':>8} {'Chamfer':>8} "
        f"{'RMS Rot':>8} {'RMS Trans':>10} {'#Corr':>6} {'Runtime':>8}"
    )
    lines.append(header)
    lines.append("-" * 72)
    for r in results:
        if r.failed:
            lines.append(f"  {r.fragment:<22} {'FAILED':>8} — {r.error}")
        elif r.add_s_mean < 0:
            # generate-only sentinel
            lines.append(f"  {r.fragment:<22} {'(skipped)':>8} — scene cloud + GT generated")
        else:
            lines.append(
                f"  {r.fragment:<22} "
                f"{r.add_s_mean*1000:>8.3f} "
                f"{r.chamfer_total*1000:>8.3f} "
                f"{r.rms_rotation_deg:>8.2f} "
                f"{r.rms_translation_m*1000:>10.3f} "
                f"{r.registrations_correspondences:>6d} "
                f"{r.registration_runtime_sec:>8.2f}"
            )
    lines.append("-" * 72)

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print(report)

    print(f"\nResults saved to {output_dir}/:")
    print(f"  summary.csv  — per-fragment metrics")
    print(f"  summary.json — full aggregate + per-fragment detail")
    print(f"  report.txt   — human-readable report")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch registration evaluation over multiple RePAIR fragments",
    )
    p.add_argument(
        "fragments",
        nargs="+",
        help="PLY files or directories containing *_ds.ply files",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/batch_eval",
        help="Output directory for results (default: results/batch_eval)",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed; fragment i uses seed = base + i (default: 42)",
    )
    p.add_argument(
        "--max-angle",
        type=float,
        default=25.0,
        help="Maximum scene rotation angle in degrees (default: 25)",
    )
    p.add_argument(
        "--max-translation",
        type=float,
        default=0.03,
        help="Maximum scene translation in metres (default: 0.03)",
    )

    # Registration params
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel downsampling for registration (default: 0.005m)",
    )
    p.add_argument(
        "--c-threshold",
        type=float,
        default=0.005,
        help="TLS truncation threshold (default: 0.005m)",
    )
    p.add_argument(
        "--noise-bound",
        type=float,
        default=0.001,
        help="Sensor noise bound (default: 0.001m)",
    )
    p.add_argument(
        "--fpfh-radius",
        type=float,
        default=0.035,
        help="FPFH search radius (default: 0.035m)",
    )
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.9,
        help="Lowe ratio test threshold (default: 0.9)",
    )

    # Modes
    p.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate scene clouds + GT .npy files; skip registration",
    )
    p.add_argument(
        "--worker",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Internal: invoked by parent batch process
    )
    return p.parse_args()


def _find_fragment_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve user-provided paths to a flat list of PLY files."""
    paths: list[Path] = []
    for entry in args.fragments:
        p = Path(entry)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.ply")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Warning: '{entry}' is neither a file nor a directory — skipping")
    if not paths:
        raise SystemExit("No .ply files found.")
    return paths


def _process_fragment_worker(args: argparse.Namespace) -> None:
    """
    Process a single fragment in worker mode (called via subprocess).

    Prints the FragmentResult as a single JSON line to stdout.
    This isolates Open3D GPU state per fragment.
    """
    import json as _json

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = Path(args.worker)

    result = process_fragment(
        ply_path=ply_path,
        seed=args.base_seed,
        voxel_size=args.voxel_size,
        c_threshold=args.c_threshold,
        noise_bound=args.noise_bound,
        fpfh_radius=args.fpfh_radius,
        ratio_threshold=args.ratio_threshold,
        max_angle_deg=args.max_angle,
        max_translation=args.max_translation,
        output_dir=output_dir,
        register=not args.generate_only,
    )
    # Write result to a file — stdout may not flush before os._exit(0)
    result_path = output_dir / f"{ply_path.stem}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        _json.dump(result.to_dict(), f)
    sys.stdout.write(_json.dumps(result.to_dict()) + "\n")
    sys.stdout.flush()


def main() -> None:
    args = parse_args()

    # ── Worker mode: process single fragment, invoked via subprocess ──
    if args.worker is not None:
        _process_fragment_worker(args)
        return

    fragment_paths = _find_fragment_paths(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(fragment_paths)} fragment(s)")
    if args.generate_only:
        print("Mode: generate-only (scene clouds + GT, no registration)")
    else:
        print("Mode: full pipeline (scene generation + registration + evaluation)")
    print(f"Output directory: {output_dir}\n")

    results: list[FragmentResult] = []
    t_batch_start = time.perf_counter()

    for i, ply_path in enumerate(fragment_paths):
        seed = args.base_seed + i
        stem = ply_path.stem
        print(f"[{i+1}/{len(fragment_paths)}] {stem}  (seed={seed})", flush=True)

        # Build subprocess command — each fragment runs in its own
        # Python process to isolate Open3D GPU state in headless envs.
        cmd = [
            sys.executable, __file__,
            str(ply_path),
            "--worker", str(ply_path),
            "--base-seed", str(seed),
            "--max-angle", str(args.max_angle),
            "--max-translation", str(args.max_translation),
            "--output", str(output_dir),
        ]
        if args.generate_only:
            cmd.append("--generate-only")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per fragment
        )

        # Read result from the JSON file written by the worker
        result = FragmentResult(
            fragment=stem, num_points=0,
            scene_angle_deg=0.0, scene_translation_m=0.0,
        )
        result_path = output_dir / f"{stem}_result.json"
        if proc.returncode != 0 or not result_path.exists():
            result.error = (
                f"Worker exit code {proc.returncode}"
                if proc.returncode != 0
                else "Worker produced no result file"
            )
        else:
            try:
                with open(result_path, encoding="utf-8") as f:
                    parsed = json.load(f)
                result = FragmentResult(**parsed)
            except (json.JSONDecodeError, TypeError) as e:
                result.error = f"Failed to parse worker output: {e}"

        if result.failed:
            print(f"  FAILED: {result.error}")
        elif result.add_s_mean < 0:
            # generate-only sentinel
            print(f"  Scene + GT saved  (rot={result.scene_angle_deg:.1f}°, "
                  f"trans={result.scene_translation_m*1000:.1f}mm)")
        else:
            print(
                f"  ADD-S={result.add_s_mean*1000:.3f}mm  "
                f"Chamfer={result.chamfer_total*1000:.3f}mm  "
                f"RMS rot={result.rms_rotation_deg:.2f}°  "
                f"trans={result.rms_translation_m*1000:.3f}mm  "
                f"({result.registration_runtime_sec:.1f}s)"
            )
        results.append(result)

    t_batch = time.perf_counter() - t_batch_start

    # ── Aggregate ──
    agg = aggregate(results)
    agg["total_runtime_sec"] = t_batch

    # ── Save ──
    save_results(results, agg, output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Force exit to avoid torch CUDA context cleanup hang in
        # headless environments.  All results are saved before this point.
        os._exit(0)
