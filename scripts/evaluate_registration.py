#!/usr/bin/env python3
"""
Quantitative registration evaluation harness for RePAIR fragments.

Computes three metrics defined in evaluation metrics (DISSERTATION_GUIDE.md):

  ADD-S   — Average Distance, Symmetric-Defined (per-point nearest-neighbour)
  Chamfer — Bidirectional geometric distance (forward + backward)
  RMS Pose Error — Rotation and translation error against ground-truth SE(3)

Mathematical definitions
------------------------

**ADD-S (model frame)**
    Let  P = {p_i} ⊂ ℝ³  be the CAD model point cloud.
    Let  T_gt :  model → scene     (used to generate the scene cloud).
    Let  T_est : scene  → model    (output of TEASER++ registration).
    The composed transform  T = T_est · T_gt  maps  model → model.
    If registration is perfect,  T = I₄.

    P_est = T · P
    ADD-S(T_hat, T_gt) = (1/|P|) · Σ_i  min_j ‖ P_est[i] − P[j] ‖₂

**Chamfer Distance**
    Between P_est and the original model P:
        d₁ = (1/|P_est|) · Σ_{a∈P_est} min_{b∈P} ‖a − b‖₂    (forward)
        d₂ = (1/|P|)     · Σ_{b∈P}     min_{a∈P_est} ‖b − a‖₂  (backward)
        d_Chamfer(P_est, P) = d₁ + d₂

**RMS Pose Error**
    Ground-truth scene→model mapping:
        T_gt⁻¹ = [R_gt^\top   −R_gt^\top · t_gt]
                 [0            1                 ]
    Compare estimated T_est with T_gt⁻¹:
        RMS_rot  = arccos( (tr(R_gt · R_est) − 1) / 2 )   [rad]
        RMS_trans = ‖ t_est  +  R_gt^\top · t_gt ‖₂       [m]

Usage
-----
    # 1. Generate scene pair with known ground truth
    python scripts/create_scene_pair.py RPf_00577_ds.ply --seed 42

    # 2. Register the scene back onto the model, saving the estimated SE(3)
    python scripts/teaser_register.py RPf_00577_ds_scene.ply RPf_00577_ds.ply \\
        --voxel-size 0.005 --output aligned.ply

    # 3a. Evaluate using pre-computed .npy pose
    python scripts/evaluate_registration.py \\
        --cad RPf_00577_ds.ply \\
        --scene RPf_00577_ds_scene.ply \\
        --ground-truth RPf_00577_ds_scene_gt.npy \\
        --estimated RPf_00577_ds_scene_gt.npy   # example: load previously saved

    # 3b. Run registration inline and evaluate
    python scripts/evaluate_registration.py \\
        --cad RPf_00577_ds.ply \\
        --scene RPf_00577_ds_scene.ply \\
        --ground-truth RPf_00577_ds_scene_gt.npy \\
        --register --voxel-size 0.005 \\
        --output evaluation_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# Inline SE(3) utilities — keep this script self-contained so it runs
# without importing registration/, which pulls in torch and may hang
# on resource cleanup in headless environments.


def _transform_points_np(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    r"""Apply SE(3) transform to NumPy points:  p' = R·p + t."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def _extract_rt_np(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split 4×4 SE(3) into (R_3x3, t_3)."""
    return T[:3, :3], T[:3, 3]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Quantitative evaluation of a registration against ground truth."""

    add_s: float
    """Average Distance, Symmetric-Defined (metres). 0 = perfect."""

    chamfer_forward: float
    """Mean distance from estimated to model (metres)."""

    chamfer_backward: float
    """Mean distance from model to estimated (metres)."""

    chamfer: float
    """Bidirectional Chamfer distance = forward + backward (metres)."""

    rms_rotation_deg: float
    """Rotation error between T_est and T_gt⁻¹ (degrees)."""

    rms_translation: float
    """Translation error between t_est and T_gt⁻¹ translation (metres)."""

    num_points: int
    """Number of points used in the evaluation (after downsampling)."""

    runtime_sec: float = 0.0
    """Wall-clock time for metric computation."""

    add_s_median: float = 0.0
    """Median of the ADD-S per-point distances."""

    add_s_p95: float = 0.0
    """95th percentile ADD-S distance."""

    def summary_table(self) -> str:
        """Format a human-readable summary table."""
        return (
            f"╔═══════════════════════╤══════════════════╗\n"
            f"║  Evaluation Metric    │  Value           ║\n"
            f"╠═══════════════════════╪══════════════════╣\n"
            f"║  ADD-S (mean)         │ {self.add_s:>10.6f} m  ║\n"
            f"║  ADD-S (median)       │ {self.add_s_median:>10.6f} m  ║\n"
            f"║  ADD-S (P95)          │ {self.add_s_p95:>10.6f} m  ║\n"
            f"║  Chamfer forward      │ {self.chamfer_forward:>10.6f} m  ║\n"
            f"║  Chamfer backward     │ {self.chamfer_backward:>10.6f} m  ║\n"
            f"║  Chamfer (total)      │ {self.chamfer:>10.6f} m  ║\n"
            f"║  RMS rotation error   │ {self.rms_rotation_deg:>10.4f} °  ║\n"
            f"║  RMS translation err  │ {self.rms_translation:>10.6f} m  ║\n"
            f"╠═══════════════════════╧══════════════════╣\n"
            f"║  Points evaluated     │ {self.num_points:>10d}     ║\n"
            f"║  Compute time         │ {self.runtime_sec:>10.3f} s   ║\n"
            f"╚════════════════════════════════════════════╝"
        )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_add_s(
    points_est: np.ndarray,
    points_model: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    r"""
    Compute ADD-S (Average Distance, Symmetric-Defined).

    For each estimated point, finds the closest model point (Euclidean
    nearest-neighbour).  The metric is the mean of these minimum distances.

    ADD-S(T_hat, T_gt) = (1/N) · Σᵢ  minⱼ ‖ P_est[i] − P_model[j] ‖₂

    Args:
        points_est:   (N, 3) estimated point positions in model frame.
        points_model: (M, 3) reference model point cloud.

    Returns:
        (mean, median, p95, per_point_distances) — all in metres.
    """
    tree = cKDTree(points_model)
    dists, _ = tree.query(points_est, k=1, workers=-1)
    dists = np.asarray(dists, dtype=np.float64)
    return (
        float(np.mean(dists)),
        float(np.median(dists)),
        float(np.percentile(dists, 95)),
        dists,
    )


def compute_chamfer(
    points_est: np.ndarray,
    points_model: np.ndarray,
) -> tuple[float, float, float]:
    r"""
    Compute Chamfer distance (bidirectional nearest-neighbour).

        d₁ = (1/|A|) · Σ_{a∈A} min_{b∈B} ‖a − b‖₂    (forward)
        d₂ = (1/|B|) · Σ_{b∈B} min_{a∈A} ‖b − a‖₂    (backward)
        d_Chamfer(A, B) = d₁ + d₂

    Args:
        points_est:   (N, 3) estimated positions.
        points_model: (M, 3) reference model.

    Returns:
        (chamfer_forward, chamfer_backward, chamfer_total) in metres.
    """
    tree_model = cKDTree(points_model)
    tree_est = cKDTree(points_est)

    forward_dists, _ = tree_model.query(points_est, k=1, workers=-1)
    forward = float(np.mean(forward_dists))

    backward_dists, _ = tree_est.query(points_model, k=1, workers=-1)
    backward = float(np.mean(backward_dists))

    return forward, backward, forward + backward


def compute_rms_pose_error(
    T_est: np.ndarray,
    T_gt: np.ndarray,
) -> tuple[float, float]:
    r"""
    Compute RMS pose error between estimated and ground-truth transforms.

    T_gt maps model→scene, T_est maps scene→model.
    The ground-truth scene→model mapping is T_gt⁻¹.

    Rotation error:
        θ = arccos( (tr(R_gt · R_est) − 1) / 2 )   [rad]

    Translation error:
        ε_t = ‖ t_est + R_gt^\top · t_gt ‖₂          [m]

    Derivation:
        T_gt⁻¹ = [R_gt^\top   −R_gt^\top·t_gt]
                 [0            1              ]
        t_gt⁻¹ = −R_gt^\top · t_gt
        ε_t = ‖ t_est − t_gt⁻¹ ‖₂ = ‖ t_est + R_gt^\top·t_gt ‖₂

    Args:
        T_est: 4×4 SE(3) estimated scene→model transform.
        T_gt:  4×4 SE(3) ground-truth model→scene transform.

    Returns:
        (rotation_error_deg, translation_error_m).
    """
    R_est, t_est = _extract_rt_np(T_est)
    R_gt, t_gt = _extract_rt_np(T_gt)

    R_gt = np.asarray(R_gt, dtype=np.float64)
    t_gt = np.asarray(t_gt, dtype=np.float64)
    R_est = np.asarray(R_est, dtype=np.float64)
    t_est = np.asarray(t_est, dtype=np.float64)

    # Rotation error via trace formula
    tr = np.trace(R_gt @ R_est)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    rot_err_rad = float(np.arccos(cos_theta))
    rot_err_deg = float(np.rad2deg(rot_err_rad))

    # Translation error: compare t_est with T_gt⁻¹ translation
    t_gt_inv = -R_gt.T @ t_gt
    trans_err = float(np.linalg.norm(t_est - t_gt_inv))

    return rot_err_deg, trans_err


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def evaluate_registration(
    pcd_cad: o3d.geometry.PointCloud,
    pcd_scene: o3d.geometry.PointCloud,
    T_gt: np.ndarray,
    T_est: np.ndarray,
    center: bool = False,
) -> EvaluationResult:
    r"""
    Evaluate a registration estimate against ground truth.

    Computes all three metric families (ADD-S, Chamfer, RMS Pose Error)
    using the model-frame formulation:

        P_est = T_est · T_gt · P_cad

    If registration is perfect (T_est = T_gt⁻¹), then P_est = P_cad
    and all metrics evaluate to zero.

    If ``center=True``, the model point cloud is first centred at its
    centroid so that ADD-S and Chamfer measure object-local geometric
    error independent of the global reference frame.  This is the
    recommended mode when the model coordinates are far from the world
    origin (e.g., raw scanner coordinates).

    Args:
        pcd_cad:   CAD / model point cloud (reference).
        pcd_scene: Scene point cloud (not directly used in metric
                   computation, but kept for future scene-frame metrics).
        T_gt:      4×4 SE(3) ground-truth model→scene transform.
        T_est:     4×4 SE(3) estimated scene→model transform.
        center:    If True, translate model to centroid before computing
                   ADD-S and Chamfer.  RMS Pose Error is unaffected.

    Returns:
        EvaluationResult with all metrics.
    """
    t_start = time.perf_counter()
    points_model = np.asarray(pcd_cad.points, dtype=np.float64)

    # Compose: model → (T_gt) → scene → (T_est) → model
    # T_composed = T_est @ T_gt  maps model→model
    T_composed = T_est @ T_gt
    R_comp = T_composed[:3, :3]
    t_comp = T_composed[:3, 3]

    points_est = _transform_points_np(T_composed, points_model)

    # ── Optional centring ──
    # When the model sits far from the world origin, a tiny rotation
    # error is amplified by the lever-arm effect:  ‖(R−I)·p‖ ≈ θ·‖p‖.
    # Centring removes this global-frame artefact so ADD-S and Chamfer
    # measure the actual object-local registration quality.
    if center:
        # Centre the model at its centroid, then apply ONLY the rotation
        # component of T_composed.  This isolates the local geometric
        # deformation from pure SO(3) error — independent of the object's
        # global position (the lever-arm effect).
        # 
        # In the centred frame both clouds have zero mean, so the ADD-S
        # measures  ‖(R − I)·p'_i‖₂ ≈ ‖ω × p'_i‖₂  which is bounded by
        #   θ · max(‖p'_i‖)  for a fragment of diameter D ≈ 2·r_max.
        centroid = points_model.mean(axis=0)
        points_model_centred = points_model - centroid
        points_est_local = points_model_centred @ R_comp.T
        _pts_model = points_model_centred
        _pts_est = points_est_local
    else:
        _pts_model = points_model
        _pts_est = points_est

    # ── ADD-S ──
    add_s_mean, add_s_median, add_s_p95, _ = compute_add_s(
        _pts_est, _pts_model
    )

    # ── Chamfer ──
    cf_fwd, cf_bwd, cf_total = compute_chamfer(_pts_est, _pts_model)

    # ── RMS Pose Error ──
    rms_rot, rms_trans = compute_rms_pose_error(T_est, T_gt)

    runtime = time.perf_counter() - t_start

    return EvaluationResult(
        add_s=add_s_mean,
        add_s_median=add_s_median,
        add_s_p95=add_s_p95,
        chamfer_forward=cf_fwd,
        chamfer_backward=cf_bwd,
        chamfer=cf_total,
        rms_rotation_deg=rms_rot,
        rms_translation=rms_trans,
        num_points=len(_pts_model),
        runtime_sec=runtime,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_or_register(
    pcd_scene: o3d.geometry.PointCloud,
    pcd_cad: o3d.geometry.PointCloud,
    estimated_path: Optional[str],
    register: bool,
    voxel_size: float,
    c_threshold: float,
    noise_bound: float,
    fpfh_radius: float,
    ratio_threshold: float,
) -> np.ndarray:
    """Load T_est from .npy or compute via TEASER++ inline."""
    if estimated_path is not None:
        T_est = np.load(estimated_path)
        if T_est.shape != (4, 4):
            raise ValueError(
                f"Estimated pose file must contain a 4×4 matrix, got {T_est.shape}"
            )
        print(f"Loaded estimated pose from {estimated_path}")
        return T_est

    if register:
        # Lazy import — only when --register is used (avoids torch GPU
        # cleanup hang on script exit when registration is not needed).
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from registration.teaser_registration import (  # noqa: E402
            TeaserParams,
            register_teaser,
        )

        print("Running TEASER++ registration inline ...")
        if voxel_size > 0:
            pcd_scene_ds = pcd_scene.voxel_down_sample(voxel_size=voxel_size)
            pcd_cad_ds = pcd_cad.voxel_down_sample(voxel_size=voxel_size)
            print(f"  Downsampled: scene {len(pcd_scene_ds.points)} pts, "
                  f"cad {len(pcd_cad_ds.points)} pts  (voxel={voxel_size}m)")
        else:
            pcd_scene_ds = pcd_scene
            pcd_cad_ds = pcd_cad

        params = TeaserParams(
            c_threshold=c_threshold,
            noise_bound=noise_bound,
            fpfh_radius=fpfh_radius,
            ratio_threshold=ratio_threshold,
        )
        result = register_teaser(pcd_scene_ds, pcd_cad_ds, params)
        print(f"  Registration: {result}")
        return result.T

    raise ValueError(
        "Provide --estimated <path.npy> or use --register for inline registration."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantitative registration evaluation — ADD-S, Chamfer, RMS Pose",
    )
    # Required
    p.add_argument("--cad", required=True, type=str,
                   help="CAD / model point cloud (PLY, PCD, XYZ)")
    p.add_argument("--scene", required=True, type=str,
                   help="Scene point cloud (PLY, PCD, XYZ)")
    p.add_argument("--ground-truth", required=True, type=str,
                   help="Path to ground-truth 4×4 SE(3) .npy file "
                        "(output of create_scene_pair.py)")

    # Pose source (mutually exclusive in practice; estimated takes priority)
    p.add_argument("--estimated", type=str, default=None,
                   help="Path to estimated 4×4 SE(3) .npy file. "
                        "If omitted, use --register to compute inline.")
    p.add_argument("--register", action="store_true",
                   help="Run TEASER++ registration inline to produce T_est.")

    # Registration params (only used with --register)
    p.add_argument("--voxel-size", type=float, default=0.005,
                   help="Voxel downsampling size for registration (default: 0.005m)")
    p.add_argument("--c-threshold", type=float, default=0.005,
                   help="TLS truncation threshold (default: 0.005m)")
    p.add_argument("--noise-bound", type=float, default=0.001,
                   help="Sensor noise bound (default: 0.001m)")
    p.add_argument("--fpfh-radius", type=float, default=0.035,
                   help="FPFH search radius (default: 0.035m)")
    p.add_argument("--ratio-threshold", type=float, default=0.9,
                   help="Lowe ratio test threshold (default: 0.9)")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Save evaluation results as JSON")
    p.add_argument("--center", action="store_true",
                   help="Centre the model at its centroid before computing "
                        "ADD-S and Chamfer (recommended for scanner-framed data)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data ──
    print("=== Loading point clouds ===")
    pcd_cad = o3d.io.read_point_cloud(args.cad)
    pcd_scene = o3d.io.read_point_cloud(args.scene)
    if not pcd_cad.has_points():
        raise ValueError(f"No points loaded from '{args.cad}'")
    if not pcd_scene.has_points():
        raise ValueError(f"No points loaded from '{args.scene}'")
    print(f"  CAD   : {len(pcd_cad.points)} points")
    print(f"  Scene : {len(pcd_scene.points)} points")

    # ── Load ground truth ──
    T_gt = np.load(args.ground_truth)
    if T_gt.shape != (4, 4):
        raise ValueError(
            f"Ground-truth file must be 4×4, got {T_gt.shape}"
        )
    print(f"\n  Ground-truth loaded from {args.ground_truth}")

    # ── Obtain estimated pose ──
    T_est = _load_or_register(
        pcd_scene, pcd_cad,
        estimated_path=args.estimated,
        register=args.register,
        voxel_size=args.voxel_size,
        c_threshold=args.c_threshold,
        noise_bound=args.noise_bound,
        fpfh_radius=args.fpfh_radius,
        ratio_threshold=args.ratio_threshold,
    )

    # ── Evaluate ──
    print("\n=== Computing evaluation metrics ===")
    result = evaluate_registration(pcd_cad, pcd_scene, T_gt, T_est,
                                   center=args.center)
    print(result.summary_table())

    # ── Interpret ──
    _print_interpretation(result)

    # ── Save ──
    if args.output:
        out = args.output
        serialisable = {
            "add_s": result.add_s,
            "add_s_median": result.add_s_median,
            "add_s_p95": result.add_s_p95,
            "chamfer_forward": result.chamfer_forward,
            "chamfer_backward": result.chamfer_backward,
            "chamfer": result.chamfer,
            "rms_rotation_deg": result.rms_rotation_deg,
            "rms_translation": result.rms_translation,
            "num_points": result.num_points,
            "runtime_sec": result.runtime_sec,
        }
        with open(out, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nResults saved to {out}")


def _print_interpretation(result: EvaluationResult) -> None:
    """Provide qualitative interpretation of the evaluation results."""
    interpretations: list[str] = []

    # ADD-S interpretation
    if result.add_s < 0.001:
        interpretations.append("ADD-S: Excellent (<1 mm).")
    elif result.add_s < 0.005:
        interpretations.append("ADD-S: Good (<5 mm).")
    elif result.add_s < 0.01:
        interpretations.append("ADD-S: Acceptable (<10 mm).")
    else:
        interpretations.append("ADD-S: Poor (>10 mm) — registration likely failed.")

    # Chamfer
    if result.chamfer < 0.001:
        interpretations.append("Chamfer: Excellent (<1 mm).")
    elif result.chamfer < 0.005:
        interpretations.append("Chamfer: Good (<5 mm).")
    elif result.chamfer < 0.01:
        interpretations.append("Chamfer: Acceptable (<10 mm).")
    else:
        interpretations.append("Chamfer: Poor (>10 mm) — clouds may not overlap well.")

    # RMS rotation
    if result.rms_rotation_deg < 1.0:
        interpretations.append(f"Rotation error: Excellent ({result.rms_rotation_deg:.2f}°).")
    elif result.rms_rotation_deg < 5.0:
        interpretations.append(f"Rotation error: Good ({result.rms_rotation_deg:.2f}°).")
    elif result.rms_rotation_deg < 15.0:
        interpretations.append(f"Rotation error: Acceptable ({result.rms_rotation_deg:.2f}°).")
    else:
        interpretations.append(f"Rotation error: Poor ({result.rms_rotation_deg:.2f}°).")

    # RMS translation
    if result.rms_translation < 0.002:
        interpretations.append(f"Translation error: Excellent ({result.rms_translation*1000:.1f} mm).")
    elif result.rms_translation < 0.01:
        interpretations.append(f"Translation error: Good ({result.rms_translation*1000:.1f} mm).")
    elif result.rms_translation < 0.02:
        interpretations.append(f"Translation error: Acceptable ({result.rms_translation*1000:.1f} mm).")
    else:
        interpretations.append(f"Translation error: Poor ({result.rms_translation*1000:.1f} mm).")

    print("\nInterpretation:")
    for line in interpretations:
        print(f"  {line}")


if __name__ == "__main__":
    main()
