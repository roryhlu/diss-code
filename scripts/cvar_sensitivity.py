#!/usr/bin/env python3
"""
CVaR sensitivity analysis — sweep α, variance_scale, and N realizations.

Quantifies how the Conditional Value-at-Risk filter's parameters affect
grasp acceptance rate and quality.  Answers the research questions:

  1. How conservative is α=0.05 vs α=0.01 or α=0.20?
  2. What if the epistemic variance is over/under-estimated (scale)?
  3. How many realizations (N) are needed for stable CVaR estimates?

Uses the existing variance cloud (variance_cloud.pcd) and candidate
grasp file (sample_candidates.json).  All force-closure and CVaR logic
is inlined (no Open3D/torch dependency).

Output
------
  cvar_sensitivity.csv      — one row per (α, scale, N) combination
  cvar_sensitivity.json     — full dump
  cvar_sensitivity.png      — multi-panel sensitivity plots:
    [1] acceptance rate vs α, colored by variance_scale
    [2] mean CVaR ε vs α
    [3] acceptance rate vs N
    [4] per-candidate CVaR ε vs α

Usage
-----
    python scripts/cvar_sensitivity.py \
        --variance-cloud variance_cloud.pcd \
        --candidates scripts/sample_candidates.json \
        --output results/cvar_sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


# ── PCD binary reader ────────────────────────────────────────────────


def read_pcd(path: str) -> dict:
    """
    Read a binary PCD v0.7 file.

    Returns dict with keys matching FIELDS: x, y, z, normal_x, etc.
    Each value is a (N,) float64 array.
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"DATA binary":
                break
        # PCD has one more byte (newline after DATA binary line sometimes)
        header_bytes = sum(len(l) for l in header_lines)
        raw = f.read()

    hdr = b"".join(header_lines).decode("ascii", errors="ignore")
    fields = []
    types = []
    n_points = 0
    for line in hdr.split("\n"):
        if line.startswith("FIELDS "):
            fields = line.split()[1:]
        elif line.startswith("TYPE "):
            types = line.split()[1:]
        elif line.startswith("POINTS "):
            n_points = int(line.split()[1])

    type_map = {"F": "f4", "U": "u4", "I": "i4"}
    dtype_spec = [(name, type_map.get(t, "f4")) for name, t in zip(fields, types)]
    expected_bytes = n_points * sum(4 for _ in fields)  # each field is 4 bytes

    if len(raw) < expected_bytes:
        raw = raw + b"\x00" * (expected_bytes - len(raw))
    raw = raw[:expected_bytes]

    data = np.frombuffer(raw, dtype=np.dtype(dtype_spec))
    result = {}
    for name in fields:
        result[name] = data[name].astype(np.float64)
    return result


# ── Geometry helpers ──────────────────────────────────────────────────


def orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / np.linalg.norm(n)
    if abs(n[0]) < 0.9:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def friction_cone_generators(normal: np.ndarray, mu: float, m: int = 8) -> np.ndarray:
    if mu <= 0.0:
        return (normal / np.linalg.norm(normal)).reshape(1, 3)
    alpha = np.arctan(mu)
    n = normal / np.linalg.norm(normal)
    u, v = orthonormal_basis(n)
    thetas = np.linspace(0, 2 * np.pi, m, endpoint=False)
    gens = np.zeros((m, 3))
    for k in range(m):
        gens[k] = (
            np.cos(alpha) * n
            + np.sin(alpha) * (np.cos(thetas[k]) * u + np.sin(thetas[k]) * v)
        )
    return gens


def build_contact_wrench(position: np.ndarray, generators: np.ndarray) -> np.ndarray:
    m = generators.shape[0]
    forces = generators.T
    px, py, pz = position
    skew = np.array([
        [0.0, -pz, py],
        [pz, 0.0, -px],
        [-py, px, 0.0],
    ])
    torques = skew @ forces
    return np.vstack([forces, torques]).astype(np.float64)


def test_fc_lp(W: np.ndarray) -> tuple[bool, float]:
    n_cols = W.shape[1]
    if n_cols < 1:
        return False, 0.0
    n_vars = n_cols + 1
    c = np.zeros(n_vars)
    c[-1] = -1.0
    A_eq = np.zeros((6, n_vars))
    A_eq[:, :n_cols] = W
    A_sum = np.zeros((1, n_vars))
    A_sum[0, :n_cols] = 1.0
    A_eq = np.vstack([A_eq, A_sum])
    b_eq = np.zeros(7)
    b_eq[6] = 1.0
    A_ub = np.zeros((n_cols, n_vars))
    for j in range(n_cols):
        A_ub[j, j] = -1.0
        A_ub[j, -1] = 1.0
    b_ub = np.zeros(n_cols)
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=[(None, None)] * n_vars, method="highs",
    )
    if not res.success:
        return False, 0.0
    eps = float(res.x[-1]) if res.x[-1] is not None else 0.0
    return eps > 1e-9, max(0.0, eps)


def check_fc_at_contacts(
    c1, n1, c2, n2, mu: float, m: int = 8,
) -> tuple[bool, float]:
    gens1 = friction_cone_generators(n1, mu, m)
    gens2 = friction_cone_generators(n2, mu, m)
    W1 = build_contact_wrench(c1, gens1)
    W2 = build_contact_wrench(c2, gens2)
    W = np.hstack([W1, W2])
    ok, eps = test_fc_lp(W)
    return ok, eps


# ── Geometric realization sampling ───────────────────────────────────


def sample_realizations(
    mean_cloud: np.ndarray,
    variance: np.ndarray,
    N: int,
    variance_scale: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample N geometric realizations from isotropic per-point Gaussians.

    Returns (N, M, 3) array where M = len(mean_cloud).
    """
    rng = np.random.default_rng(seed)
    M = len(mean_cloud)
    std = np.sqrt(np.maximum(variance * variance_scale, 0.0))
    noise = rng.normal(0, 1, (N, M, 3)) * std[np.newaxis, :, np.newaxis]
    return mean_cloud[np.newaxis, :, :] + noise


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    neighbours = points[idx]
    mu = neighbours.mean(axis=1, keepdims=True)
    centred = neighbours - mu
    cov = np.einsum("nki,nkj->nij", centred, centred) / (min(k, len(points)) - 1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].copy()
    centroid = points.mean(axis=0)
    dot = np.sum(normals * (points - centroid), axis=1)
    normals[dot < 0] *= -1.0
    ns = np.linalg.norm(normals, axis=1, keepdims=True)
    ns[ns < 1e-12] = 1.0
    return normals / ns


# ── CVaR computation ─────────────────────────────────────────────────


def compute_cvar(epsilon_values: np.ndarray, alpha: float) -> tuple[float, int]:
    sorted_eps = np.sort(epsilon_values)
    k = max(int(np.ceil(alpha * len(sorted_eps))), 1)
    return float(sorted_eps[:k].mean()), k


# ── Single-parameter-combination evaluation ──────────────────────────


def evaluate_single(
    candidates: list[dict],
    mean_cloud: np.ndarray,
    variance: np.ndarray,
    base_normals: np.ndarray,
    mu: float,
    cvar_alpha: float,
    variance_scale: float,
    num_realizations: int,
    seed: int,
) -> dict:
    """Evaluate all candidates at one parameter combination."""
    accepted = 0
    total = len(candidates)
    cvar_scores = []

    for cand in candidates:
        c1 = np.array(cand["contact1"], dtype=np.float64)
        c2 = np.array(cand["contact2"], dtype=np.float64)

        # Find closest point indices for normals
        from scipy.spatial import cKDTree
        tree = cKDTree(mean_cloud)
        _, i1 = tree.query(c1, k=1)
        _, i2 = tree.query(c2, k=1)
        n1 = base_normals[i1]
        n2 = base_normals[i2]

        # Baseline FC
        ok_base, eps_base = check_fc_at_contacts(c1, n1, c2, n2, mu)
        if not ok_base:
            cvar_scores.append(-1.0)  # baseline failed
            continue

        # Sample realizations
        realizations = sample_realizations(
            mean_cloud, variance, num_realizations, variance_scale,
            seed=seed + int(cand.get("id", 0)) if seed else None,
        )

        eps_values = np.zeros(num_realizations)
        for k in range(num_realizations):
            real_pts = realizations[k]
            real_normals = estimate_normals(real_pts)
            _, eps_k = check_fc_at_contacts(c1, real_normals[i1], c2, real_normals[i2], mu)
            eps_values[k] = eps_k

        cvar_val, _ = compute_cvar(eps_values, cvar_alpha)

        if cvar_val > 0:
            accepted += 1
            cvar_scores.append(float(cvar_val))
        else:
            cvar_scores.append(-1.0)  # CVaR failed

    return {
        "cvar_alpha": cvar_alpha,
        "variance_scale": variance_scale,
        "num_realizations": num_realizations,
        "total_candidates": total,
        "accepted": accepted,
        "acceptance_rate": accepted / total if total > 0 else 0.0,
        "mean_cvar_of_accepted": float(np.mean([s for s in cvar_scores if s > 0])) if accepted > 0 else 0.0,
        "std_cvar_of_accepted": float(np.std([s for s in cvar_scores if s > 0])) if accepted > 0 else 0.0,
        "per_candidate_scores": cvar_scores,
    }


# ── Batch sweep ─────────────────────────────────────────────────────


def run_sweep(
    candidates: list[dict],
    mean_cloud: np.ndarray,
    variance: np.ndarray,
    mu: float,
    alphas: list[float],
    scales: list[float],
    n_realizations_list: list[int],
    seed: int,
    output_dir: Path,
) -> list[dict]:
    base_normals = estimate_normals(mean_cloud)
    results: list[dict] = []
    total = len(alphas) * len(scales) * len(n_realizations_list)
    count = 0

    for alpha in alphas:
        for scale in scales:
            for n_real in n_realizations_list:
                count += 1
                print(f"  [{count:3d}/{total}] α={alpha:.3f}  "
                      f"scale={scale:.2f}  N={n_real:4d}  ", end="", flush=True)
                t0 = time.perf_counter()

                result = evaluate_single(
                    candidates, mean_cloud, variance, base_normals,
                    mu, alpha, scale, n_real, seed,
                )
                result["runtime_s"] = time.perf_counter() - t0
                results.append(result)

                print(f"→ accepted={result['accepted']}/{result['total_candidates']}  "
                      f"mean_CVaR={result['mean_cvar_of_accepted']:.6f}  "
                      f"({result['runtime_s']:.1f}s)")

    return results


# ── Output ───────────────────────────────────────────────────────────


def save_sweep_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "cvar_sensitivity.csv"
    csv_keys = [
        "cvar_alpha", "variance_scale", "num_realizations",
        "total_candidates", "accepted", "acceptance_rate",
        "mean_cvar_of_accepted", "std_cvar_of_accepted", "runtime_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # JSON
    json_path = output_dir / "cvar_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # Plot
    _plot_sensitivity(results, output_dir)

    print(f"\nResults saved to {output_dir}/")
    print(f"  cvar_sensitivity.csv")
    print(f"  cvar_sensitivity.json")
    print(f"  cvar_sensitivity.png")


def _plot_sensitivity(results: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    # Group by (scale, N)
    unique = sorted(set((r["variance_scale"], r["num_realizations"]) for r in results))

    # Find the "default" N value to plot α sweeps against
    n_values = sorted(set(r["num_realizations"] for r in results))
    default_n = n_values[len(n_values) // 2] if n_values else 100

    # Find distinct scales
    scales = sorted(set(r["variance_scale"] for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # [1] Acceptance rate vs α (colored by scale, at default N)
    ax = axes[0, 0]
    for scale in scales:
        subset = [r for r in results
                  if r["variance_scale"] == scale and r["num_realizations"] == default_n]
        subset.sort(key=lambda r: r["cvar_alpha"])
        x = [r["cvar_alpha"] for r in subset]
        y = [r["acceptance_rate"] * 100 for r in subset]
        ax.plot(x, y, marker="o", label=f"scale={scale}")
    ax.set_xlabel("CVaR α (tail fraction)")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title(f"Acceptance rate vs CVaR α  (N={default_n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(r["cvar_alpha"] for r in results) * 1.05)

    # [2] Mean CVaR ε vs α
    ax = axes[0, 1]
    for scale in scales:
        subset = [r for r in results
                  if r["variance_scale"] == scale and r["num_realizations"] == default_n]
        subset.sort(key=lambda r: r["cvar_alpha"])
        x = [r["cvar_alpha"] for r in subset]
        y = [r["mean_cvar_of_accepted"] for r in subset]
        ax.plot(x, y, marker="s", label=f"scale={scale}")
    ax.set_xlabel("CVaR α (tail fraction)")
    ax.set_ylabel("Mean CVaR ε of accepted grasps")
    ax.set_title(f"Grasp quality vs CVaR α  (N={default_n})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [3] Acceptance rate vs N (at default α)
    default_alpha = 0.05
    ax = axes[1, 0]
    for scale in scales:
        subset = [r for r in results
                  if r["variance_scale"] == scale and r["cvar_alpha"] == default_alpha]
        subset.sort(key=lambda r: r["num_realizations"])
        x = [r["num_realizations"] for r in subset]
        y = [r["acceptance_rate"] * 100 for r in subset]
        ax.plot(x, y, marker="o", label=f"scale={scale}")
    ax.set_xlabel("Number of realizations (N)")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title(f"Convergence: acceptance vs N  (α={default_alpha})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # [4] Per-candidate CVaR vs α (at default N, default scale)
    ax = axes[1, 1]
    default_scale = 1.0
    subset = [r for r in results
              if r["variance_scale"] == default_scale
              and r["num_realizations"] == default_n]
    subset.sort(key=lambda r: r["cvar_alpha"])
    n_candidates = subset[0]["total_candidates"] if subset else 0
    for cand_idx in range(n_candidates):
        y_vals = []
        for r in subset:
            scores = r.get("per_candidate_scores", [])
            if cand_idx < len(scores):
                y_vals.append(max(0, scores[cand_idx]))
            else:
                y_vals.append(0)
        ax.plot([r["cvar_alpha"] for r in subset], y_vals,
                marker=".", label=f"candidate {cand_idx+1}" if cand_idx < 5 else "")
    ax.set_xlabel("CVaR α (tail fraction)")
    ax.set_ylabel("Per-candidate CVaR ε")
    ax.set_title(f"Individual grasp quality vs α  (scale=1, N={default_n})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "cvar_sensitivity.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CVaR sensitivity analysis — sweep α, variance_scale, and N",
    )
    p.add_argument("--variance-cloud", type=str, default="variance_cloud.pcd",
                   help="Path to MC Dropout variance PCD (default: variance_cloud.pcd)")
    p.add_argument("--candidates", type=str, default="scripts/sample_candidates.json",
                   help="Path to grasp candidates JSON (default: scripts/sample_candidates.json)")
    p.add_argument("--output", type=str, default="results/cvar_sensitivity",
                   help="Output directory (default: results/cvar_sensitivity)")
    p.add_argument("--mu", type=float, default=0.5,
                   help="Friction coefficient (default: 0.5)")
    p.add_argument("--alphas", type=str, default="0.01,0.025,0.05,0.10,0.15,0.20",
                   help="CVaR α values (default: 0.01,0.025,0.05,0.10,0.15,0.20)")
    p.add_argument("--scales", type=str, default="0.25,0.5,1.0,2.0,4.0",
                   help="Variance scale factors (default: 0.25,0.5,1.0,2.0,4.0)")
    p.add_argument("--num-realizations", type=str, default="25,50,100,200,500",
                   help="N realizations to test (default: 25,50,100,200,500)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)

    # ── Load data ──
    print(f"Loading variance cloud: {args.variance_cloud}")
    pcd = read_pcd(args.variance_cloud)
    mean_cloud = np.column_stack([pcd["x"], pcd["y"], pcd["z"]])
    variance = np.abs(pcd["normal_x"])  # variance stored in normal_x
    variance = np.maximum(variance, 0.0)
    print(f"  Points: {len(mean_cloud)}")
    print(f"  Variance range: [{variance.min():.6f}, {variance.max():.6f}]")
    print(f"  Variance mean:  {variance.mean():.6f}")

    # ── Centre the cloud — contact points are in object-local frame
    #     while the variance cloud is in scanner frame (X ≈ −1028).
    centroid = mean_cloud.mean(axis=0)
    mean_cloud_centred = mean_cloud - centroid
    print(f"  Centroid: {centroid}")
    print(f"  Centred range: [{mean_cloud_centred.min():.3f}, {mean_cloud_centred.max():.3f}]")

    print(f"\nLoading candidates: {args.candidates}")
    with open(args.candidates) as f:
        candidates = json.load(f)
    print(f"  Candidates: {len(candidates)}")

    # ── Parse sweep parameters ──
    alphas = [float(x) for x in args.alphas.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    n_real_list = [int(x) for x in args.num_realizations.split(",")]

    total_combos = len(alphas) * len(scales) * len(n_real_list)
    print(f"\nSweep: {len(alphas)} α × {len(scales)} scales × "
          f"{len(n_real_list)} N = {total_combos} combinations")
    print(f"  α ∈ {alphas}")
    print(f"  variance_scale ∈ {scales}")
    print(f"  N ∈ {n_real_list}")
    print()

    # ── Run sweep ──
    results = run_sweep(
        candidates, mean_cloud_centred, variance, args.mu,
        alphas, scales, n_real_list, args.seed, output_dir,
    )

    # ── Save ──
    save_sweep_results(results, output_dir)

    # ── Interpretation ──
    print("\n=== Key Findings ===")
    for scale in [1.0]:
        for alpha in [0.01, 0.05, 0.20]:
            subset = [r for r in results
                      if r["variance_scale"] == scale and r["cvar_alpha"] == alpha
                      and r["num_realizations"] == 100]
            if subset:
                r = subset[0]
                print(f"  α={alpha:.2f} (scale={scale}): "
                      f"{r['accepted']}/{r['total_candidates']} accepted "
                      f"({r['acceptance_rate']*100:.0f}%), "
                      f"mean CVaR ε = {r['mean_cvar_of_accepted']:.6f}")


if __name__ == "__main__":
    main()
