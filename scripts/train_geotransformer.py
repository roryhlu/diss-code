#!/usr/bin/env python3
"""
GeoTransformer Self-Supervised Training on RePAIR Archaeological Fragments.

Trains the GeoTransformer point completion network with heavy data
augmentation to produce a checkpoint that enables the MC Dropout
epistemic uncertainty pipeline (Module 1.4) for CVaR grasp filtering.

Training task:
    Input  (N, 6):  [x,y,z, nx,ny,nz]  — noisy, rotated patch
    Target (N, 3):  [x,y,z]            — clean rotated positions

Augmentations per sample:
    - Random SO(3) rotation of positions + normals
    - Gaussian noise on positions (σ = bbox_diag * 0.005)
    - Random point dropout (p = 10%)

Output:
    checkpoints/geotransformer_best.pt   — lowest validation loss
    checkpoints/geotransformer_latest.pt — final epoch

Usage:
    python scripts/train_geotransformer.py RPf_00577_ds.ply RPf_00579_ds.ply \
        --patch-size 512 --batch-size 8 --epochs 200 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uncertainty.geotransformer import GeoTransformer  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_fragment(
    path: str,
    *,
    auto_normal_radius: float | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, np.ndarray | None, float | None]:
    """
    Load a PLY/OBJ fragment and return (N, 6) tensor [x,y,z, nx,ny,nz].

    Normals are estimated via Open3D's PCA k-NN if absent.
    If normalize=True, points are centered and scaled so the bounding-box
    diagonal ≈ 2.0 (coordinates in [-1, +1]).  Returns (data, centroid, scale)
    so the inverse transform can be applied later.
    """
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"No points in '{path}'")

    if not pcd.has_normals():
        pts_np = np.asarray(pcd.points)
        if auto_normal_radius is None and len(pts_np) > 1:
            from scipy.spatial import cKDTree as _KD
            tree = _KD(pts_np)
            dists, _ = tree.query(pts_np, k=2)
            med = float(np.median(dists[:, 1]))
            auto_normal_radius = max(med * 5.0, 1e-6)
        elif auto_normal_radius is None:
            auto_normal_radius = 0.01
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=auto_normal_radius, max_nn=30,
            )
        )

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    bbox = points.max(axis=0) - points.min(axis=0)

    centroid: np.ndarray | None = None
    scale: float | None = None

    if normalize:
        centroid = points.mean(axis=0)
        diag = float(np.linalg.norm(bbox))
        scale = max(diag * 0.5, 1e-8)  # maps to roughly [-1, +1]
        points = (points - centroid) / scale

    data = np.column_stack([points, normals])
    print(f"  {Path(path).name}: {len(points):,} pts"
          f" | bbox [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}]",
          end="")
    if centroid is not None:
        print(f" | cntr ({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})"
              f" | scl {scale:.1f}")
    else:
        print()
    return torch.from_numpy(data), centroid, scale


# ---------------------------------------------------------------------------
# Patch dataset with augmentation
# ---------------------------------------------------------------------------


class FragmentPatchDataset(Dataset):
    """
    Samples random patches from multiple fragments with data augmentation.

    Each __getitem__ produces:
        input  — (patch_size, 6) augmented point+normal data
        target — (patch_size, 3) clean rotated point coordinates
    """

    def __init__(
        self,
        fragments: list[torch.Tensor],
        patch_size: int = 512,
        noise_std_rel: float = 0.005,    # σ / bbox-diag
        dropout_prob: float = 0.1,        # per-point dropout
        max_angle_deg: float = 30.0,
        train: bool = True,
    ):
        self.fragments = fragments
        self.patch_size = patch_size
        self.noise_std_rel = noise_std_rel
        self.dropout_prob = dropout_prob if train else 0.0
        self.max_angle_deg = max_angle_deg if train else 0.0
        self.train = train

        # Pre-compute bbox diagonals for noise scaling
        self._diags = []
        for frag in fragments:
            pts = frag[:, :3].numpy()
            diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            self._diags.append(diag)

        # Effective number of patches per fragment (for epoch balancing)
        self._frag_sizes = [max(1, f.shape[0] // patch_size) for f in fragments]
        self._n_patches = sum(self._frag_sizes)
    def __len__(self) -> int:
        return self._n_patches

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick a fragment (weighted by size)
        frag_idx = int(torch.randint(0, len(self.fragments), (1,)).item())
        frag = self.fragments[frag_idx]
        diag = self._diags[frag_idx]

        N = frag.shape[0]
        k = min(self.patch_size, N)
        indices = torch.randperm(N)[:k]

        patch = frag[indices]  # (k, 6)
        pos = patch[:, :3].clone()
        nrm = patch[:, 3:].clone()

        # ── Random SO(3) rotation ──
        if self.train and self.max_angle_deg > 0:
            R, _ = _rand_rot(self.max_angle_deg)
            R_t = torch.from_numpy(R.astype(np.float32))
            pos = pos @ R_t.T
            nrm = nrm @ R_t.T

        # ── Clean target = rotated (or original) positions ──
        target = pos.clone()

        # ── Gaussian noise ──
        if self.train and self.noise_std_rel > 0:
            noise_std = diag * self.noise_std_rel
            pos = pos + torch.randn_like(pos) * noise_std

        input_data = torch.cat([pos, nrm], dim=-1).float()
        return input_data, target.float()


def _rand_rot(max_angle_deg: float, seed: int | None = None) -> tuple[np.ndarray, float]:
    """Rodrigues-formula random rotation matrix with bounded angle."""
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
    return R, float(angle)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: GeoTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    model.set_mc_mode(False)
    total_loss = 0.0
    n_batches = 0

    for batch_input, batch_target in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        B = batch_input.shape[0]

        optimizer.zero_grad()
        # Process each patch independently — geometric attention is per-cloud
        loss = 0.0
        for i in range(B):
            output = model(batch_input[i])        # (N, 6) → (N, 3)
            loss += nn.functional.mse_loss(output, batch_target[i])
        loss = loss / B
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(
    model: GeoTransformer,
    dataloader: DataLoader,
    device: torch.device,
    T_mc: int = 10,
) -> tuple[float, float]:
    """
    Validation pass with MC Dropout.

    Returns (mse_loss, mean_epistemic_variance) averaged over T_mc passes.
    """
    model.eval()
    model.set_mc_mode(True)

    total_mse = 0.0
    total_var = 0.0
    n_batches = 0

    for batch_input, batch_target in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        B = batch_input.shape[0]

        batch_mse = 0.0
        batch_var = 0.0
        for i in range(B):
            x = batch_input[i]       # (N, 6)
            target = batch_target[i]  # (N, 3)

            # Collect T_mc stochastic forward passes
            outputs = torch.stack([
                model(x) for _ in range(T_mc)
            ])  # (T, N, 3)
            mean_pred = outputs.mean(dim=0)  # (N, 3)

            batch_mse += nn.functional.mse_loss(mean_pred, target).item()
            var_k = ((outputs - mean_pred.unsqueeze(0)) ** 2).sum(dim=-1)  # (T, N)
            batch_var += var_k.mean().item()

        total_mse += batch_mse / B
        total_var += batch_var / B
        n_batches += 1

    model.set_mc_mode(False)
    if n_batches == 0:
        return 0.0, 0.0
    return total_mse / n_batches, total_var / n_batches


def save_checkpoint(
    model: GeoTransformer,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    loss: float,
    path: str,
    *,
    centroids: list[np.ndarray] | None = None,
    scales: list[float] | None = None,
) -> None:
    checkpoint: dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if centroids is not None:
        checkpoint["centroids"] = [c.tolist() for c in centroids]
    if scales is not None:
        checkpoint["scales"] = scales
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-supervised GeoTransformer training on RePAIR fragments",
    )
    p.add_argument("fragments", nargs="+", help="Point cloud files (PLY/OBJ)")
    p.add_argument("--patch-size", type=int, default=512, help="Points per patch")
    p.add_argument("--batch-size", type=int, default=8, help="Patches per batch")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW decay")
    p.add_argument("--dropout-rate", type=float, default=0.2, help="MC dropout prob")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--noise-std-rel", type=float, default=0.005, help="Noise σ / bbox diag")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint .pt file (loads model + optimizer + epoch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load fragments ──
    print("=== Loading fragments ===")
    loaded = [load_fragment(p) for p in args.fragments]
    fragments = [t for t, _, _ in loaded]
    centroids = [c for _, c, _ in loaded if c is not None]
    scales = [s for _, _, s in loaded if s is not None]
    n_points = sum(f.shape[0] for f in fragments)
    print(f"  Total points: {n_points:,} across {len(fragments)} fragments\n")

    # ── 2. Train / val split ──
    # Use all fragments for training if val_split ≤ 0 or only 1 fragment
    n_val = int(len(fragments) * args.val_split)
    if n_val < 1 or len(fragments) <= 1:
        n_val = 0
    train_frags = fragments[:-n_val] if n_val > 0 else fragments
    val_frags = fragments[-n_val:] if n_val > 0 else []

    train_ds = FragmentPatchDataset(
        train_frags,
        patch_size=args.patch_size,
        noise_std_rel=args.noise_std_rel,
        max_angle_deg=30.0,
        train=True,
    )
    val_ds = FragmentPatchDataset(
        val_frags,
        patch_size=args.patch_size,
        max_angle_deg=0.0,
        noise_std_rel=0.0,
        dropout_prob=0.0,
        train=False,
    )

    train_ds = FragmentPatchDataset(
        train_frags,
        patch_size=args.patch_size,
        noise_std_rel=args.noise_std_rel,
        max_angle_deg=30.0,
        train=True,
    )
    val_ds = FragmentPatchDataset(
        val_frags,
        patch_size=args.patch_size,
        max_angle_deg=0.0,
        noise_std_rel=0.0,
        dropout_prob=0.0,
        train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── 3. Model ──
    device = torch.device(args.device)
    model = GeoTransformer(
        in_channels=6,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        bottleneck_dropout=args.dropout_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {total_params:,} params ({trainable_params:,} trainable)")
    print(f"  Device: {device}")

    # ── 4. Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        print(f"  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # ── 5. Output dir ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "geotransformer_best.pt"
    latest_path = out_dir / "geotransformer_latest.pt"

    print()
    print(f"=== Training (epochs {start_epoch}–{args.epochs}) ===")
    print(f"  Patches/epoch: {len(train_ds)}, Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, WD: {args.weight_decay}, Dropout p: {args.dropout_rate}")
    print()

    t0 = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        if len(val_ds) > 0:
            val_loss, val_var = validate_epoch(model, val_loader, device)
        else:
            val_loss, val_var = train_loss, 0.0
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, str(best_path),
                            centroids=centroids, scales=scales)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.perf_counter() - t0
            marker = " *" if is_best else ""
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"train mse: {train_loss:.6f} | "
                f"val mse: {val_loss:.6f} | "
                f"val σ²: {val_var:.6f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s{marker}"
            )

    # Save final checkpoint
    save_checkpoint(model, optimizer, args.epochs, val_loss, str(latest_path),
                    centroids=centroids, scales=scales)

    elapsed = time.perf_counter() - t0
    print(f"\n=== Done in {elapsed:.0f}s ({elapsed/60:.1f}m) ===")
    print(f"  Best val loss: {best_val_loss:.6f} → {best_path}")
    print(f"  Latest:        {val_loss:.6f} → {latest_path}")


if __name__ == "__main__":
    main()
