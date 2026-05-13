"""
Monte Carlo Dropout Inference with Welford's Online Accumulator.

Executes T stochastic forward passes through a GeoTransformer model with
dropout enabled at inference time, accumulating per-point mean and variance
incrementally — avoiding the O(T·N) memory cost of storing all outputs.

=== Welford's Online Algorithm ===

For a stream of observations {x₁, x₂, …, x_T}, the running mean and
variance are updated as:

    count ← count + 1
    δ      ← x_t − mean
    mean   ← mean + δ / count
    δ₂     ← x_t − mean  (new mean)
    M2     ← M2 + δ · δ₂

After T observations:
    variance = M2 / (T − 1)   (Bessel-corrected sample variance)

This is numerically stable and requires only O(N) storage regardless of T.

=== Epistemic Variance for 3D Point Clouds ===

Each forward pass produces a predicted point cloud ŷ_t ∈ ℝ^{N×3}.
The epistemic variance per point is the scalar:

    σ²_i = (1 / (T−1)) · Σ_{t=1}^{T} ‖ŷ_{t,i} − μ_i‖²

where μ_i = (1/T) Σ_t ŷ_{t,i} is the predictive mean position.

High σ²_i indicates the model is uncertain about the geometry at point i —
typically due to occlusion, erosion, or lack of similar training examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def run_mc_passes(
    model: nn.Module,
    point_cloud: torch.Tensor,
    T: int = 50,
    batch_size: int = 4096,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run T stochastic forward passes with MC Dropout enabled.

    Uses Welford's online algorithm to accumulate per-point mean and
    variance without storing all T outputs in memory.

    Args:
        model:       GeoTransformer with MC Dropout bottleneck.
                     Must have model.set_mc_mode(True) called before.
        point_cloud: Input point cloud, shape (N, 6) — [x,y,z, nx,ny,nz].
        T:           Number of stochastic forward passes (default 50).
        batch_size:  Points processed per forward pass to avoid OOM.
        device:      'cpu' or 'cuda'.
        verbose:     Print progress updates.

    Returns:
        mean:     Predictive mean point cloud, shape (N, 3).
        variance: Per-point epistemic variance (scalar), shape (N,).
    """
    # Ensure model is in MC mode
    model.set_mc_mode(True)
    model.eval()  # BatchNorm/LayerNorm in eval mode, but dropout stays on

    N = point_cloud.shape[0]
    point_cloud = point_cloud.to(device)
    model = model.to(device)

    # Welford's accumulators
    count = 0
    mean = torch.zeros(N, 3, device=device, dtype=torch.float64)
    M2 = torch.zeros(N, device=device, dtype=torch.float64)

    with torch.no_grad():
        for t in range(1, T + 1):
            if verbose and (t % 10 == 0 or t == 1 or t == T):
                print(f"  Pass {t}/{T}...", end="", flush=True)

            # Process in batches
            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch = point_cloud[batch_start:batch_end]

                # Stochastic forward pass (dropout active)
                output = model(batch)  # (B, 3)

                # Welford's online update
                count += 1
                delta = output.double() - mean[batch_start:batch_end]
                mean[batch_start:batch_end] += delta / count
                delta2 = output.double() - mean[batch_start:batch_end]
                M2[batch_start:batch_end] += (delta * delta2).sum(dim=1)

            if verbose and (t % 10 == 0 or t == T):
                print(" ✓")

    # Bessel-corrected variance: M2 / (T - 1)
    variance = M2 / (T - 1)

    # Reset MC mode
    model.set_mc_mode(False)

    return mean.float(), variance.float()


def run_mc_passes_batched(
    model: nn.Module,
    point_cloud: torch.Tensor,
    T: int = 50,
    batch_size: int = 4096,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Alternative: store all T outputs and compute variance at the end.

    This is simpler but uses O(T·N) memory.  Useful for debugging or
    when T is small (<20) and N is moderate (<10k points).

    Args:
        Same as run_mc_passes().

    Returns:
        mean:     (N, 3) predictive mean.
        variance: (N,) per-point epistemic variance.
    """
    model.set_mc_mode(True)
    model.eval()

    N = point_cloud.shape[0]
    point_cloud = point_cloud.to(device)
    model = model.to(device)

    all_outputs = torch.zeros(T, N, 3, device=device, dtype=torch.float32)

    with torch.no_grad():
        for t in range(T):
            if verbose and (t % 10 == 0 or t == T - 1):
                print(f"  Pass {t+1}/{T}...", end="", flush=True)

            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch = point_cloud[batch_start:batch_end]
                output = model(batch)
                all_outputs[t, batch_start:batch_end] = output

            if verbose and (t % 10 == 0 or t == T - 1):
                print(" ✓")

    mean = all_outputs.mean(dim=0)  # (N, 3)
    variance = ((all_outputs - mean.unsqueeze(0)) ** 2).sum(dim=-1).mean(dim=0)  # (N,)

    model.set_mc_mode(False)

    return mean, variance
