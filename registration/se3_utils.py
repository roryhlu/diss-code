"""
SE(3) Utility Functions.

Provides point transformation, composition, inversion, and extraction
utilities for rigid body transforms in the Special Euclidean group SE(3).

All functions support batched (B, ...) and unbatched tensor inputs.
"""

from __future__ import annotations

import torch


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) transform to points:  P' = R @ P + t.

    Args:
        T:      4x4 or (B,4,4) transformation matrix.
        points: (N,3) or (B,N,3) point cloud(s).

    Returns:
        Transformed points, same shape as input.
    """
    batched = T.dim() == 3
    if not batched:
        T = T.unsqueeze(0)
        points = points.unsqueeze(0)

    R = T[:, :3, :3]
    t = T[:, :3, 3]

    rotated = torch.bmm(points, R.transpose(-2, -1))  # (B, N, 3)
    transformed = rotated + t[:, None, :]

    return transformed if batched else transformed.squeeze(0)


def extract_rt(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split 4x4 or (B,4,4) SE(3) matrix into rotation (3,3) and translation (3,).
    """
    return T[..., :3, :3], T[..., :3, 3]


def compose(T_ab: torch.Tensor, T_bc: torch.Tensor) -> torch.Tensor:
    """
    Compose two SE(3) transforms: T_ac = T_ab @ T_bc.
    Supports batched inputs.
    """
    return T_ab @ T_bc


def inverse_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Compute SE(3) inverse fast.

    Given T = [R  t; 0^T  1], returns [R^T  -R^T t; 0^T  1].

    Uses the SE(3) inverse identity for computational efficiency
    without performing a full 4x4 matrix inverse.

    Supports batched (B, 4, 4) inputs.
    """
    batched = T.dim() == 3
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_inv = R.transpose(-2, -1)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    if batched:
        T_inv = T_inv.expand(T.shape[0], 4, 4).clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    return T_inv
