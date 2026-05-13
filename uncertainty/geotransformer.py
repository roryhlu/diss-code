"""
GeoTransformer with Monte Carlo Dropout Bottleneck.

Geometric transformer for point cloud feature extraction on textureless
archaeological fragments.  Replaces FPFH descriptors with learned features
that incorporate both local geometry (normals, curvature) and global
context (attention over the full fragment surface).

Architecture:
  Input: (N, 6) — points (x,y,z) + normals (nx,ny,nz)
    ↓
  VoxelFeatureEncoder — sparse voxelisation + point-wise MLP
    ↓
  GeometricTransformer — multi-head attention with distance-based geometric bias
    ↓
  MCDropoutBottleneck — Dropout(p) enabled at INFERENCE for epistemic uncertainty
    ↓
  FeatureDecoder — MLP mapping features → reconstructed 3D coordinates
    ↓
  Output: (N, 3) — predicted/completed point positions

The MC Dropout bottleneck is the critical layer: by keeping dropout active
during inference, each forward pass samples a different sub-network,
producing a distribution over outputs that approximates the Bayesian
posterior predictive.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Voxel Feature Encoder
# ---------------------------------------------------------------------------


class VoxelFeatureEncoder(nn.Module):
    """
    Voxel-based point cloud encoder.

    Converts raw (N, 6) point+normal data into per-point feature vectors
    via voxel grid hashing followed by a point-wise MLP.

    This is a simplified encoder that avoids sparse convolution dependencies
    (MinkowskiEngine) — suitable for fragments with <50k points after
    voxel downsampling.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        out_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input point cloud, shape (N, 6) — [x,y,z, nx,ny,nz].

        Returns:
            Per-point features, shape (N, out_dim).
        """
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Geometric Transformer
# ---------------------------------------------------------------------------


class GeometricAttention(nn.Module):
    """
    Multi-head attention with geometric distance bias.

    Standard scaled dot-product attention is augmented with a learned
    geometric prior that penalises attention between spatially distant
    points:

        score(i, j) = (Q_i · K_j) / sqrt(d_k)  +  γ · exp(−‖p_i − p_j‖² / 2σ²)

    where γ and σ are learnable parameters.  This bias ensures that
    attention is focused on geometrically proximate surface regions,
    which is critical for fragment reconstruction where long-range
    attention would mix unrelated fracture surfaces.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Geometric bias parameters
        self.gamma = nn.Parameter(torch.tensor(1.0))  # bias strength
        self.sigma = nn.Parameter(torch.tensor(0.1))  # bandwidth (metres)

        self.attn_dropout = nn.Dropout(dropout)

    def _geometric_bias(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance-based attention bias.

        Args:
            positions: (N, 3) point coordinates.

        Returns:
            Bias matrix (N, N) where bias[i,j] = γ·exp(−‖p_i−p_j‖²/2σ²).
        """
        # Pairwise squared distances: (N, N)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
        sq_dist = (diff ** 2).sum(dim=-1)  # (N, N)
        bias = self.gamma * torch.exp(-sq_dist / (2 * self.sigma ** 2))
        return bias

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, embed_dim) input features.
            positions: (N, 3) point coordinates for geometric bias.
            mask: Optional attention mask (N, N).

        Returns:
            Attended features, shape (N, embed_dim).
        """
        N = features.shape[0]

        # Linear projections
        Q = self.q_proj(features)  # (N, embed_dim)
        K = self.k_proj(features)
        V = self.v_proj(features)

        # Reshape for multi-head: (num_heads, N, head_dim)
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention: (num_heads, N, N)
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) * self.scale

        # Add geometric bias (broadcast across heads)
        geo_bias = self._geometric_bias(positions)  # (N, N)
        attn_scores = attn_scores + geo_bias.unsqueeze(0)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (num_heads, N, head_dim)
        output = torch.bmm(attn_weights, V)

        # Reshape back: (N, embed_dim)
        output = output.transpose(0, 1).contiguous().view(N, self.embed_dim)
        output = self.out_proj(output)

        return output


class GeometricTransformerBlock(nn.Module):
    """Single transformer block: attention + MLP with residual connections."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = GeometricAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, embed_dim).
            positions: (N, 3).

        Returns:
            Updated features, shape (N, embed_dim).
        """
        # Pre-norm residual attention
        features = features + self.attn(self.norm1(features), positions)
        # Pre-norm residual MLP
        features = features + self.mlp(self.norm2(features))
        return features


class GeometricTransformer(nn.Module):
    """Stack of geometric transformer blocks."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GeometricTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, embed_dim) from encoder.
            positions: (N, 3) original coordinates.

        Returns:
            Transformed features, shape (N, embed_dim).
        """
        for layer in self.layers:
            features = layer(features, positions)
        return self.norm(features)


# ---------------------------------------------------------------------------
# MC Dropout Bottleneck
# ---------------------------------------------------------------------------


class MCDropoutBottleneck(nn.Module):
    """
    Dropout layer that remains ACTIVE during inference for MC Dropout.

    Standard PyTorch nn.Dropout is disabled in eval mode (model.eval()).
    This wrapper overrides train/eval behaviour to keep dropout active
    regardless of the model's training flag — enabling stochastic sampling
    of the posterior predictive at inference time.

    Usage:
        model = GeoTransformer(...)
        model.eval()  # Normal inference — dropout OFF
        model.bottleneck.set_mc_mode(True)  # MC inference — dropout ON

    The dropout rate p controls the posterior approximation granularity:
      - p = 0.1: fine-grained uncertainty (subtle variations)
      - p = 0.2: moderate uncertainty (recommended for RePAIR fragments)
      - p = 0.3: coarse uncertainty (large structural variations)
    """

    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self._mc_mode = False

    def set_mc_mode(self, enabled: bool):
        """Toggle MC Dropout: True = active at inference, False = standard eval."""
        self._mc_mode = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mc_mode or self.training:
            return self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Feature Decoder
# ---------------------------------------------------------------------------


class FeatureDecoder(nn.Module):
    """
    MLP decoder mapping high-dimensional features to 3D coordinates.

    Reconstructs/completes the point cloud from the transformer's
    latent representation.  The output is a predicted position for
    each input point — effectively a denoised, completed surface.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        out_dim: int = 3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, in_dim) latent features.

        Returns:
            Predicted 3D coordinates, shape (N, 3).
        """
        return self.mlp(features)


# ---------------------------------------------------------------------------
# Full GeoTransformer Model
# ---------------------------------------------------------------------------


class GeoTransformer(nn.Module):
    """
    Complete GeoTransformer pipeline with MC Dropout bottleneck.

    Forward pass:
        Input (N,6) → VoxelEncoder → GeometricTransformer → MCDropoutBottleneck → FeatureDecoder → Output (N,3)

    For standard inference (deterministic):
        model.eval()
        model.bottleneck.set_mc_mode(False)
        output = model(input)

    For MC Dropout inference (stochastic):
        model.eval()
        model.bottleneck.set_mc_mode(True)
        for _ in range(T):
            output = model(input)  # different each time due to dropout
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        bottleneck_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = VoxelFeatureEncoder(in_channels, embed_dim // 2, embed_dim)
        self.transformer = GeometricTransformer(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout,
        )
        self.bottleneck = MCDropoutBottleneck(bottleneck_dropout)
        self.decoder = FeatureDecoder(embed_dim, embed_dim // 2, 3)

        # Positional encoding: sinusoidal embedding of 3D coordinates
        self.pos_encoding = SinusoidalPositionEncoding(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input point cloud, shape (N, 6) — [x,y,z, nx,ny,nz].

        Returns:
            Predicted 3D coordinates, shape (N, 3).
        """
        positions = x[:, :3]  # (N, 3)

        # Encode
        features = self.encoder(x)  # (N, embed_dim)

        # Add positional encoding
        features = features + self.pos_encoding(positions)

        # Transform with geometric attention
        features = self.transformer(features, positions)

        # MC Dropout bottleneck
        features = self.bottleneck(features)

        # Decode to 3D coordinates
        output = self.decoder(features)

        return output

    def set_mc_mode(self, enabled: bool):
        """Enable/disable MC Dropout for stochastic inference."""
        self.bottleneck.set_mc_mode(enabled)


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal positional encoding for 3D point coordinates.

    Encodes continuous 3D positions into a high-dimensional embedding
    using sinusoidal functions at multiple frequency bands, enabling
    the transformer to reason about absolute spatial relationships.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Applied independently to x, y, z coordinates and concatenated.
    """

    def __init__(self, d_model: int, max_freq: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

        # Frequency bands: (d_model // 6,) per axis
        d_per_axis = d_model // 3
        frequencies = torch.exp(
            torch.arange(0, d_per_axis, 2, dtype=torch.float32)
            * (-math.log(max_freq) / d_per_axis)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (N, 3) point coordinates.

        Returns:
            Positional encoding, shape (N, d_model).
        """
        # positions: (N, 3) → (N, 3, 1) * (1, 1, d/6) → (N, 3, d/6)
        angles = positions.unsqueeze(-1) * self.frequencies.unsqueeze(0).unsqueeze(0)
        # (N, 3, d/6) → sin, cos → (N, 3, d/3) → (N, d)
        encoding = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)
        return encoding.view(positions.shape[0], -1)
