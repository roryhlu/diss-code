"""
Fast Point Feature Histograms (FPFH) for Texture-Robust Registration.

Computes local geometric descriptors that overcome photometric feature collapse
on textureless archaeological fragments from the RePAIR dataset.

FPFH formulation:
  SPFH(p):  33-bin histogram of angular triplets (alpha, phi, theta) between
            the normal of query point p and each neighbour p_k.
  FPFH(p):  SPFH(p) + (1/k) * sum_{p_k in N(p)} (1/omega_k) * SPFH(p_k)

where omega_k = ||p - p_k|| is the distance weight.

Feature matching uses mutual nearest-neighbour search with Lowe's ratio test
to reject ambiguous correspondences before TEASER++ refinement.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def compute_fpfh(
    pcd: o3d.geometry.PointCloud,
    *,
    normal_radius: float = 0.01,
    normal_k: int = 30,
    fpfh_radius: float = 0.025,
) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH descriptors for a point cloud.

    1. Estimate surface normals via PCA (k-NN search).
    2. Compute 33-dimensional FPFH vectors per point.

    The normal_radius should be ~2x the voxel downsampling size.
    The fpfh_radius should be ~5x to capture sufficient local context
    for distinguishing geometric features.

    Args:
        pcd:           Input point cloud.
        normal_radius: Search radius (metres) for normal estimation.
        normal_k:      Max KNN for normal PCA (30 recommended).
        fpfh_radius:   Search radius (metres) for FPFH computation.

    Returns:
        Open3D Feature object of shape (N, 33).
    """
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_k,
            )
        )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=100,
        ),
    )
    return fpfh


def match_features(
    fpfh_src: o3d.pipelines.registration.Feature,
    fpfh_tgt: o3d.pipelines.registration.Feature,
    *,
    mutual_filter: bool = True,
    ratio_threshold: float = 0.9,
    max_correspondences: int = 5000,
) -> o3d.utility.Vector2iVector:
    """
    Match FPFH descriptors between source and target point clouds.

    Performs nearest-neighbour matching in 33-dim FPFH space with
    Lowe's ratio test and optional mutual-consistency filter.

    Args:
        fpfh_src:           Source FPFH features (N_src, 33).
        fpfh_tgt:           Target FPFH features (N_tgt, 33).
        mutual_filter:      If True, keep only correspondences that are mutual
                            nearest neighbours (bidirectional match).
        ratio_threshold:    Lowe's ratio threshold (0 < r <= 1). Lower = stricter.
        max_correspondences: Upper bound on returned correspondences.

    Returns:
        Vector2iVector of (src_idx, tgt_idx) correspondence pairs.
    """
    matcher = _build_correspondence_set(
        fpfh_src,
        fpfh_tgt,
        mutual_filter=mutual_filter,
        ratio_threshold=ratio_threshold,
    )

    # limit to max_correspondences by taking best matches
    if len(matcher) > max_correspondences:
        distances = _match_distances(fpfh_src, fpfh_tgt, matcher)
        keep = np.argsort(distances)[:max_correspondences]
        matcher = np.asarray(matcher)[keep]

    return o3d.utility.Vector2iVector(matcher)


def extract_correspondence_clouds(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    correspondences: o3d.utility.Vector2iVector,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract matched point sets from correspondence indices.

    Args:
        pcd_src:         Source point cloud.
        pcd_tgt:         Target point cloud.
        correspondences:  Vector2iVector of (src_idx, tgt_idx) pairs.

    Returns:
        (src_pts, tgt_pts) — each shape (M, 3).
    """
    corrs = np.asarray(correspondences)
    src_pts = np.asarray(pcd_src.points)[corrs[:, 0]]
    tgt_pts = np.asarray(pcd_tgt.points)[corrs[:, 1]]
    return src_pts, tgt_pts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_correspondence_set(
    fpfh_src: o3d.pipelines.registration.Feature,
    fpfh_tgt: o3d.pipelines.registration.Feature,
    *,
    mutual_filter: bool,
    ratio_threshold: float,
) -> np.ndarray:
    """Core nearest-neighbour matching with ratio test and mutual filter."""
    src_data = np.asarray(fpfh_src.data).astype(np.float64)
    tgt_data = np.asarray(fpfh_tgt.data).astype(np.float64)

    # forward: src -> tgt
    fwd_indices, fwd_dists = _knn_search(src_data, tgt_data, k=2)
    # ratio test forward
    passed_fwd = fwd_dists[:, 0] < ratio_threshold * fwd_dists[:, 1]

    correspondences = []
    for i_src in np.where(passed_fwd)[0]:
        i_tgt = int(fwd_indices[i_src, 0])
        correspondences.append((i_src, i_tgt))

    if mutual_filter:
        # backward: tgt -> src
        bwd_indices, bwd_dists = _knn_search(tgt_data, src_data, k=2)
        passed_bwd = bwd_dists[:, 0] < ratio_threshold * bwd_dists[:, 1]

        tgt_to_src = {}
        for i_tgt in np.where(passed_bwd)[0]:
            tgt_to_src[i_tgt] = int(bwd_indices[i_tgt, 0])

        correspondences = [
            (s, t) for (s, t) in correspondences
            if tgt_to_src.get(t) == s
        ]

    if not correspondences:
        return np.empty((0, 2), dtype=np.int32)

    return np.array(correspondences, dtype=np.int32)


def _knn_search(
    query: np.ndarray,
    database: np.ndarray,
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Brute-force k-NN search in descriptor space.
    For small-medium feature sets (<50k) this is faster than KD-Tree overhead.
    """
    diff = query[:, None, :] - database[None, :, :]  # (Q, D, F)
    dists2 = np.sum(diff**2, axis=-1)                  # (Q, D)
    idx = np.argpartition(dists2, min(k, database.shape[0] - 1), axis=-1)[:, :k]
    k_dists = np.take_along_axis(dists2, idx, axis=-1)
    sort_order = np.argsort(k_dists, axis=-1)
    idx = np.take_along_axis(idx, sort_order, axis=-1)
    k_dists = np.take_along_axis(k_dists, sort_order, axis=-1)
    return idx, np.sqrt(k_dists)


def _match_distances(
    fpfh_src: o3d.pipelines.registration.Feature,
    fpfh_tgt: o3d.pipelines.registration.Feature,
    correspondences: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distance in FPFH space for each correspondence."""
    src_data = np.asarray(fpfh_src.data).astype(np.float64)
    tgt_data = np.asarray(fpfh_tgt.data).astype(np.float64)
    return np.linalg.norm(
        src_data[correspondences[:, 0]] - tgt_data[correspondences[:, 1]],
        axis=1,
    )
