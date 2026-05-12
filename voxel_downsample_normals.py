"""
Voxel Grid Downsampling + PCA Normal Estimation via KD-Tree (k=30).

RePAIR Dissertation — Module 1: Deep Learning Perception Pipeline.
Estimates 6-DoF pose of irregular, textureless archaeological fragments.
"""

import argparse

import numpy as np
import open3d as o3d


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from file (PLY, PCD, XYZ, OBJ, STL)."""
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in '{file_path}'")
    return pcd


def downsample_voxel_grid(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """
    Voxel Grid Downsampling.

    Partitions 3D space into cubes of edge length `voxel_size`.
    All points inside a voxel are collapsed to their centroid:

        p_voxel = (1 / N_v) * sum_i p_i

    This enforces uniform point density while retaining global shape.
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def estimate_normals_pca_knn(
    pcd: o3d.geometry.PointCloud,
    k_neighbors: int = 30,
) -> o3d.geometry.PointCloud:
    """
    PCA-based surface normal estimation via KD-Tree k-NN search.

    For each point p_i the KD-Tree retrieves its k nearest neighbours.
    The local 3x3 covariance matrix is built:

        C = (1/k) * sum_{j=1}^{k} (p_j - mu)(p_j - mu)^T

    with mu = (1/k) * sum p_j.

    Eigendecomposition  C = V * Lambda * V^T  gives eigenvalues
    lambda_1 <= lambda_2 <= lambda_3.  The normal is v_1, the
    eigenvector of the smallest eigenvalue — the direction of minimal
    variance orthogonal to the local tangent plane.

    Normals are then consistently oriented.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors),
    )
    pcd.orient_normals_consistent_tangent_plane(k_neighbors)
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    """Save point cloud (points + normals) to file."""
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved to {output_path}")


def visualize(pcd: o3d.geometry.PointCloud) -> None:
    """Render the point cloud with normal vectors."""
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud + Estimated Normals",
        point_show_normal=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voxel downsampling + PCA normal estimation for RePAIR fragments",
    )
    parser.add_argument("input", type=str, help="Path to input point cloud")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel edge length (metres, default: 0.005)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="KNN count for PCA normal estimation (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the processed point cloud",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load
    print(f"Loading: {args.input}")
    pcd = load_point_cloud(args.input)
    print(f"  Original points: {len(pcd.points)}")

    # 2. Voxel Grid Downsampling
    print(f"Voxel downsampling (size = {args.voxel_size}) …")
    pcd = downsample_voxel_grid(pcd, voxel_size=args.voxel_size)
    print(f"  Downsampled points: {len(pcd.points)}")

    # 3. PCA Normal Estimation via KD-Tree (k = 30)
    print(f"PCA normal estimation (k = {args.k}) …")
    pcd = estimate_normals_pca_knn(pcd, k_neighbors=args.k)
    print("  Normals estimated.")

    # 4. Save
    if args.output:
        save_point_cloud(pcd, args.output)

    # 5. Visualize
    if not args.no_viz:
        visualize(pcd)


if __name__ == "__main__":
    main()
