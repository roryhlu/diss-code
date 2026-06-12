#!/usr/bin/env python3
"""
Perception-to-control bridge ROS2 node.

Subscribes to the camera point cloud topic, runs the file-based
RePAIR perception pipeline (registration + evaluation + CVaR), and
publishes the best grasp pose for downstream execution.

This bridges the gap between ROS2 sensor streaming and the offline
file-based perception modules without rewriting them as ROS2 nodes.

Pipeline (triggered by service call or timer)
----------------------------------------------
  1. Save latest point cloud from /camera/depth/points → PLY on disk.
  2. Run TEASER++ registration (scene → CAD model) via subprocess.
  3. Run MC Dropout epistemic variance cloud (optional).
  4. Run CVaR grasp validation.
  5. Publish best grasp pose on /perception/grasp_pose.

Published topics
----------------
  /perception/grasp_pose  (geometry_msgs/PoseStamped)
      Best CVaR-ranked grasp pose in camera frame.

Service
-------
  /perception/run  (std_srvs/Trigger)
      Triggers the full perception pipeline on the latest camera cloud.

Parameters
----------
  cad_model       : str — path to CAD PLY file (required)
  grasp_candidates : str — path to grasp candidates JSON
  checkpoint      : str — path to GeoTransformer checkpoint (for MC)
  output_dir      : str — directory for intermediate/output files
  mu              : float — friction coefficient (default 0.5)
  voxel_size      : float — registration voxel size (default 0.005)
  c_threshold     : float — TLS truncation threshold (default 0.005)
  run_mc          : bool — run MC Dropout variance estimation
  publish_cloud   : bool — publish aligned/registered cloud for viz
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
    from std_srvs.srv import Trigger

    rclpy.init(args=sys.argv)
    node = Node("perception_bridge")

    # ── Parameters ──
    node.declare_parameter("cad_model", "")
    node.declare_parameter("grasp_candidates", "scripts/sample_candidates.json")
    node.declare_parameter("checkpoint", "checkpoints/geotransformer_best.pt")
    node.declare_parameter("output_dir", "/tmp/repair_pipeline")
    node.declare_parameter("mu", 0.5)
    node.declare_parameter("voxel_size", 0.005)
    node.declare_parameter("c_threshold", 0.005)
    node.declare_parameter("run_mc", False)
    node.declare_parameter("publish_cloud", False)

    cad_model = node.get_parameter("cad_model").get_parameter_value().string_value
    grasp_candidates = node.get_parameter("grasp_candidates").get_parameter_value().string_value
    checkpoint = node.get_parameter("checkpoint").get_parameter_value().string_value
    output_dir = node.get_parameter("output_dir").get_parameter_value().string_value
    mu = node.get_parameter("mu").get_parameter_value().double_value
    voxel_size = node.get_parameter("voxel_size").get_parameter_value().double_value
    c_threshold = node.get_parameter("c_threshold").get_parameter_value().double_value
    run_mc = node.get_parameter("run_mc").get_parameter_value().bool_value
    publish_cloud = node.get_parameter("publish_cloud").get_parameter_value().bool_value

    if not cad_model:
        node.get_logger().error(
            "cad_model parameter is required. "
            "Set with --ros-args -p cad_model:=path/to/cad.ply"
        )
        rclpy.shutdown()
        return

    # ── State ──
    _latest_cloud: Optional[PointCloud2] = None
    _latest_points: Optional[np.ndarray] = None
    _project_root = str(Path(__file__).resolve().parent.parent)

    def _pc_callback(msg: PointCloud2) -> None:
        nonlocal _latest_cloud, _latest_points
        _latest_cloud = msg
        # Decode PointCloud2 → numpy
        dtype = np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ])
        _latest_points = np.frombuffer(msg.data, dtype=dtype)
        _latest_points = np.column_stack([
            _latest_points["x"], _latest_points["y"], _latest_points["z"],
        ])

    node.create_subscription(PointCloud2, "/camera/depth/points", _pc_callback, 10)

    # ── PLY publisher (aligned cloud visualisation) ──
    aligned_pub = None
    if publish_cloud:
        aligned_pub = node.create_publisher(PointCloud2, "/perception/aligned_cloud", 10)

    # ── Grasp pose publisher ──
    grasp_pub = node.create_publisher(PoseStamped, "/perception/grasp_pose", 10)

    # ── Pipeline trigger service ──
    def _run_pipeline(request, response) -> Trigger.Response:
        nonlocal _latest_points
        node.get_logger().info("=== Perception pipeline triggered ===")

        if _latest_points is None or len(_latest_points) == 0:
            response.success = False
            response.message = "No point cloud available — is camera_capture running?"
            return response

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scene_path = str(out / f"scene_{timestamp}.ply")

        # 1. Save scene PLY
        node.get_logger().info(f"Saving scene cloud → {scene_path}")
        _write_ply(scene_path, _latest_points)

        # 2. TEASER++ registration
        node.get_logger().info("Running TEASER++ registration ...")
        reg_script = str(Path(_project_root) / "scripts" / "teaser_register.py")
        reg_cmd = [
            sys.executable, reg_script,
            scene_path, cad_model,
            "--voxel-size", str(voxel_size),
            "--c-threshold", str(c_threshold),
            "--output", str(out / f"aligned_{timestamp}.ply"),
            "--no-viz",
        ]
        try:
            reg_proc = subprocess.run(
                reg_cmd, capture_output=True, text=True, timeout=600,
                cwd=_project_root,
            )
        except subprocess.TimeoutExpired:
            response.success = False
            response.message = "TEASER++ registration timed out"
            return response
        if reg_proc.returncode != 0:
            response.success = False
            response.message = f"Registration failed: {reg_proc.stderr[-200:]}"
            return response
        node.get_logger().info("Registration complete.")

        # 3. MC Dropout variance (optional)
        variance_path = None
        if run_mc and checkpoint:
            node.get_logger().info("Running MC Dropout variance ...")
            mc_script = str(Path(_project_root) / "scripts" / "mc_dropout_variance.py")
            variance_path = str(out / f"variance_{timestamp}.pcd")
            mc_cmd = [
                sys.executable, mc_script,
                scene_path,
                "--model", checkpoint,
                "--num-passes", "50",
                "--dropout-rate", "0.2",
                "--output", variance_path,
            ]
            try:
                mc_proc = subprocess.run(
                    mc_cmd, capture_output=True, text=True, timeout=600,
                    cwd=_project_root,
                )
                if mc_proc.returncode == 0:
                    node.get_logger().info("MC Dropout complete.")
                else:
                    node.get_logger().warn(f"MC Dropout failed: {mc_proc.stderr[-200:]}")
                    variance_path = None
            except subprocess.TimeoutExpired:
                node.get_logger().warn("MC Dropout timed out — skipping CVaR")
                variance_path = None

        # 4. CVaR grasp validation
        best_grasp: Optional[dict] = None
        if variance_path:
            node.get_logger().info("Running CVaR grasp validation ...")
            cvar_script = str(Path(_project_root) / "scripts" / "cvar_grasp_validator.py")
            cvar_out = str(out / f"accepted_grasps_{timestamp}.json")
            cvar_cmd = [
                sys.executable, cvar_script,
                variance_path,
                "--candidates", grasp_candidates,
                "--mu", str(mu),
                "--num-realizations", "100",
                "--cvar-alpha", "0.05",
                "--output", cvar_out,
            ]
            try:
                cvar_proc = subprocess.run(
                    cvar_cmd, capture_output=True, text=True, timeout=600,
                    cwd=_project_root,
                )
                if cvar_proc.returncode == 0 and os.path.exists(cvar_out):
                    with open(cvar_out) as f:
                        accepted = json.load(f)
                    if isinstance(accepted, list) and len(accepted) > 0:
                        best_grasp = accepted[0]
                        node.get_logger().info(
                            f"CVaR complete — {len(accepted)} grasps accepted, "
                            f"best CVaR ε = {best_grasp.get('cvar_epsilon', '?')}"
                        )
                    else:
                        node.get_logger().warn("No grasps passed CVaR filter.")
                else:
                    node.get_logger().warn(f"CVaR failed: {cvar_proc.stderr[-200:]}")
            except subprocess.TimeoutExpired:
                node.get_logger().warn("CVaR timed out")

        # 5. Publish best grasp pose
        if best_grasp:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = node.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_depth_optical_frame"
            # Estimate grasp midpoint from contact pair
            c1 = best_grasp.get("contact1", [0, 0, 0])
            c2 = best_grasp.get("contact2", [0, 0, 0])
            center = np.array(c1) * 0.5 + np.array(c2) * 0.5
            pose_msg.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            # Top-down orientation (quaternion for z-axis pointing down)
            pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            grasp_pub.publish(pose_msg)
            node.get_logger().info(
                f"Published grasp pose → /perception/grasp_pose "
                f"({pose_msg.pose.position.x:.3f}, {pose_msg.pose.position.y:.3f}, "
                f"{pose_msg.pose.position.z:.3f})"
            )

        response.success = True
        response.message = f"Pipeline complete. Output: {out}"
        return response

    node.create_service(Trigger, "/perception/run", _run_pipeline)

    node.get_logger().info(
        f"Perception bridge ready.\n"
        f"  CAD model: {cad_model}\n"
        f"  Output:    {output_dir}\n"
        f"  Trigger:   ros2 service call /perception/run std_srvs/srv/Trigger"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


def _write_ply(path: str, points: np.ndarray) -> None:
    """Write a binary PLY file with just x,y,z."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by perception_bridge\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype(np.float64).tobytes())


if __name__ == "__main__":
    main()
