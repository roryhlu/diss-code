#!/usr/bin/env python3
"""
Generic camera capture node for RePAIR perception pipeline.

Acquires depth point clouds from various backends and publishes them
on ROS2 topics for downstream registration and grasping.  Designed to
work with any depth-capable camera or as a file-based test harness.

Backends (auto-detected, in priority order)
-------------------------------------------
  realsense   — Intel RealSense D405/D415/D435 via pyrealsense2.
                Configures "High Accuracy" preset for sub-mm depth.
  opencv      — Generic USB/CSI camera via OpenCV VideoCapture.
                Requires a paired depth stream or structured light.
  file        — Reads PLY/PCD files from disk on a timer.
                Falls back to this when no hardware is detected.

Published topics
----------------
  /camera/depth/points        (sensor_msgs/PointCloud2)
      Downsampled point cloud (voxel grid at configurable size).

  /camera/depth/camera_info   (sensor_msgs/CameraInfo)
      Camera intrinsics (read from device or loaded from file).

  /tf                          (tf2_msgs/TFMessage)
      camera_depth_optical_frame → world transform.

Service
-------
  /camera/save_ply  (repair_interfaces/srv/SavePLY)
      Saves the current point cloud as a PLY file for the perception
      pipeline.  Returns the file path.

Parameters
----------
  voxel_size        : float  (default 0.005) — downsampling cube edge (m)
  backend           : str    (default "auto") — force backend selection
  file_source       : str    (default "") — PLY/PCD path for file backend
  file_interval     : float  (default 5.0) — seconds between reads in file mode
  depth_scale       : float  (default 0.001) — depth units to metres
  publish_tf        : bool   (default True) — publish camera_frame TF
  camera_frame      : str    (default "camera_depth_optical_frame")
  target_frame      : str    (default "world")
  calib_path        : str    (default "") — JSON file with camera intrinsics
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Backend imports (lazy, to avoid ImportError on systems without them) ────


def _has_librealsense() -> bool:
    try:
        import pyrealsense2 as rs  # noqa: F401
        return True
    except ImportError:
        return False


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


# ── Voxel downsampling (pure numpy) ───────────────────────────────────


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Uniform voxel-grid downsampling via vectorised accumulation."""
    if voxel_size <= 0 or len(points) < 2:
        return points
    min_pt = points.min(axis=0)
    voxel_idx = np.floor((points - min_pt) / voxel_size).astype(np.int64)
    span = voxel_idx.max(axis=0) - voxel_idx.min(axis=0) + 1
    ids = (
        voxel_idx[:, 0] * span[1] * span[2]
        + voxel_idx[:, 1] * span[2]
        + voxel_idx[:, 2]
    )
    _, inverse = np.unique(ids, return_inverse=True)
    n_voxels = inverse.max() + 1
    down = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(down, inverse, points)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return down / counts[:, None]


# ── Backend: RealSense ────────────────────────────────────────────────


class RealSenseBackend:
    """Intel RealSense depth camera via pyrealsense2."""

    def __init__(self, depth_scale: float = 0.001):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # Apply High Accuracy preset (best for archaeological fragments)
        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        try:
            depth_sensor.set_option(rs.option.visual_preset, 3.0)  # High Accuracy
        except Exception:
            pass  # some models don't support presets

        self._intrinsics = (
            profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        )
        # Skip initial frames while auto-exposure stabilises
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Returns (points_Nx3, rgb_Nx3_or_None)."""
        import pyrealsense2 as rs

        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            return None, None

        pc = rs.pointcloud()
        pc.map_to(depth)
        points = pc.calculate(depth)
        pts_np = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        # Filter zeros (invalid depth)
        valid = np.any(pts_np != 0, axis=1)
        return pts_np[valid], None

    @property
    def intrinsics(self) -> dict:
        i = self._intrinsics
        return {
            "width": i.width, "height": i.height,
            "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy,
            "model": "brown_conrady",
            "coeffs": i.coeffs,
        }

    def close(self) -> None:
        self.pipeline.stop()


# ── Backend: OpenCV ───────────────────────────────────────────────────


class OpenCVBackend:
    """Generic camera via OpenCV VideoCapture (webcam, CSI, etc.)."""

    def __init__(self, device_id: int = 0, depth_scale: float = 0.001):
        import cv2

        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._depth_scale = depth_scale

    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        import cv2

        ret, frame = self._cap.read()
        if not ret:
            return None, None
        # Assume a depth image (single-channel float or uint16)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.mgrid[0 : grey.shape[0], 0 : grey.shape[1]]
        z = grey.astype(np.float32) * self._depth_scale
        # Simple pinhole back-projection (fx=fy=600, cx=320, cy=240)
        fx, fy, cx, cy = 600.0, 600.0, grey.shape[1] / 2, grey.shape[0] / 2
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = z.ravel() > 0
        return points[valid], None

    @property
    def intrinsics(self) -> dict:
        w = int(self._cap.get(3))
        h = int(self._cap.get(4))
        return {"width": w, "height": h, "fx": 600.0, "fy": 600.0,
                "cx": w / 2, "cy": h / 2, "model": "plumb_bob", "coeffs": [0]*5}

    def close(self) -> None:
        self._cap.release()


# ── Backend: File ─────────────────────────────────────────────────────


class FileBackend:
    """Reads point clouds from disk files on a timer (test harness)."""

    def __init__(self, file_path: str, interval: float = 5.0):
        self._path = file_path
        self._interval = interval
        self._last_read = 0.0
        self._cached: np.ndarray | None = None

    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        now = time.time()
        if now - self._last_read < self._interval:
            return self._cached, None if self._cached is None else None

        self._last_read = now
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(self._path)
            if pcd.has_points():
                pts = np.asarray(pcd.points, dtype=np.float64)
                self._cached = pts
                return pts, None
        except Exception:
            pass
        return self._cached, None

    @property
    def intrinsics(self) -> dict:
        return {"width": 640, "height": 480, "fx": 600.0, "fy": 600.0,
                "cx": 320.0, "cy": 240.0, "model": "plumb_bob", "coeffs": [0]*5}

    def close(self) -> None:
        pass


# ── ROS2 Node ─────────────────────────────────────────────────────────


def _build_pointcloud2_msg(points: np.ndarray, frame_id: str, stamp):
    """Build a sensor_msgs/PointCloud2 from (N,3) numpy array."""
    from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * len(points)
    msg.is_bigendian = False
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


def _build_camera_info_msg(intrinsics: dict, frame_id: str, stamp):
    """Build a sensor_msgs/CameraInfo from intrinsics dict."""
    from sensor_msgs.msg import CameraInfo  # noqa: E402

    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = intrinsics.get("width", 640)
    msg.height = intrinsics.get("height", 480)
    msg.distortion_model = intrinsics.get("model", "plumb_bob")
    msg.d = list(intrinsics.get("coeffs", [0.0] * 5))
    msg.k = [
        intrinsics.get("fx", 600.0), 0.0, intrinsics.get("cx", 320.0),
        0.0, intrinsics.get("fy", 600.0), intrinsics.get("cy", 240.0),
        0.0, 0.0, 1.0,
    ]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = msg.k[:] + [0.0, 0.0, 0.0]
    return msg


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger

    rclpy.init(args=sys.argv)
    node = Node("camera_capture")

    # ── Parameters ──
    node.declare_parameter("voxel_size", 0.005)
    node.declare_parameter("backend", "auto")
    node.declare_parameter("file_source", "")
    node.declare_parameter("file_interval", 5.0)
    node.declare_parameter("depth_scale", 0.001)
    node.declare_parameter("publish_tf", True)
    node.declare_parameter("camera_frame", "camera_depth_optical_frame")
    node.declare_parameter("target_frame", "world")
    node.declare_parameter("calib_path", "")

    voxel_size = node.get_parameter("voxel_size").get_parameter_value().double_value
    backend_name = node.get_parameter("backend").get_parameter_value().string_value
    file_source = node.get_parameter("file_source").get_parameter_value().string_value
    file_interval = node.get_parameter("file_interval").get_parameter_value().double_value
    depth_scale = node.get_parameter("depth_scale").get_parameter_value().double_value
    publish_tf = node.get_parameter("publish_tf").get_parameter_value().bool_value
    camera_frame = node.get_parameter("camera_frame").get_parameter_value().string_value
    target_frame = node.get_parameter("target_frame").get_parameter_value().string_value
    calib_path = node.get_parameter("calib_path").get_parameter_value().string_value

    # ── Backend selection ──
    backend = None
    backend_kind = "none"

    if backend_name == "realsense" or (backend_name == "auto" and _has_librealsense()):
        try:
            backend = RealSenseBackend(depth_scale)
            backend_kind = "realsense"
            node.get_logger().info("Using RealSense backend")
        except Exception as e:
            node.get_logger().warn(f"RealSense init failed: {e}")

    if backend is None and (backend_name == "opencv" or (backend_name == "auto" and _has_cv2())):
        try:
            backend = OpenCVBackend(depth_scale=depth_scale)
            backend_kind = "opencv"
            node.get_logger().info("Using OpenCV backend")
        except Exception as e:
            node.get_logger().warn(f"OpenCV init failed: {e}")

    if backend is None and (backend_name in ("file", "auto") and file_source):
        try:
            backend = FileBackend(file_source, interval=file_interval)
            backend_kind = "file"
            node.get_logger().info(f"Using file backend: {file_source}")
        except Exception as e:
            node.get_logger().warn(f"File backend init failed: {e}")

    if backend is None:
        node.get_logger().error(
            "No camera backend available. "
            "Install pyrealsense2 or opencv-python, or set --ros-args -p file_source:=<path>"
        )
        rclpy.shutdown()
        return

    # ── Publishers ──
    from sensor_msgs.msg import PointCloud2, CameraInfo  # noqa: E402

    pc_pub = node.create_publisher(PointCloud2, "/camera/depth/points", 10)
    ci_pub = node.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)

    # TF broadcaster
    if publish_tf:
        from tf2_ros import StaticTransformBroadcaster  # noqa: E402
        from geometry_msgs.msg import TransformStamped  # noqa: E402

        tf_broadcaster = StaticTransformBroadcaster(node)
        static_tf = TransformStamped()
        static_tf.header.frame_id = target_frame
        static_tf.child_frame_id = camera_frame
        static_tf.transform.translation.x = 0.5   # camera in front of robot
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.35  # table height
        static_tf.transform.rotation.x = -0.5      # pointing down at table
        static_tf.transform.rotation.y = 0.5
        static_tf.transform.rotation.z = 0.5
        static_tf.transform.rotation.w = -0.5
        tf_broadcaster.sendTransform(static_tf)
        node.get_logger().info(f"Published static TF: {target_frame} → {camera_frame}")

    # ── PLY save service ──
    _last_cloud: np.ndarray | None = None

    if hasattr(Trigger, "__module__"):
        # std_srvs may not be available in all ROS2 distros; wrap safely
        try:
            def _save_callback(request, response):
                nonlocal _last_cloud
                if _last_cloud is None or len(_last_cloud) == 0:
                    response.success = False
                    response.message = "No point cloud available"
                    return response
                out_dir = Path(os.environ.get("HOME", "/tmp")) / "repair_captures"
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = out_dir / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.ply"
                with open(str(fname), "wb") as f:
                    header = (
                        f"ply\nformat binary_little_endian 1.0\n"
                        f"element vertex {len(_last_cloud)}\n"
                        f"property double x\nproperty double y\nproperty double z\nend_header\n"
                    )
                    f.write(header.encode("ascii"))
                    f.write(_last_cloud.astype(np.float64).tobytes())
                response.success = True
                response.message = str(fname)
                node.get_logger().info(f"Saved PLY → {fname}")
                return response
            node.create_service(Trigger, "/camera/save_ply", _save_callback)
        except Exception:
            node.get_logger().warn("std_srvs/Trigger not available — /camera/save_ply disabled")

    # ── Main loop ──
    node.get_logger().info(
        f"Camera capture running (backend={backend_kind}, voxel={voxel_size}m)"
    )

    try:
        while rclpy.ok():
            points, _ = backend.grab()
            if points is not None and len(points) > 0:
                # Downsample
                ds_points = voxel_downsample(points, voxel_size)
                _last_cloud = ds_points

                stamp = node.get_clock().now().to_msg()
                pc_msg = _build_pointcloud2_msg(ds_points, camera_frame, stamp)
                pc_pub.publish(pc_msg)

                ci_msg = _build_camera_info_msg(backend.intrinsics, camera_frame, stamp)
                ci_pub.publish(ci_msg)

            rclpy.spin_once(node, timeout_sec=0.05)

    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
