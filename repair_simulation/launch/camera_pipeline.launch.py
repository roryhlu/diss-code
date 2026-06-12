"""
RePAIR perception pipeline — ROS2 launch file.

Starts the camera capture node and makes parameters configurable.
In a full deployment this would also launch perception, registration,
and grasp execution nodes in sequence.

Usage
-----
    # File-based test mode (no camera needed)
    ros2 launch repair_simulation camera_pipeline.launch.py \
        backend:=file file_source:=/path/to/scene.ply

    # RealSense mode (requires pyrealsense2)
    ros2 launch repair_simulation camera_pipeline.launch.py \
        backend:=realsense

    # OpenCV webcam mode
    ros2 launch repair_simulation camera_pipeline.launch.py \
        backend:=opencv
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    # ── Arguments ──
    ld.add_action(DeclareLaunchArgument(
        "backend", default_value="auto",
        description="Camera backend: auto, realsense, opencv, file"))
    ld.add_action(DeclareLaunchArgument(
        "file_source", default_value="",
        description="PLY/PCD path for file backend"))
    ld.add_action(DeclareLaunchArgument(
        "voxel_size", default_value="0.005",
        description="Voxel downsampling size in metres"))
    ld.add_action(DeclareLaunchArgument(
        "depth_scale", default_value="0.001",
        description="Depth units to metres conversion"))
    ld.add_action(DeclareLaunchArgument(
        "camera_frame", default_value="camera_depth_optical_frame",
        description="TF frame for the camera"))
    ld.add_action(DeclareLaunchArgument(
        "publish_tf", default_value="true",
        description="Publish static camera TF"))

    # ── Capture node ──
    ld.add_action(Node(
        package="repair_simulation",
        executable="camera_capture",
        name="camera_capture",
        output="screen",
        parameters=[{
            "backend": LaunchConfiguration("backend"),
            "file_source": LaunchConfiguration("file_source"),
            "voxel_size": LaunchConfiguration("voxel_size"),
            "depth_scale": LaunchConfiguration("depth_scale"),
            "camera_frame": LaunchConfiguration("camera_frame"),
            "publish_tf": LaunchConfiguration("publish_tf"),
        }],
        emulate_tty=True,
    ))

    return ld
