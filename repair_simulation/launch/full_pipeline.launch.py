"""
RePAIR full perception-to-control pipeline — ROS2 launch file.

Launches the complete chain: camera → perception → grasp in one command.

Startup sequence
----------------
  1. camera_capture    — acquires depth point clouds
  2. perception_bridge — subscribes to camera, runs full pipeline on trigger
  3. grasp_executor    — reads accepted grasps and executes top-ranked pose

Usage
-----
    # Full pipeline with file-based camera
    ros2 launch repair_simulation full_pipeline.launch.py \
        cad_model:=RPf_00577_ds.ply \
        camera_backend:=file camera_file_source:=RPf_00577_ds.ply

    # Full pipeline with RealSense
    ros2 launch repair_simulation full_pipeline.launch.py \
        cad_model:=RPf_00577_ds.ply \
        camera_backend:=realsense

    # Perception + grasp only (camera already running separately)
    ros2 launch repair_simulation full_pipeline.launch.py \
        cad_model:=RPf_00577_ds.ply \
        launch_camera:=false

Workflow after launch
---------------------
    1. ros2 service call /perception/run std_srvs/srv/Trigger
       → runs registration + CVaR + publishes grasp pose
    2. grasp_executor receives grasp pose and executes trajectory
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    # ── Top-level arguments ──
    ld.add_action(DeclareLaunchArgument(
        "cad_model", default_value="",
        description="Path to CAD PLY file (required for registration)"))
    ld.add_action(DeclareLaunchArgument(
        "grasp_candidates", default_value="scripts/sample_candidates.json",
        description="Path to grasp candidates JSON"))
    ld.add_action(DeclareLaunchArgument(
        "checkpoint", default_value="checkpoints/geotransformer_best.pt",
        description="Path to GeoTransformer checkpoint for MC Dropout"))
    ld.add_action(DeclareLaunchArgument(
        "output_dir", default_value="/tmp/repair_pipeline",
        description="Directory for pipeline intermediate/output files"))

    # ── Camera arguments ──
    ld.add_action(DeclareLaunchArgument(
        "launch_camera", default_value="true",
        description="Launch the camera capture node"))
    ld.add_action(DeclareLaunchArgument(
        "camera_backend", default_value="auto",
        description="Camera backend: auto, realsense, opencv, file"))
    ld.add_action(DeclareLaunchArgument(
        "camera_file_source", default_value="",
        description="PLY/PCD path for file backend"))
    ld.add_action(DeclareLaunchArgument(
        "camera_voxel_size", default_value="0.005",
        description="Camera output voxel size (metres)"))

    # ── Perception arguments ──
    ld.add_action(DeclareLaunchArgument(
        "run_mc", default_value="false",
        description="Run MC Dropout variance estimation"))
    ld.add_action(DeclareLaunchArgument(
        "registration_voxel_size", default_value="0.005",
        description="TEASER++ voxel size (metres)"))
    ld.add_action(DeclareLaunchArgument(
        "c_threshold", default_value="0.005",
        description="TLS truncation threshold (metres)"))

    # ── Grasp arguments ──
    ld.add_action(DeclareLaunchArgument(
        "gripper_width", default_value="0.035",
        description="Gripper opening at grasp (metres)"))
    ld.add_action(DeclareLaunchArgument(
        "velocity_scale", default_value="0.5",
        description="MoveIt2 velocity scaling factor"))
    ld.add_action(DeclareLaunchArgument(
        "approach_distance", default_value="0.050",
        description="Pre-grasp offset above target (metres)"))

    # ── 1. Camera capture node ──
    ld.add_action(Node(
        condition=IfCondition(LaunchConfiguration("launch_camera")),
        package="repair_simulation",
        executable="camera_capture",
        name="camera_capture",
        output="screen",
        parameters=[{
            "backend": LaunchConfiguration("camera_backend"),
            "file_source": LaunchConfiguration("camera_file_source"),
            "voxel_size": LaunchConfiguration("camera_voxel_size"),
        }],
        emulate_tty=True,
    ))

    # ── 2. Perception bridge node (starts after camera) ──
    ld.add_action(TimerAction(
        period=3.0,  # wait for camera to initialise
        actions=[
            Node(
                package="repair_simulation",
                executable="perception_bridge",
                name="perception_bridge",
                output="screen",
                parameters=[{
                    "cad_model": LaunchConfiguration("cad_model"),
                    "grasp_candidates": LaunchConfiguration("grasp_candidates"),
                    "checkpoint": LaunchConfiguration("checkpoint"),
                    "output_dir": LaunchConfiguration("output_dir"),
                    "voxel_size": LaunchConfiguration("registration_voxel_size"),
                    "c_threshold": LaunchConfiguration("c_threshold"),
                    "run_mc": LaunchConfiguration("run_mc"),
                }],
                emulate_tty=True,
            ),
        ],
    ))

    # ── 3. Grasp executor node (starts after perception bridge) ──
    # Uses file mode — reads the JSON produced by perception_bridge.
    # For live mode, would subscribe to /perception/grasp_pose instead.
    ld.add_action(TimerAction(
        period=6.0,  # wait for perception bridge to initialise
        actions=[
            Node(
                package="repair_simulation",
                executable="grasp_executor",
                name="grasp_executor",
                output="screen",
                parameters=[{
                    "gripper_width": LaunchConfiguration("gripper_width"),
                    "velocity_scale": LaunchConfiguration("velocity_scale"),
                    "approach_distance": LaunchConfiguration("approach_distance"),
                }],
                emulate_tty=True,
            ),
        ],
    ))

    return ld
