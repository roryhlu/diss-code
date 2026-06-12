"""
RePAIR grasp executor — ROS2 launch file.

Launches the MoveIt2 grasp_executor node with configurable parameters
for either file-based grasp loading or direct pose specification.

Usage
-----
    # Load grasps from CVaR output file
    ros2 launch repair_simulation grasp_executor.launch.py \
        grasp_file:=accepted_grasps.json

    # Direct pose (bypasses perception pipeline)
    ros2 launch repair_simulation grasp_executor.launch.py \
        target_x:=0.35 target_y:=0.0 target_z:=0.12 pitch:=3.14

    # Adjust approach speed and gripper width
    ros2 launch repair_simulation grasp_executor.launch.py \
        grasp_file:=accepted_grasps.json velocity_scale:=0.3 gripper_width:=0.04
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    # ── Grasp source (mutually exclusive in practice) ──
    ld.add_action(DeclareLaunchArgument(
        "grasp_file", default_value="",
        description="Path to accepted_grasps.json from CVaR validator"))
    ld.add_action(DeclareLaunchArgument(
        "target_x", default_value="0.35",
        description="Target X position (robot base frame, metres)"))
    ld.add_action(DeclareLaunchArgument(
        "target_y", default_value="0.0",
        description="Target Y position (robot base frame, metres)"))
    ld.add_action(DeclareLaunchArgument(
        "target_z", default_value="0.12",
        description="Target Z position (robot base frame, metres)"))
    ld.add_action(DeclareLaunchArgument(
        "roll", default_value="0.0",
        description="Target roll angle (radians)"))
    ld.add_action(DeclareLaunchArgument(
        "pitch", default_value="3.14159",
        description="Target pitch angle (radians, π = top-down)"))
    ld.add_action(DeclareLaunchArgument(
        "yaw", default_value="0.0",
        description="Target yaw angle (radians)"))
    ld.add_action(DeclareLaunchArgument(
        "gripper_width", default_value="0.035",
        description="Gripper opening width at grasp (metres)"))
    ld.add_action(DeclareLaunchArgument(
        "approach_distance", default_value="0.050",
        description="Pre-grasp offset above target (metres)"))
    ld.add_action(DeclareLaunchArgument(
        "velocity_scale", default_value="0.5",
        description="MoveIt2 velocity scaling factor (0.0–1.0)"))

    # ── Node ──
    ld.add_action(Node(
        package="repair_simulation",
        executable="grasp_executor",
        name="grasp_executor",
        output="screen",
        parameters=[{
            "grasp_file": LaunchConfiguration("grasp_file"),
            "target_x": LaunchConfiguration("target_x"),
            "target_y": LaunchConfiguration("target_y"),
            "target_z": LaunchConfiguration("target_z"),
            "roll": LaunchConfiguration("roll"),
            "pitch": LaunchConfiguration("pitch"),
            "yaw": LaunchConfiguration("yaw"),
            "gripper_width": LaunchConfiguration("gripper_width"),
            "approach_distance": LaunchConfiguration("approach_distance"),
            "velocity_scale": LaunchConfiguration("velocity_scale"),
        }],
        emulate_tty=True,
    ))

    return ld
