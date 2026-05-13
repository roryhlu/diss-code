#!/usr/bin/env python3
"""
ROS2 MoveIt2 Grasp Executor for RePAIR Simulation.

Takes a top-ranked CVaR grasp from the perception pipeline, transforms
it into the robot base frame via the hand-eye calibration matrix, and
commands the Wlkata Mirobot arm through a top-down approach-and-grasp
trajectory.

=== Top-Down Approach Strategy ===

A top-down approach moves the gripper vertically from above, minimising
lateral motion near the fragment surface.  This is safer than direct
linear approach because:

  1. The approach direction (-Z in tool frame) is orthogonal to the
     table plane, reducing the risk of lateral collisions.
  2. The pre-grasp pose provides a safe clearance height while the
     final descent is a short, controlled Cartesian motion.
  3. Any IK failures are detected at the pre-grasp stage (above the
     fragment) rather than after the arm enters the cluttered scene.

Trajectory stages:
  Stage 1  — Move to pre-grasp pose (joint-space plan)
  Stage 2  — Cartesian descent to grasp pose (straight-line -Z)
  Stage 3  — Close gripper (grasp)
  Stage 4  — Cartesian lift to pre-grasp (retreat)

=== Hand-Eye Transform Pipeline ===

  T_camera_target  ∈ SE(3)   (from TEASER++ registration: object in camera frame)
      ↓   X = AX⁻¹Z⁻¹B (hand-eye calibration, eye-to-hand mode)
  T_robot_target   ∈ SE(3)   (target grasp pose in robot base frame)

=== Usage ===

    # From accepted_grasps.json (CVaR output)
    ros2 run repair_simulation grasp_executor \
        --ros-args -p grasp_file:=accepted_grasps.json

    # Direct pose specification
    ros2 run repair_simulation grasp_executor \
        --ros-args -p target_x:=0.35 -p target_y:=0.0 -p target_z:=0.15 \
        -p roll:=0.0 -p pitch:=3.14 -p yaw:=0.0
"""

import math
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_py import MoveItPy
from moveit_py.core import RobotState
from moveit_py.planning import (
    PlanningComponent,
    PlanRequestParameters,
    MotionPlanResponse,
)
from tf2_geometry_msgs import do_transform_pose

from repair_simulation.hand_eye import HandEyeCalibration


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GraspConfig:
    """Default configuration for the Wlkata Mirobot RePAIR setup."""

    # MoveIt2 planning group (from SRDF)
    PLANNING_GROUP = "mirobot_arm"

    # Gripper joint names (from SRDF)
    GRIPPER_JOINTS = ["gripper_left_joint", "gripper_right_joint"]

    # End-effector link (from SRDF)
    EE_LINK = "tool0"

    # Base frame (from SRDF virtual joint)
    BASE_FRAME = "world"

    # Top-down approach parameters
    APPROACH_DISTANCE = 0.050   # metres above grasp pose
    RETREAT_DISTANCE = 0.050    # metres above grasp pose after release
    CARTESIAN_STEP = 0.005      # metre resolution for Cartesian path
    CARTESIAN_JUMP_THRESH = 0.0 # no large joint jumps

    # Gripper configuration
    GRIPPER_OPEN = [0.0, 0.0]        # fully open
    GRIPPER_CLOSE = [0.01, 0.01]     # close on fragment

    # Hand-eye calibration (placeholder — calibrated per setup)
    # Camera is D405 mounted ~0.5m in front of robot base, pointing at table
    HAND_EYE_MATRIX = np.array([
        [ 0.0,  1.0,  0.0,  0.500],  # camera X = robot Y + offset
        [ 0.0,  0.0,  1.0,  0.350],  # camera Y = robot Z (table height)
        [ 1.0,  0.0,  0.0, -0.100],  # camera Z = robot X (forward)
        [ 0.0,  0.0,  0.0,  1.000],
    ], dtype=np.float64)

    # Velocity/acceleration scaling
    VELOCITY_SCALE = 0.5     # 50% max velocity for safety
    ACCELERATION_SCALE = 0.3  # 30% max acceleration


# ---------------------------------------------------------------------------
# Pose utilities
# ---------------------------------------------------------------------------

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Euler angles (radians, ZYX intrinsic) → ROS2 Quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def se3_to_pose(T: np.ndarray) -> Pose:
    """Convert 4×4 SE(3) matrix to ROS2 Pose message."""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])

    # Rotation matrix → quaternion
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        pose.orientation.w = 0.25 / s
        pose.orientation.x = (R[2, 1] - R[1, 2]) * s
        pose.orientation.y = (R[0, 2] - R[2, 0]) * s
        pose.orientation.z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        pose.orientation.w = (R[2, 1] - R[1, 2]) / s
        pose.orientation.x = 0.25 * s
        pose.orientation.y = (R[0, 1] + R[1, 0]) / s
        pose.orientation.z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        pose.orientation.w = (R[0, 2] - R[2, 0]) / s
        pose.orientation.x = (R[0, 1] + R[1, 0]) / s
        pose.orientation.y = 0.25 * s
        pose.orientation.z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        pose.orientation.w = (R[1, 0] - R[0, 1]) / s
        pose.orientation.x = (R[0, 2] + R[2, 0]) / s
        pose.orientation.y = (R[1, 2] + R[2, 1]) / s
        pose.orientation.z = 0.25 * s
    return pose


def offset_pose_z(pose: Pose, dz: float) -> Pose:
    """Return a copy of the pose offset by dz metres in world Z."""
    result = Pose()
    result.position.x = pose.position.x
    result.position.y = pose.position.y
    result.position.z = pose.position.z + dz
    result.orientation = pose.orientation
    return result


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------


class GraspExecutor(Node):
    """
    ROS2 node that executes a top-down grasp trajectory.

    Subscribes to: (none — triggered at startup or via service)
    Publishes to:   MoveIt2 action servers (internal)
    """

    def __init__(self):
        super().__init__("grasp_executor")

        # Parameters
        self.declare_parameter("grasp_file", "")
        self.declare_parameter("target_x", 0.350)
        self.declare_parameter("target_y", 0.000)
        self.declare_parameter("target_z", 0.120)
        self.declare_parameter("roll", 0.0)
        self.declare_parameter("pitch", math.pi)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("gripper_width", 0.035)
        self.declare_parameter("approach_distance", 0.050)
        self.declare_parameter("velocity_scale", 0.5)

        # Hand-eye calibration
        self.hand_eye = HandEyeCalibration("eye_to_hand")
        self.hand_eye.set_calibration(GraspConfig.HAND_EYE_MATRIX)

        # Initialise MoveIt2
        self.get_logger().info("Initialising MoveItPy...")
        self._moveit = MoveItPy(node_name="repair_simulation")
        self._arm = PlanningComponent(
            GraspConfig.PLANNING_GROUP,
            self._moveit,
            "repair_simulation",
        )
        self._plan_params = PlanRequestParameters(
            self._moveit,
            "repair_simulation",
        )
        self._plan_params.velocity_scaling_factor = GraspConfig.VELOCITY_SCALE
        self._plan_params.acceleration_scaling_factor = GraspConfig.ACCELERATION_SCALE

        self._gripper_open = {j: v for j, v in zip(
            GraspConfig.GRIPPER_JOINTS, GraspConfig.GRIPPER_OPEN
        )}
        self._gripper_close = {j: v for j, v in zip(
            GraspConfig.GRIPPER_JOINTS, GraspConfig.GRIPPER_CLOSE
        )}

        self.get_logger().info("GraspExecutor initialised.")

    # ── Public API ──

    def execute_from_file(self, grasp_file: str, rank: int = 0) -> bool:
        """
        Load and execute a grasp from a CVaR output JSON file.

        Args:
            grasp_file: Path to accepted_grasps.json from CVaR validator.
            rank:       Which ranked grasp to execute (0 = best).

        Returns:
            True if grasp executed successfully.
        """
        with open(grasp_file) as f:
            data = json.load(f)

        accepted = data.get("accepted", [])
        if not accepted:
            self.get_logger().error("No accepted grasps in file!")
            return False
        if rank >= len(accepted):
            self.get_logger().error(
                f"Rank {rank} exceeds available grasps ({len(accepted)})"
            )
            return False

        grasp = accepted[rank]
        c1 = np.array(grasp["contact1"], dtype=np.float64)
        c2 = np.array(grasp["contact2"], dtype=np.float64)

        # Compute grasp centre and approach direction
        centre = (c1 + c2) / 2.0
        direction = c2 - c1
        direction /= np.linalg.norm(direction)

        # Build target pose: centre between contacts, gripper pointing down
        target_pose = self._build_grasp_pose(centre, direction)

        self.get_logger().info(
            f"Executing ranked grasp #{grasp['rank']} "
            f"(CVaR ε = {grasp['cvar_epsilon']:.6f})"
        )
        return self.execute_grasp(target_pose, grasp.get("gripper_width", 0.035))

    def execute_pose(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> bool:
        """Execute a grasp at a directly specified Cartesian pose."""
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        target_pose.orientation = euler_to_quaternion(roll, pitch, yaw)
        return self.execute_grasp(target_pose, GraspConfig.GRIPPER_CLOSE[0])

    def execute_grasp(self, target_pose: Pose, gripper_width: float) -> bool:
        """
        Execute the full top-down grasp sequence.

        Args:
            target_pose:  Target grasp pose in robot base frame.
            gripper_width: Gripper opening width at grasp (metres).

        Returns:
            True if all stages succeeded.
        """
        self.get_logger().info(
            f"Target: ({target_pose.position.x:.3f}, "
            f"{target_pose.position.y:.3f}, "
            f"{target_pose.position.z:.3f})"
        )

        # ── Stage 0: Open gripper ──
        self.get_logger().info("Stage 0: Opening gripper...")
        self._arm.plan_and_execute(
            self._gripper_open,
            collective_group_name=GraspConfig.PLANNING_GROUP,
        )

        # ── Stage 1: Move to pre-grasp pose (joint-space) ──
        pre_grasp = offset_pose_z(target_pose, GraspConfig.APPROACH_DISTANCE)
        self.get_logger().info(
            f"Stage 1: Planning joint-space to pre-grasp "
            f"({pre_grasp.position.x:.3f}, {pre_grasp.position.y:.3f}, "
            f"{pre_grasp.position.z:.3f})"
        )
        if not self._plan_and_move_to_pose(pre_grasp, "pre_grasp"):
            self.get_logger().error("Stage 1 failed: cannot reach pre-grasp pose")
            return False

        # ── Stage 2: Cartesian descent to grasp pose ──
        self.get_logger().info("Stage 2: Cartesian descent to grasp...")
        waypoints = [
            pre_grasp,
            offset_pose_z(target_pose, 0.010),  # approach point
            target_pose,                         # grasp
        ]
        if not self._execute_cartesian_path(waypoints, "grasp_descent"):
            self.get_logger().error("Stage 2 failed: Cartesian path invalid")
            self._retreat(pre_grasp)
            return False

        # ── Stage 3: Close gripper ──
        self.get_logger().info("Stage 3: Closing gripper...")
        close_target = {j: gripper_width for j in GraspConfig.GRIPPER_JOINTS}
        self._arm.plan_and_execute(
            close_target,
            collective_group_name=GraspConfig.PLANNING_GROUP,
        )

        # ── Stage 4: Cartesian lift to retreat ──
        self.get_logger().info("Stage 4: Lifting to retreat...")
        self._retreat(pre_grasp)

        self.get_logger().info("Grasp sequence complete ✓")
        return True

    # ── Internal methods ──

    def _build_grasp_pose(
        self, centre: np.ndarray, direction: np.ndarray
    ) -> Pose:
        """
        Build an SE(3) grasp pose from contact geometry.

        The gripper approaches from above (-Z in world frame).
        The X-axis of the tool frame is aligned with the inter-contact
        axis (contact1 → contact2).
        The Z-axis points downward (opposite to world Z).

        This pose is in camera frame and must be transformed to robot
        base frame via hand-eye calibration before execution.
        """
        # Tool Z-axis = downward (world -Z, but in camera frame this may differ)
        tool_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Tool X-axis = inter-contact direction, projected to horizontal
        direction_h = direction.copy()
        direction_h[2] = 0.0
        dir_norm = np.linalg.norm(direction_h)
        if dir_norm < 1e-6:
            direction_h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction_h /= dir_norm

        tool_x = direction_h

        # Tool Y-axis = Z × X
        tool_y = np.cross(tool_z, tool_x)
        tool_y /= np.linalg.norm(tool_y)

        # Ensure orthonormal: X = Y × Z
        tool_x = np.cross(tool_y, tool_z)

        R = np.column_stack([tool_x, tool_y, tool_z])

        T_camera = np.eye(4, dtype=np.float64)
        T_camera[:3, :3] = R
        T_camera[:3, 3] = centre

        # Transform to robot base frame
        T_robot = self.hand_eye.transform_pose(T_camera)

        return se3_to_pose(T_robot)

    def _plan_and_move_to_pose(self, pose: Pose, label: str) -> bool:
        """Plan a joint-space trajectory to a target pose and execute."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = GraspConfig.BASE_FRAME
        pose_stamped.pose = pose

        plan_result = self._arm.plan(
            pose_stamped,
            self._plan_params,
        )
        if not plan_result:
            self.get_logger().warn(f"Planning failed for {label}")
            return False

        self.get_logger().info(f"Executing plan for {label}...")
        exec_result = self._arm.execute(plan_result)
        return bool(exec_result)

    def _execute_cartesian_path(
        self, waypoints: list[Pose], label: str
    ) -> bool:
        """Plan and execute a Cartesian path through a list of waypoints."""
        self.get_logger().info(
            f"Computing Cartesian path ({len(waypoints)} waypoints)..."
        )

        cart_result = self._arm.compute_cartesian_path(
            waypoints,
            eef_step=GraspConfig.CARTESIAN_STEP,
            jump_threshold=GraspConfig.CARTESIAN_JUMP_THRESH,
        )

        if cart_result.fraction < 1.0:
            self.get_logger().warn(
                f"Cartesian path only {cart_result.fraction*100:.1f}% "
                f"feasible for {label}"
            )
            if cart_result.fraction < 0.95:
                return False

        self.get_logger().info(
            f"Cartesian path {cart_result.fraction*100:.0f}% feasible. "
            f"Executing..."
        )
        exec_result = self._arm.execute(cart_result)
        return bool(exec_result)

    def _retreat(self, pre_grasp: Pose) -> bool:
        """Move back to the pre-grasp pose (joint-space plan)."""
        return self._plan_and_move_to_pose(pre_grasp, "retreat")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutor()

    grasp_file = node.get_parameter("grasp_file").get_parameter_value().string_value
    target_x = node.get_parameter("target_x").get_parameter_value().double_value
    target_y = node.get_parameter("target_y").get_parameter_value().double_value
    target_z = node.get_parameter("target_z").get_parameter_value().double_value
    roll = node.get_parameter("roll").get_parameter_value().double_value
    pitch = node.get_parameter("pitch").get_parameter_value().double_value
    yaw = node.get_parameter("yaw").get_parameter_value().double_value

    success = False

    if grasp_file:
        node.get_logger().info(f"Loading grasp from {grasp_file}")
        success = node.execute_from_file(grasp_file, rank=0)
    else:
        node.get_logger().info("Using direct pose parameters")
        success = node.execute_pose(target_x, target_y, target_z, roll, pitch, yaw)

    if success:
        node.get_logger().info("Grasp execution succeeded ✓")
    else:
        node.get_logger().error("Grasp execution failed ✗")

    rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    main()
