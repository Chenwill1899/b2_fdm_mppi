from __future__ import annotations

import math
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
AUSIM = Path("/home/mexxiie/prj/ausim2")
GEOMAPPING = Path("/home/mexxiie/prj/Geomapping_ros2")
ROBOT_CONFIG = AUSIM / "ground_vehicle/cfg/robot/scout_v2_mppi_freejoint_config.yaml"
STATIC_OBSTACLE_SCENE = AUSIM / "assets/scout_v2/scene.static_random_obstacles_seed119.xml"


def test_ros2_launch_sequence_reaches_16m_then_origin_without_collision(tmp_path):
    if os.environ.get("RUN_MUJOCO_MPPI_ROS2_SEQUENCE_INTEGRATION") != "1":
        pytest.skip("set RUN_MUJOCO_MPPI_ROS2_SEQUENCE_INTEGRATION=1 to run the ROS2 launch sequence smoke")
    pytest.importorskip("rclpy")
    pytest.importorskip("geometry_msgs")
    pytest.importorskip("nav_msgs")

    assert (AUSIM / "em_run.sh").exists()
    assert ROBOT_CONFIG.exists()
    assert STATIC_OBSTACLE_SCENE.exists()

    ausim_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"cd {AUSIM} && "
        f"./em_run.sh --robot-config {ROBOT_CONFIG} --terrain-xml {STATIC_OBSTACLE_SCENE} --headless"
    )
    launch_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"source {GEOMAPPING}/install/setup.bash && "
        f"cd {GEOMAPPING} && "
        "ros2 launch traversability_mapping ausim_cube_mppi.launch.py "
        "launch_rviz:=false use_medirl:=true launch_mppi:=true mppi_controller:=nominal_numpy"
    )

    obstacles = _scene_obstacles(STATIC_OBSTACLE_SCENE)
    processes: list[subprocess.Popen] = []
    monitor = None
    try:
        processes.append(_popen(ausim_cmd, tmp_path / "ausim.log"))
        monitor = _SequenceMonitor(obstacles=obstacles)
        monitor.wait_for_odom(timeout=25.0)
        processes.append(_popen(launch_cmd, tmp_path / "ros2_launch.log"))
        monitor.wait_for_goal_subscribers(count=2, timeout=30.0)

        monitor.publish_goal(16.0, 0.0, duration=2.0)
        monitor.wait_until_x_at_least(14.0, timeout=90.0)
        monitor.publish_goal(0.0, 0.0, duration=2.0)
        monitor.wait_until_near(0.0, 0.0, distance=0.8, timeout=120.0)

        assert monitor.min_clearance > 0.0
    finally:
        for proc in reversed(processes):
            _terminate_process_tree(proc)
        if monitor is not None:
            monitor.close()


def _scene_obstacles(path: Path) -> list[tuple[float, float, float]]:
    root = ET.parse(path).getroot()
    obstacles: list[tuple[float, float, float]] = []
    for geom in root.findall(".//geom"):
        name = str(geom.get("name", ""))
        if not name.startswith("dynamic_obs_"):
            continue
        pos = [float(v) for v in str(geom.get("pos", "0 0 0")).split()]
        size = [float(v) for v in str(geom.get("size", "0")).split()]
        geom_type = str(geom.get("type", ""))
        if len(pos) < 2 or not size:
            continue
        if geom_type == "box" and len(size) >= 2:
            radius = math.hypot(size[0], size[1])
        else:
            radius = size[0]
        obstacles.append((pos[0], pos[1], radius))
    return obstacles


def _popen(command: str, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["bash", "-lc", command],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5.0)


class _SequenceMonitor:
    def __init__(self, *, obstacles: list[tuple[float, float, float]]) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry

        rclpy.init(args=None)
        self._rclpy = rclpy
        self._pose_type = PoseStamped
        self._node = rclpy.create_node("mujoco_ros2_launch_sequence_test")
        self._odom = None
        self._obstacles = obstacles
        self.min_clearance = float("inf")
        self._publisher = self._node.create_publisher(PoseStamped, "/move_base_simple/goal", 10)
        self._subscription = self._node.create_subscription(Odometry, "/scout1/odom", self._on_odom, 10)

    def _on_odom(self, message) -> None:
        self._odom = message
        if not self._obstacles:
            return
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        robot_radius = 0.55
        for ox, oy, obstacle_radius in self._obstacles:
            clearance = math.hypot(x - ox, y - oy) - robot_radius - obstacle_radius
            self.min_clearance = min(self.min_clearance, clearance)

    def spin_once(self, timeout: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=float(timeout))

    def wait_for_odom(self, *, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline and self._odom is None:
            self.spin_once(0.1)
        assert self._odom is not None, "no /scout1/odom received from MuJoCo"

    def wait_for_goal_subscribers(self, *, count: int, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self._publisher.get_subscription_count() >= int(count):
                return
            self.spin_once(0.1)
        observed = self._publisher.get_subscription_count()
        assert False, f"expected {count} subscribers on /move_base_simple/goal, got {observed}"

    def publish_goal(self, x: float, y: float, *, duration: float) -> None:
        deadline = time.monotonic() + float(duration)
        while time.monotonic() < deadline:
            goal = self._pose_type()
            goal.header.frame_id = "map"
            goal.header.stamp = self._node.get_clock().now().to_msg()
            goal.pose.position.x = float(x)
            goal.pose.position.y = float(y)
            goal.pose.orientation.w = 1.0
            self._publisher.publish(goal)
            self.spin_once(0.1)

    def wait_until_near(self, x: float, y: float, *, distance: float, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self.spin_once(0.1)
            if self._odom is None:
                continue
            pos = self._odom.pose.pose.position
            if math.hypot(float(pos.x) - x, float(pos.y) - y) <= float(distance):
                return
        assert self._odom is not None
        pos = self._odom.pose.pose.position
        raise AssertionError(f"goal ({x:.1f}, {y:.1f}) not reached, latest=({pos.x:.2f}, {pos.y:.2f})")

    def wait_until_x_at_least(self, x: float, *, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self.spin_once(0.1)
            if self._odom is None:
                continue
            pos = self._odom.pose.pose.position
            if float(pos.x) >= float(x):
                return
        assert self._odom is not None
        pos = self._odom.pose.pose.position
        raise AssertionError(f"x did not reach {x:.1f}, latest=({pos.x:.2f}, {pos.y:.2f})")

    def close(self) -> None:
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
