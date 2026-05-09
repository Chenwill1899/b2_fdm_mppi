from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
AUSIM = Path("/home/mexxiie/prj/ausim2")
GEOMAPPING = Path("/home/mexxiie/prj/Geomapping_ros2")
ROBOT_CONFIG = AUSIM / "ground_vehicle/cfg/robot/scout_v2_mppi_freejoint_config.yaml"
PROFILE = REPO / "configs/mujoco_behind_goal_smoke.yaml"


def test_mujoco_mppi_reaches_goal_five_meters_behind(tmp_path):
    _run_mujoco_goal_offset_smoke(
        tmp_path,
        goal_offset_x=-5.0,
        results_name="behind_goal_run",
        max_steps=320,
        timeout=70.0,
        final_distance_threshold=0.75,
        min_progress=4.0,
        progress_direction=-1.0,
    )


def test_mujoco_mppi_reaches_goal_eighteen_meters_ahead(tmp_path):
    if os.environ.get("RUN_MUJOCO_MPPI_FORWARD_INTEGRATION") != "1":
        pytest.skip("set RUN_MUJOCO_MPPI_FORWARD_INTEGRATION=1 to run the 18 m forward MuJoCo+MPPI smoke")
    _run_mujoco_goal_offset_smoke(
        tmp_path,
        goal_offset_x=18.0,
        results_name="forward_goal_run",
        max_steps=520,
        timeout=95.0,
        final_distance_threshold=0.85,
        min_progress=16.0,
        progress_direction=1.0,
    )


def _run_mujoco_goal_offset_smoke(
    tmp_path: Path,
    *,
    goal_offset_x: float,
    results_name: str,
    max_steps: int,
    timeout: float,
    final_distance_threshold: float,
    min_progress: float,
    progress_direction: float,
) -> None:
    if os.environ.get("RUN_MUJOCO_MPPI_INTEGRATION") != "1":
        pytest.skip("set RUN_MUJOCO_MPPI_INTEGRATION=1 to run the MuJoCo+ROS2 integration smoke")
    pytest.importorskip("rclpy")
    pytest.importorskip("geometry_msgs")
    pytest.importorskip("nav_msgs")

    assert (AUSIM / "em_run.sh").exists()
    assert ROBOT_CONFIG.exists()
    assert PROFILE.exists()

    results_dir = tmp_path / results_name
    ausim_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"cd {AUSIM} && "
        f"./em_run.sh --robot-config {ROBOT_CONFIG} --headless"
    )
    mppi_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"source {GEOMAPPING}/install/setup.bash && "
        f"cd {REPO} && "
        "/usr/bin/python3 tools/fdm_mppi.py mujoco-closed-loop "
        f"--profile {PROFILE} --controller nominal_numpy --results-dir {results_dir} --max-steps {int(max_steps)}"
    )

    processes: list[subprocess.Popen] = []
    try:
        processes.append(_popen(ausim_cmd, tmp_path / "ausim.log"))
        monitor = _GoalBehindMonitor(goal_offset_x=goal_offset_x)
        monitor.wait_for_odom(timeout=20.0)
        initial_x = monitor.latest_x()
        processes.append(_popen(mppi_cmd, tmp_path / "mppi.log"))
        monitor.publish_goal_after_subscriber(timeout=15.0)

        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline and processes[-1].poll() is None:
            monitor.spin_once(0.1)
        assert processes[-1].poll() is not None, "MPPI did not finish before the integration timeout"
        assert processes[-1].returncode == 0, (tmp_path / "mppi.log").read_text(encoding="utf-8", errors="replace")

        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["reached_goal"] is True
        assert summary["final_distance"] <= final_distance_threshold
        assert summary["goal"][0] == pytest.approx(initial_x + goal_offset_x, abs=0.25)

        trajectory = (results_dir / "trajectory.csv").read_text(encoding="utf-8").splitlines()
        x_index = trajectory[0].split(",").index("x")
        xs = [float(row.split(",")[x_index]) for row in trajectory[1:] if row]
        if progress_direction < 0.0:
            assert min(xs) < initial_x - min_progress * 0.5
            assert xs[-1] < initial_x - min_progress
        else:
            assert max(xs) > initial_x + min_progress * 0.5
            assert xs[-1] > initial_x + min_progress
    finally:
        for proc in reversed(processes):
            _terminate_process_tree(proc)
        if "monitor" in locals():
            monitor.close()


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


class _GoalBehindMonitor:
    def __init__(self, *, goal_offset_x: float) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry

        rclpy.init(args=None)
        self._rclpy = rclpy
        self._pose_type = PoseStamped
        self._node = rclpy.create_node("mujoco_behind_goal_test")
        self._goal_offset_x = float(goal_offset_x)
        self._odom = None
        self._fixed_goal_xy: tuple[float, float] | None = None
        self._publisher = self._node.create_publisher(PoseStamped, "/move_base_simple/goal", 10)
        self._subscription = self._node.create_subscription(Odometry, "/scout1/odom", self._on_odom, 10)

    def _on_odom(self, message) -> None:
        self._odom = message

    def spin_once(self, timeout: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=float(timeout))

    def wait_for_odom(self, *, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline and self._odom is None:
            self.spin_once(0.1)
        assert self._odom is not None, "no /scout1/odom received from MuJoCo"

    def latest_x(self) -> float:
        assert self._odom is not None
        return float(self._odom.pose.pose.position.x)

    def publish_goal_after_subscriber(self, *, timeout: float) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self._publisher.get_subscription_count() > 0:
                break
            self.spin_once(0.1)
        assert self._publisher.get_subscription_count() > 0, "MPPI did not subscribe to /move_base_simple/goal"
        end_time = time.monotonic() + 2.0
        while time.monotonic() < end_time:
            self._publisher.publish(self._goal_message())
            self.spin_once(0.1)

    def _goal_message(self):
        assert self._odom is not None
        if self._fixed_goal_xy is None:
            self._fixed_goal_xy = (
                float(self._odom.pose.pose.position.x) + self._goal_offset_x,
                float(self._odom.pose.pose.position.y),
            )
        goal = self._pose_type()
        goal.header.frame_id = "map"
        goal.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.position.x = self._fixed_goal_xy[0]
        goal.pose.position.y = self._fixed_goal_xy[1]
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        return goal

    def close(self) -> None:
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
