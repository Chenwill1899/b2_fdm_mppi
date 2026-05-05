from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from b2_fdm_mppi.mujoco_closed_loop import (
    ElevationMapObstacleAdapter,
    ExternalPathAdapter,
    ExternalPathConfig,
    GlobalPathConfig,
    LocalCostmapAdapter,
    LocalCostmapConfig,
    LocalGoalConfig,
    CommandFilterConfig,
    MujocoClosedLoopRecorder,
    append_history_state,
    body_yaw_from_quaternion,
    compute_executed_residual,
    decode_grid_map_layer,
    final_approach_control,
    filter_diff_drive_command,
    filter_mujoco_command,
    grid_map_traversability_obstacles,
    initial_odom_timeout_expired,
    merge_static_and_map_obstacles,
    odom_message_to_state,
    omni_twist_command,
    optional_goal_from_config,
    path_terminal_goal,
    plan_global_path_astar,
    predict_omni_rollout,
    project_point_to_path,
    project_mujoco_state,
    point_segment_distance,
    project_scout_state,
    runtime_goal_required,
    ros_path_message_to_waypoints,
    scout_twist_command,
    select_external_path_goal,
    select_obstacle_aware_goal,
    select_path_lookahead_goal,
    sync_runtime_goal_relief_center,
    _limit_path_points,
    yaw_to_quaternion,
)
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.experiment import build_experiment_config


def path_turn_sum(path: np.ndarray) -> float:
    vectors = np.diff(np.asarray(path, dtype=np.float32).reshape(-1, 2), axis=0)
    headings = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float(np.sum(np.abs(np.diff(headings))))


def make_odom(*, yaw: float = 0.0, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
    half = yaw * 0.5
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.2, y=-0.4),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=np.sin(half), w=np.cos(half)),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=vx, y=vy),
                angular=SimpleNamespace(z=wz),
            )
        ),
    )


def make_grid_map(layer_values: np.ndarray, *, layer: str = "traversability", resolution: float = 0.5):
    rows, cols = layer_values.shape
    data = np.asarray(layer_values, dtype=np.float32).flatten(order="F").tolist()
    return SimpleNamespace(
        info=SimpleNamespace(
            resolution=resolution,
            length_x=rows * resolution,
            length_y=cols * resolution,
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        layers=[layer],
        data=[
            SimpleNamespace(
                data=data,
                layout=SimpleNamespace(
                    dim=[
                        SimpleNamespace(label="column_index", size=cols, stride=rows * cols),
                        SimpleNamespace(label="row_index", size=rows, stride=rows),
                    ]
                ),
            )
        ],
    )


def make_occupancy_elevation(
    values: np.ndarray,
    *,
    layer: str = "reward_cost",
    resolution: float = 0.5,
    origin: tuple[float, float] = (-1.0, -1.0),
    occupancy_data: list[int] | None = None,
):
    array = np.asarray(values, dtype=np.float32)
    height, width = array.shape
    occupancy = SimpleNamespace(
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=float(origin[0]), y=float(origin[1]), z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        data=(
            np.asarray(occupancy_data, dtype=np.int8).reshape(-1).tolist()
            if occupancy_data is not None
            else np.zeros(width * height, dtype=np.int8).tolist()
        ),
    )
    msg = SimpleNamespace(
        occupancy=occupancy,
        height=np.zeros(width * height, dtype=np.float32).tolist(),
        roughness=np.zeros(width * height, dtype=np.float32).tolist(),
        cost_map=np.zeros(width * height, dtype=np.float32).tolist(),
        reward_cost=np.zeros(width * height, dtype=np.float32).tolist(),
    )
    setattr(msg, layer, array.reshape(-1).tolist())
    return msg


def make_path_message(points: list[tuple[float, float]], *, frame_id: str = "map"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        poses=[
            SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=float(x), y=float(y), z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
            for x, y in points
        ],
    )


def test_body_yaw_from_quaternion_extracts_planar_heading():
    yaw = body_yaw_from_quaternion(SimpleNamespace(x=0.0, y=0.0, z=np.sin(0.3), w=np.cos(0.3)))

    assert yaw == pytest.approx(0.6)


def test_initial_odom_timeout_zero_waits_indefinitely():
    assert initial_odom_timeout_expired(10.0, 999.0, 0.0) is False


def test_initial_odom_timeout_positive_expires_after_limit():
    assert initial_odom_timeout_expired(10.0, 10.5, 1.0) is False
    assert initial_odom_timeout_expired(10.0, 11.1, 1.0) is True


def test_odom_message_to_state_maps_nav_odometry_to_fdm_state():
    state = odom_message_to_state(make_odom(yaw=0.25, vx=0.8, vy=0.02, wz=-0.3))

    assert state.dtype == np.float32
    assert state.tolist() == pytest.approx([1.2, -0.4, 0.25, 0.8, 0.02, -0.3])


def test_project_scout_state_zeroes_lateral_velocity_for_differential_drive():
    state = project_scout_state(np.array([1.0, -0.2, 0.3, 0.8, 0.12, -0.4], dtype=np.float32))

    assert state.tolist() == pytest.approx([1.0, -0.2, 0.3, 0.8, 0.0, -0.4])


def test_project_mujoco_state_preserves_lateral_velocity_for_omni_drive():
    state = project_mujoco_state(np.array([1.0, -0.2, 0.3, 0.8, 0.12, -0.4], dtype=np.float32), "omni_freejoint")

    assert state.tolist() == pytest.approx([1.0, -0.2, 0.3, 0.8, 0.12, -0.4])


def test_scout_twist_command_forces_differential_drive_lateral_velocity_to_zero():
    command, constrained = scout_twist_command(np.array([0.7, 0.4, -0.2], dtype=np.float32))

    assert constrained.tolist() == pytest.approx([0.7, 0.0, -0.2])
    assert command.linear.x == pytest.approx(0.7)
    assert command.linear.y == pytest.approx(0.0)
    assert command.linear.z == pytest.approx(0.0)
    assert command.angular.x == pytest.approx(0.0)
    assert command.angular.y == pytest.approx(0.0)
    assert command.angular.z == pytest.approx(-0.2)


def test_omni_twist_command_preserves_lateral_velocity():
    command, constrained = omni_twist_command(np.array([0.7, 0.4, -0.2], dtype=np.float32))

    assert constrained.tolist() == pytest.approx([0.7, 0.4, -0.2])
    assert command.linear.x == pytest.approx(0.7)
    assert command.linear.y == pytest.approx(0.4)
    assert command.angular.z == pytest.approx(-0.2)


def test_yaw_to_quaternion_encodes_planar_heading():
    qx, qy, qz, qw = yaw_to_quaternion(np.pi / 2.0)

    assert qx == pytest.approx(0.0)
    assert qy == pytest.approx(0.0)
    assert qz == pytest.approx(np.sin(np.pi / 4.0))
    assert qw == pytest.approx(np.cos(np.pi / 4.0))


def test_predict_omni_rollout_integrates_body_frame_controls():
    controls = np.array([[1.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)

    states = predict_omni_rollout(
        np.zeros(6, dtype=np.float32),
        controls,
        dt=0.1,
        max_control=np.array([1.0, 0.25, 1.0], dtype=np.float32),
    )

    assert states.shape == (3, 6)
    assert states[-1, 0] == pytest.approx(0.2)
    assert states[-1, 1] == pytest.approx(0.025)
    assert states[-1, 4] == pytest.approx(0.25)


def test_append_history_state_caps_history_to_recent_points():
    history = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    updated = append_history_state(
        history,
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        max_points=2,
    )

    assert updated.shape == (2, 6)
    assert updated[:, 0].tolist() == pytest.approx([1.0, 2.0])


def test_final_approach_control_rotates_before_driving_forward():
    config = {
        "robot": {"max_vx": 1.0, "max_wz": 1.0},
        "final_controller": {"enabled": True, "trigger_distance": 2.0, "rotate_threshold": 0.6},
    }
    state = np.array([17.3, 1.3, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control = final_approach_control(state, goal, config)

    assert control is not None
    assert control[0] == pytest.approx(0.0)
    assert control[2] < 0.0


def test_final_approach_control_drives_when_aligned():
    config = {
        "robot": {"max_vx": 1.0, "max_wz": 1.0},
        "final_controller": {"enabled": True, "trigger_distance": 2.0, "rotate_threshold": 0.6},
    }
    state = np.array([17.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control = final_approach_control(state, goal, config)

    assert control is not None
    assert control[0] > 0.0
    assert abs(float(control[2])) < 1e-6


def test_final_approach_control_uses_lateral_velocity_for_omni():
    config = {
        "mujoco": {"drive_mode": "omni_freejoint"},
        "robot": {"max_vx": 1.0, "max_vy": 1.0, "max_wz": 1.0},
        "final_controller": {"enabled": True, "trigger_distance": 3.0, "xy_gain": 0.7},
    }
    state = np.array([17.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control = final_approach_control(state, goal, config)

    assert control is not None
    assert control[0] > 0.0
    assert control[1] < 0.0
    assert abs(float(control[2])) < 1e-6


def test_final_approach_control_limits_lateral_velocity_for_smooth_omni_finish():
    config = {
        "mujoco": {"drive_mode": "omni_freejoint"},
        "robot": {"max_vx": 1.0, "max_vy": 1.0, "max_wz": 1.0},
        "final_controller": {
            "enabled": True,
            "trigger_distance": 3.0,
            "xy_gain": 0.85,
            "lateral_gain": 0.18,
            "max_vy": 0.10,
            "heading_gain": 0.75,
            "final_yaw_gain": 0.35,
        },
    }
    state = np.array([17.8, -0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control = final_approach_control(state, goal, config)

    assert control is not None
    assert control[0] > 0.0
    assert 0.0 < control[1]
    assert control[1] == pytest.approx(0.10)
    assert control[2] > 0.0


def test_filter_diff_drive_command_low_passes_and_rate_limits():
    cfg = CommandFilterConfig(enabled=True, alpha=0.5, max_ax=0.4, max_awz=1.0)
    target = np.array([1.0, 0.5, 1.0], dtype=np.float32)
    previous = np.zeros(3, dtype=np.float32)

    filtered = filter_diff_drive_command(target, previous, cfg, dt=0.1)

    assert filtered.tolist() == pytest.approx([0.04, 0.0, 0.1])


def test_filter_diff_drive_command_returns_projected_target_when_disabled():
    cfg = CommandFilterConfig(enabled=False)

    filtered = filter_diff_drive_command(
        np.array([0.3, 0.4, -0.2], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
        dt=0.1,
    )

    assert filtered.tolist() == pytest.approx([0.3, 0.0, -0.2])


def test_filter_mujoco_command_preserves_omni_lateral_when_disabled():
    cfg = CommandFilterConfig(enabled=False, drive_mode="omni_freejoint")

    filtered = filter_diff_drive_command(
        np.array([0.3, 0.4, -0.2], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
        dt=0.1,
    )

    assert filtered.tolist() == pytest.approx([0.3, 0.4, -0.2])


def test_filter_mujoco_command_scales_omni_lateral_target_before_rate_limit():
    cfg = CommandFilterConfig(
        enabled=True,
        alpha=0.0,
        max_ax=10.0,
        max_ay=10.0,
        max_awz=10.0,
        drive_mode="omni_freejoint",
        lateral_scale=0.5,
    )

    filtered = filter_diff_drive_command(
        np.array([0.3, 0.4, -0.2], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
        dt=0.1,
    )

    assert filtered.tolist() == pytest.approx([0.3, 0.2, -0.2])


def test_filter_mujoco_command_applies_turn_forward_floor_before_rate_limit():
    cfg = CommandFilterConfig(
        enabled=True,
        alpha=0.0,
        max_ax=10.0,
        max_ay=10.0,
        max_awz=10.0,
        drive_mode="omni_freejoint",
        min_turn_vx=0.16,
        turn_wz_threshold=0.12,
        min_turn_vx_goal_distance=1.6,
    )

    filtered = filter_diff_drive_command(
        np.array([0.04, 0.0, 0.2], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
        dt=0.1,
    )

    assert filtered.tolist() == pytest.approx([0.16, 0.0, 0.2])


def test_filter_mujoco_command_does_not_apply_turn_floor_near_goal():
    cfg = CommandFilterConfig(
        enabled=True,
        alpha=0.0,
        max_ax=10.0,
        max_ay=10.0,
        max_awz=10.0,
        drive_mode="omni_freejoint",
        min_turn_vx=0.16,
        turn_wz_threshold=0.12,
        min_turn_vx_goal_distance=1.6,
    )

    filtered = filter_mujoco_command(
        np.array([0.04, 0.0, 0.2], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
        dt=0.1,
        distance_to_goal=1.0,
    )

    assert filtered.tolist() == pytest.approx([0.04, 0.0, 0.2])


def test_compute_executed_residual_uses_measured_body_velocity_minus_command():
    residual = compute_executed_residual(
        measured_state=np.array([0.0, 0.0, 0.0, 0.45, 0.03, 0.11], dtype=np.float32),
        commanded_control=np.array([0.5, 0.0, 0.2], dtype=np.float32),
    )

    assert residual.tolist() == pytest.approx([-0.05, 0.03, -0.09])


def test_point_segment_distance_clamps_to_segment_endpoints():
    distance = point_segment_distance(
        np.array([2.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert distance == pytest.approx(np.sqrt(2.0))


def test_select_obstacle_aware_goal_keeps_direct_goal_without_corridor_obstacle():
    cfg = LocalGoalConfig(enabled=True, lookahead=4.0, lateral_offsets=(0.0, 1.5, -1.5))
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[4.0, 3.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    local_goal = select_obstacle_aware_goal(state, goal, obstacles, cfg)

    assert local_goal[:2].tolist() == pytest.approx([4.0, 0.0])


def test_select_obstacle_aware_goal_moves_laterally_when_direct_corridor_is_blocked():
    cfg = LocalGoalConfig(
        enabled=True,
        lookahead=4.0,
        recenter_gain=1.0,
        lateral_offsets=(0.0, 1.5, -1.5),
        corridor_buffer=0.5,
        obstacle_weight=100.0,
        lateral_weight=0.05,
        goal_weight=0.01,
    )
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[3.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    local_goal = select_obstacle_aware_goal(state, goal, obstacles, cfg)

    assert local_goal[0] == pytest.approx(4.0)
    assert abs(float(local_goal[1])) == pytest.approx(1.5)
    assert abs(float(local_goal[2])) > 0.1


def test_select_obstacle_aware_goal_recenters_gradually_from_bypass_lane():
    cfg = LocalGoalConfig(
        enabled=True,
        lookahead=6.0,
        recenter_gain=0.25,
        lateral_offsets=(0.0,),
        final_approach_distance=0.0,
    )
    state = np.array([6.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[8.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    local_goal = select_obstacle_aware_goal(state, goal, obstacles, cfg)

    assert local_goal[1] > 1.5


def test_select_obstacle_aware_goal_uses_global_goal_for_final_approach():
    cfg = LocalGoalConfig(enabled=True, lookahead=6.0, recenter_gain=0.25, final_approach_distance=6.0)
    state = np.array([17.8, 4.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[16.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    local_goal = select_obstacle_aware_goal(state, goal, obstacles, cfg)

    assert local_goal[:2].tolist() == pytest.approx(goal[:2].tolist())
    assert local_goal[2] < -1.0


def test_plan_global_path_astar_routes_around_blocking_obstacle():
    cfg = GlobalPathConfig(enabled=True, resolution=0.25, padding=1.5, lookahead=2.0, obstacle_inflation=0.0)
    path = plan_global_path_astar(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([6.0, 0.0], dtype=np.float32),
        np.array([[3.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0]], dtype=np.float32),
        cfg,
        robot_radius=0.35,
        safety_dist=0.10,
    )

    assert len(path) > 2
    assert path[0].tolist() == pytest.approx([0.0, 0.0])
    assert path[-1].tolist() == pytest.approx([6.0, 0.0])
    assert np.max(np.abs(path[:, 1])) > 0.7


def test_plan_global_path_astar_smooths_waypoint_corners_with_clearance():
    base_cfg = GlobalPathConfig(
        enabled=True,
        resolution=0.25,
        padding=1.5,
        lookahead=2.0,
        obstacle_inflation=0.0,
        smoothing_iterations=0,
    )
    smooth_cfg = GlobalPathConfig(
        enabled=True,
        resolution=0.25,
        padding=1.5,
        lookahead=2.0,
        obstacle_inflation=0.0,
        smoothing_iterations=8,
        smoothing_alpha=0.35,
    )
    obstacles = np.array([[3.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0]], dtype=np.float32)

    jagged = plan_global_path_astar(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([6.0, 0.0], dtype=np.float32),
        obstacles,
        base_cfg,
        robot_radius=0.35,
        safety_dist=0.10,
    )
    smooth = plan_global_path_astar(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([6.0, 0.0], dtype=np.float32),
        obstacles,
        smooth_cfg,
        robot_radius=0.35,
        safety_dist=0.10,
    )

    assert smooth[0].tolist() == pytest.approx(jagged[0].tolist())
    assert smooth[-1].tolist() == pytest.approx(jagged[-1].tolist())
    assert path_turn_sum(smooth) < path_turn_sum(jagged)
    clearance = np.min(np.linalg.norm(smooth[:, :2] - obstacles[0, :2], axis=1))
    assert float(clearance) >= float(obstacles[0, 2] + 0.35 + 0.10) - 0.05


def test_select_path_lookahead_goal_tracks_frontend_waypoint():
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [6.0, 1.0]], dtype=np.float32)

    local_goal = select_path_lookahead_goal(state, goal, path, lookahead=2.0)

    assert local_goal[0] == pytest.approx(1.7071068)
    assert local_goal[1] == pytest.approx(0.7071068)
    assert local_goal[2] > 0.0


def test_project_point_to_path_projects_between_waypoints():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 1.0]], dtype=np.float32)

    segment_idx, projected, distance = project_point_to_path(np.array([1.2, 0.7], dtype=np.float32), path)

    assert segment_idx == 0
    assert projected.tolist() == pytest.approx([1.2, 0.0])
    assert distance == pytest.approx(0.7)


def test_select_path_lookahead_goal_uses_polyline_projection_not_nearest_waypoint():
    state = np.array([1.2, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)

    local_goal = select_path_lookahead_goal(state, goal, path, lookahead=1.0)

    assert local_goal[0] == pytest.approx(2.2)
    assert local_goal[1] == pytest.approx(0.0)
    assert local_goal[2] < 0.0


def test_select_path_lookahead_goal_can_use_path_tangent_yaw():
    state = np.array([1.2, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([5.0, 0.0, 0.7, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)

    local_goal = select_path_lookahead_goal(state, goal, path, lookahead=1.0, yaw_mode="path_tangent")

    assert local_goal[0] == pytest.approx(2.2)
    assert local_goal[1] == pytest.approx(0.0)
    assert local_goal[2] == pytest.approx(0.0)


def test_ros_path_message_to_waypoints_extracts_xy_points():
    path_msg = make_path_message([(0.0, 0.0), (1.5, -0.2), (2.0, 0.5)])

    waypoints = ros_path_message_to_waypoints(path_msg)

    assert waypoints.dtype == np.float32
    np.testing.assert_allclose(waypoints, np.array([[0.0, 0.0], [1.5, -0.2], [2.0, 0.5]], dtype=np.float32))


def test_external_path_adapter_reports_only_fresh_nonempty_path():
    adapter = ExternalPathAdapter(ExternalPathConfig(enabled=True, stale_timeout=0.5))

    assert adapter.has_fresh_path(now=1.0) is False
    adapter.update(make_path_message([(0.0, 0.0), (1.0, 0.0)]), now=1.0)

    assert adapter.has_fresh_path(now=1.4) is True
    assert adapter.has_fresh_path(now=1.6) is False


def test_select_external_path_goal_tracks_smooth_path_lookahead():
    cfg = ExternalPathConfig(enabled=True, lookahead=1.5, max_points=4)
    state = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([5.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [5.0, 1.0]], dtype=np.float32)

    local_goal, active_path = select_external_path_goal(state, goal, path, cfg)

    assert local_goal[0] == pytest.approx(1.4949747)
    assert local_goal[1] == pytest.approx(0.4949747)
    assert local_goal[2] > 0.0
    assert active_path.shape == (4, 2)


def test_path_terminal_goal_uses_external_path_endpoint_without_fixed_goal():
    state = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 1.0]], dtype=np.float32)

    goal = path_terminal_goal(path, state)

    assert optional_goal_from_config({"simulation": {}}) is None
    assert goal[:2].tolist() == pytest.approx([4.0, 1.0])
    assert goal[2] == pytest.approx(0.0)


def test_runtime_goal_required_ignores_yaml_goal_until_rviz_goal_arrives():
    config = {
        "simulation": {"goal": [18.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        "goal_topic": {"enabled": True, "required": True},
    }

    assert runtime_goal_required(config) is True
    assert optional_goal_from_config(config) is None


def test_runtime_goal_updates_auto_goal_relief_center():
    config = {
        "terrain": {
            "enabled": True,
            "goal_relief": {
                "enabled": True,
                "center": "auto",
                "sigma": [2.0, 1.2],
                "strength": 0.75,
                "floor": 0.25,
            },
        },
    }
    terrain = TerrainField.from_config(config["terrain"])
    controller = SimpleNamespace(terrain=TerrainField.from_config(config["terrain"]))
    goal = np.array([18.0, -1.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    sync_runtime_goal_relief_center(config, goal, terrain, controller, follow_goal=True)

    assert config["terrain"]["goal_relief"]["center"] == pytest.approx([18.0, -1.5])
    assert terrain.goal_relief["center"] == pytest.approx([18.0, -1.5])
    assert controller.terrain.goal_relief["center"] == pytest.approx([18.0, -1.5])
    terrain.feature(18.0, -1.5)


def test_runtime_goal_relief_auto_center_follows_repeated_goals():
    config = {
        "terrain": {
            "enabled": True,
            "goal_relief": {
                "enabled": True,
                "center": "auto",
            },
        },
    }
    terrain = TerrainField.from_config(config["terrain"])

    sync_runtime_goal_relief_center(
        config,
        np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        terrain,
        follow_goal=True,
    )
    sync_runtime_goal_relief_center(
        config,
        np.array([8.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        terrain,
        follow_goal=True,
    )

    assert config["terrain"]["goal_relief"]["center"] == pytest.approx([8.0, 3.0])
    assert terrain.goal_relief["center"] == pytest.approx([8.0, 3.0])


def test_limit_path_points_preserves_endpoints_and_caps_dense_path():
    path = np.column_stack([np.arange(100, dtype=np.float32), np.zeros(100, dtype=np.float32)])

    limited = _limit_path_points(path, max_points=12)

    assert len(limited) <= 12
    assert limited[0].tolist() == pytest.approx(path[0].tolist())
    assert limited[-1].tolist() == pytest.approx(path[-1].tolist())


def test_decode_grid_map_layer_supports_gridmap_column_layout():
    values = np.array([[1.0, 0.2], [0.8, 0.1]], dtype=np.float32)
    decoded = decode_grid_map_layer(make_grid_map(values), "traversability")

    np.testing.assert_allclose(decoded, values)


def test_grid_map_traversability_obstacles_extracts_near_unsafe_cells():
    values = np.ones((5, 5), dtype=np.float32)
    values[2, 1] = 0.1
    values[0, 0] = 0.1
    msg = make_grid_map(values, resolution=0.5)

    obstacles = grid_map_traversability_obstacles(
        msg,
        state_xy=np.array([0.0, 0.0], dtype=np.float32),
        threshold=0.55,
        obstacle_radius=0.25,
        max_obstacles=4,
        min_distance=0.0,
        max_distance=2.0,
        stride=1,
    )

    assert obstacles.shape[1] == 7
    assert len(obstacles) >= 1
    assert obstacles[:, 2].tolist() == pytest.approx([0.25] * len(obstacles))


def test_grid_map_obstacles_can_use_elevation_height_threshold():
    values = np.zeros((5, 5), dtype=np.float32)
    values[2, 2] = 0.6
    msg = make_grid_map(values, layer="elevation", resolution=0.5)

    obstacles = grid_map_traversability_obstacles(
        msg,
        state_xy=np.array([0.0, 0.0], dtype=np.float32),
        mode="elevation",
        elevation_threshold=0.25,
        max_obstacles=4,
        min_distance=0.0,
        max_distance=2.0,
        stride=1,
    )

    assert len(obstacles) == 1
    assert obstacles[0, 2] == pytest.approx(0.25)


def test_merge_static_and_map_obstacles_deduplicates_known_static_obstacles():
    static = np.array([[3.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
    map_obstacles = np.array(
        [
            [3.1, 0.1, 0.25, 0.25, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    merged = merge_static_and_map_obstacles(static, map_obstacles, dedupe_distance=0.3)

    assert merged.shape == (2, 7)
    assert merged[0, :2].tolist() == pytest.approx([3.0, 0.0])
    assert merged[1, :2].tolist() == pytest.approx([5.0, 0.0])


def test_elevation_map_obstacle_adapter_respects_required_fresh_map():
    adapter = ElevationMapObstacleAdapter.from_config(
        {"elevation_map": {"enabled": True, "required": True, "stale_timeout": 0.2}}
    )
    assert adapter.required is True
    assert adapter.has_fresh_map(now=1.0) is False
    adapter.update(make_grid_map(np.ones((3, 3), dtype=np.float32)), now=1.0)
    assert adapter.has_fresh_map(now=1.1) is True
    assert adapter.has_fresh_map(now=1.3) is False


def test_local_costmap_adapter_decodes_reward_cost_snapshot():
    adapter = LocalCostmapAdapter(
        LocalCostmapConfig(
            enabled=True,
            required=True,
            layer="reward_cost",
            cost_weight=35.0,
            cost_power=2.0,
            unknown_cost=100.0,
            max_cost=80.0,
            stale_timeout=0.5,
        )
    )
    msg = make_occupancy_elevation(
        np.array([[0.0, 10.0], [40.0, 80.0]], dtype=np.float32),
        resolution=0.25,
        origin=(-2.0, -1.0),
    )

    adapter.update(msg, now=3.0)
    snapshot = adapter.snapshot(now=3.2)

    assert adapter.has_fresh_map(now=3.2) is True
    assert snapshot["enabled"] is True
    assert snapshot["width"] == 2
    assert snapshot["height"] == 2
    assert snapshot["resolution"] == pytest.approx(0.25)
    assert snapshot["origin"].tolist() == pytest.approx([-2.0, -1.0])
    assert snapshot["data"].tolist() == pytest.approx([0.0, 10.0, 40.0, 80.0])
    assert snapshot["weight"] == pytest.approx(35.0)
    assert snapshot["power"] == pytest.approx(2.0)
    assert snapshot["unknown_cost"] == pytest.approx(100.0)
    assert snapshot["max_cost"] == pytest.approx(80.0)
    assert adapter.has_fresh_map(now=3.6) is False
    assert adapter.snapshot(now=3.6)["enabled"] is False


def test_local_costmap_adapter_marks_occupancy_unknown_cells():
    adapter = LocalCostmapAdapter(
        LocalCostmapConfig(
            enabled=True,
            layer="reward_cost",
            unknown_clear_radius=0.9,
            unknown_clear_value=0.0,
        )
    )
    msg = make_occupancy_elevation(
        np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float32),
        occupancy_data=[0, -1, -1, 0],
    )

    adapter.update(msg, now=1.0)
    snapshot = adapter.snapshot(now=1.0)

    assert snapshot["unknown_mask"].tolist() == [False, True, True, False]
    assert snapshot["unknown_clear_radius"] == pytest.approx(0.9)
    assert snapshot["unknown_clear_value"] == pytest.approx(0.0)


def test_mujoco_scout_profile_uses_direct_local_costmap_not_smooth_path():
    config, _metadata = build_experiment_config("configs/mujoco_scout.yaml", controller_name="nominal_numpy")

    assert runtime_goal_required(config) is True
    assert "goal" not in config["simulation"]
    assert config["mujoco"]["drive_mode"] == "omni_freejoint"
    assert config["external_path"]["enabled"] is False
    assert config["global_path"]["enabled"] is False
    assert config["local_goal"]["enabled"] is False
    assert config["final_controller"]["enabled"] is True
    assert config["final_controller"]["max_vy"] <= 0.03
    assert config["local_costmap"]["enabled"] is True
    assert config["local_costmap"]["required"] is True
    assert config["local_costmap"]["topic"] == "/msg_local_reward"
    assert config["local_costmap"]["layer"] == "reward_cost"
    assert config["local_costmap"]["cost_weight"] <= 6.0
    assert config["local_costmap"]["cost_power"] >= 3.0
    assert config["local_costmap"]["unknown_clear_radius"] == pytest.approx(1.0)
    assert config["goal_topic"]["enabled"] is True
    assert config["goal_topic"]["required"] is True
    assert config["goal_topic"]["topic"] == "/move_base_simple/goal"
    assert config["robot"]["max_vy"] <= 0.2
    assert config["command_filter"]["lateral_scale"] <= 0.4
    assert config["mppi"]["std_normal"][1] <= 0.08
    assert config["mppi"]["lateral_weight"] >= 1.0
    assert config["mppi"]["goal_progress_weight"] > 0.0
    assert config["mppi"]["heading_to_goal_weight"] > 0.0
    assert config["mppi"]["path_tracking_weight"] == pytest.approx(0.0)
    assert config["mppi"]["path_progress_weight"] == pytest.approx(0.0)


def test_closed_loop_recorder_writes_experiment_compatible_outputs(tmp_path):
    recorder = MujocoClosedLoopRecorder(
        results_path=tmp_path,
        config={"simulation": {"goal": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, "results": {}},
        controller_name="nominal",
        backend="numpy",
        seed=123,
    )
    plan_id = recorder.record_global_path(
        step=0,
        path=np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]], dtype=np.float32),
    )
    recorder.record_step(
        state=np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.1], dtype=np.float32),
        raw_control=np.array([0.5, 0.2, 0.2], dtype=np.float32),
        commanded_control=np.array([0.5, 0.0, 0.2], dtype=np.float32),
        measured_control=np.array([0.4, 0.0, 0.1], dtype=np.float32),
        terrain_features=np.array([0.0, 0.0, 0.1, 0.7], dtype=np.float32),
        terrain_risk=0.2,
        mppi_time_ms=1.5,
        min_cost=3.0,
        planning_goal=np.array([1.0, 0.5, 0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        active_path_plan_id=plan_id,
    )
    summary = recorder.write(
        final_state=np.array([0.04, 0.0, 0.01, 0.4, 0.0, 0.1], dtype=np.float32),
        reached_goal=False,
        failed=False,
        failure_reason=None,
    )

    assert summary["steps"] == 1
    assert summary["path_tracking_mean_m"] == pytest.approx(0.0)
    assert (tmp_path / "experiment_summary.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "trajectory.csv").is_file()
    assert (tmp_path / "controls.csv").is_file()
    assert (tmp_path / "raw_controls.csv").is_file()
    assert (tmp_path / "residuals.csv").is_file()
    assert (tmp_path / "terrain.csv").is_file()
    assert (tmp_path / "global_path.csv").is_file()
    assert (tmp_path / "planning_goals.csv").is_file()
    assert (tmp_path / "trajectory_overlay.png").is_file()
    assert (tmp_path / "trajectory.png").is_file()
    assert (tmp_path / "time_results.csv").is_file()
    residuals = pd.read_csv(tmp_path / "residuals.csv")
    assert residuals.loc[0, "cmd_vy"] == pytest.approx(0.0)
    assert residuals.loc[0, "exec_du_vx"] == pytest.approx(-0.1)
    assert residuals.loc[0, "oracle_du_vx"] == pytest.approx(-0.1)
    trajectory = pd.read_csv(tmp_path / "trajectory.csv")
    assert len(trajectory) == 2
    assert trajectory.iloc[-1]["step"] == 1
    global_path = pd.read_csv(tmp_path / "global_path.csv")
    assert global_path["plan_id"].tolist() == [0, 0, 0]
    planning_goals = pd.read_csv(tmp_path / "planning_goals.csv")
    assert planning_goals.loc[0, "plan_id"] == 0
    assert planning_goals.loc[0, "y"] == pytest.approx(0.5)
    experiment_summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert experiment_summary["controller_name"] == "nominal"
    assert experiment_summary["artifacts"]["residuals_csv"].endswith("residuals.csv")
