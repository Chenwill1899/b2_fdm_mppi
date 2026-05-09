import numpy as np
import pytest

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.terrain import TerrainField


def make_controller(seed=1, num_samples=64, horizon_steps=8, **overrides):
    params = {
        "dt": 0.1,
        "horizon_steps": horizon_steps,
        "num_samples": num_samples,
        "lambda_": 0.5,
        "noise_std": np.array([0.25, 0.15, 0.25], dtype=np.float32),
        "max_vx": 1.5,
        "max_vy": 0.5,
        "max_wz": 1.0,
        "goal_xy_weight": 8.0,
        "yaw_weight": 0.2,
        "control_weight": 0.01,
        "smooth_weight": 0.2,
        "obstacle_weight": 25.0,
        "obstacle_soft_weight": 0.0,
        "obstacle_influence_dist": 0.0,
        "max_ax": 0.8,
        "max_ay": 0.5,
        "max_awz": 1.2,
        "velocity_lag_beta": 0.35,
        "lateral_weight": 0.2,
        "yaw_rate_weight": 0.05,
        "accel_weight": 0.5,
        "jerk_weight": 0.0,
        "robot_radius": 0.6,
        "safety_dist": 0.3,
        "seed": seed,
    }
    params.update(overrides)
    return MppiOmniNumpy(**params)


def test_omni_mppi_returns_three_dimensional_limited_control():
    controller = make_controller()
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control, optimal_u, sample_u, normalizer, min_cost = controller.compute_control(
        state, [None, None, None, goal, [], 0]
    )

    assert control.shape == (3,)
    assert optimal_u.shape == (controller.horizon_steps, 3)
    assert sample_u.shape == (controller.draw_num_traj, controller.horizon_steps, 3)
    assert np.isfinite(normalizer)
    assert np.isfinite(min_cost)
    assert abs(control[0]) <= 1.5
    assert abs(control[1]) <= 0.5
    assert abs(control[2]) <= 1.0


def test_omni_mppi_drives_forward_toward_goal():
    controller = make_controller(seed=2, num_samples=128)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control, *_ = controller.compute_control(state, [None, None, None, goal, [], 0])

    assert control[0] > 0.0
    assert abs(control[1]) < 0.3


def test_omni_mppi_obstacle_cost_prefers_lateral_clearance():
    controller = make_controller(seed=3, num_samples=128)
    straight = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    straight[:, 0] = 1.5
    lateral = straight.copy()
    lateral[:, 1] = 0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[0.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    straight_cost = controller.trajectory_cost(state, straight, goal, obstacles)
    lateral_cost = controller.trajectory_cost(state, lateral, goal, obstacles)

    assert lateral_cost < straight_cost


def test_omni_mppi_soft_obstacle_cost_penalizes_far_field_clearance():
    controller = make_controller(seed=12, num_samples=4, horizon_steps=4)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.obstacle_weight = 0.0
    controller.obstacle_soft_weight = 10.0
    controller.obstacle_influence_dist = 2.0
    controls = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    near_obstacle = np.array([[1.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    far_obstacle = np.array([[4.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    assert controller.trajectory_cost(state, controls, goal, near_obstacle) > controller.trajectory_cost(
        state, controls, goal, far_obstacle
    )


def test_omni_mppi_smooth_cost_penalizes_control_jumps():
    controller = make_controller(seed=7, num_samples=8, horizon_steps=6)
    smooth = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    smooth[:, 0] = 0.8
    jerky = smooth.copy()
    jerky[1::2, 1] = 0.5
    jerky[::2, 1] = -0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    smooth_cost = controller.trajectory_cost(state, smooth, goal, obstacles)
    jerky_cost = controller.trajectory_cost(state, jerky, goal, obstacles)

    assert jerky_cost > smooth_cost


def test_omni_mppi_smooth_cost_penalizes_first_control_jump():
    controller = make_controller(seed=8, num_samples=8, horizon_steps=4)
    controller.previous_control = np.array([0.8, 0.0, 0.0], dtype=np.float32)
    continuous = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    continuous[:, 0] = 0.8
    jump = continuous.copy()
    jump[0, 0] = -0.8
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    continuous_cost = controller.trajectory_cost(state, continuous, goal, obstacles)
    jump_cost = controller.trajectory_cost(state, jump, goal, obstacles)

    assert jump_cost > continuous_cost


def test_omni_mppi_rollout_applies_velocity_lag_and_accel_limits():
    controller = make_controller(seed=9, num_samples=2, horizon_steps=4)
    controls = np.zeros((1, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, :] = np.array([1.5, 0.5, 1.0], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)

    states, real_controls = controller._rollout_batch(state, controls, return_controls=True)

    expected_first = np.array([0.08, 0.05, 0.12], dtype=np.float32)
    assert real_controls[0, 0] == pytest.approx(expected_first, abs=1e-6)
    assert states[0, 1, 3:] == pytest.approx(expected_first, abs=1e-6)
    deltas = np.diff(np.concatenate([state[None, None, 3:], real_controls], axis=1), axis=1)
    assert np.max(np.abs(deltas[:, :, 0])) <= controller.max_accel[0] * controller.dt + 1e-6
    assert np.max(np.abs(deltas[:, :, 1])) <= controller.max_accel[1] * controller.dt + 1e-6
    assert np.max(np.abs(deltas[:, :, 2])) <= controller.max_accel[2] * controller.dt + 1e-6


def test_omni_mppi_returns_first_realizable_velocity_response():
    controller = make_controller(seed=11, num_samples=2, horizon_steps=4)
    state = np.zeros(6, dtype=np.float32)
    command = np.array([1.5, -0.5, 1.0], dtype=np.float32)

    control = controller._apply_velocity_response(state, command)

    assert control == pytest.approx([0.08, -0.05, 0.12], abs=1e-6)


def test_omni_mppi_lateral_yaw_and_accel_costs_penalize_usage():
    controller = make_controller(seed=10, num_samples=2, horizon_steps=4)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    baseline = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    lateral_yaw = baseline.copy()
    lateral_yaw[:, 1] = 0.5
    lateral_yaw[:, 2] = 1.0
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    assert controller.trajectory_cost(state, lateral_yaw, goal, obstacles) > controller.trajectory_cost(
        state, baseline, goal, obstacles
    )


def test_omni_mppi_jerk_cost_penalizes_real_velocity_oscillation():
    controller = make_controller(seed=13, num_samples=2, horizon_steps=5)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 10.0
    steady = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    steady[:, 0] = 0.8
    oscillating = steady.copy()
    oscillating[::2, 1] = 0.5
    oscillating[1::2, 1] = -0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    assert controller.trajectory_cost(state, oscillating, goal, obstacles) > controller.trajectory_cost(
        state, steady, goal, obstacles
    )


def test_omni_mppi_path_tracking_cost_penalizes_cross_track_error():
    controller = make_controller(seed=16, num_samples=2, horizon_steps=5)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    controller.path_tracking_weight = 10.0
    controller.path_tracking_tolerance = 0.0
    on_path = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    on_path[:, 0] = 0.8
    off_path = on_path.copy()
    off_path[:, 1] = 0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)
    path = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    assert controller.trajectory_cost(state, off_path, goal, obstacles, path) > controller.trajectory_cost(
        state, on_path, goal, obstacles, path
    )


def test_omni_mppi_path_tracking_weight_zero_preserves_batch_cost():
    controller = make_controller(seed=17, num_samples=2, horizon_steps=5)
    controller.path_tracking_weight = 0.0
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)
    controls = np.zeros((2, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, 0] = 0.8
    controls[1, :, 1] = 0.4
    path = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    without_path = controller.trajectory_cost_batch(state, controls, goal, obstacles)
    with_path = controller.trajectory_cost_batch(state, controls, goal, obstacles, path)

    assert with_path == pytest.approx(without_path)


def test_omni_mppi_local_costmap_penalizes_high_cost_cells():
    controller = make_controller(
        seed=19,
        num_samples=2,
        horizon_steps=4,
        max_ax=1000.0,
        max_ay=1000.0,
        velocity_lag_beta=0.0,
    )
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    controller.path_tracking_weight = 0.0
    controller.path_progress_weight = 0.0
    data = np.zeros((3, 6), dtype=np.float32)
    data[0, :] = 100.0
    costmap = {
        "enabled": True,
        "origin": np.array([0.0, -0.05], dtype=np.float32),
        "resolution": 0.1,
        "width": 6,
        "height": 3,
        "data": data.reshape(-1),
        "weight": 20.0,
        "power": 2.0,
        "unknown_cost": 0.0,
        "max_cost": 100.0,
    }
    straight = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    straight[:, 0] = 1.0
    lateral = straight.copy()
    lateral[:, 1] = 0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    straight_cost = controller.trajectory_cost(state, straight, goal, obstacles, costmap=costmap)
    lateral_cost = controller.trajectory_cost(state, lateral, goal, obstacles, costmap=costmap)

    assert straight_cost > lateral_cost


def test_omni_mppi_local_costmap_footprint_penalizes_grazing_path():
    controller = make_controller(
        seed=24,
        num_samples=2,
        horizon_steps=4,
        max_ax=1000.0,
        max_ay=1000.0,
        velocity_lag_beta=0.0,
    )
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    data = np.zeros((7, 8), dtype=np.float32)
    data[5, :] = 100.0
    costmap = {
        "enabled": True,
        "origin": np.array([0.0, -0.3], dtype=np.float32),
        "resolution": 0.1,
        "width": 8,
        "height": 7,
        "data": data.reshape(-1),
        "weight": 10.0,
        "power": 1.0,
        "unknown_cost": 0.0,
        "max_cost": 100.0,
        "footprint_enabled": True,
        "footprint_radius": 0.25,
        "footprint_safety_margin": 0.0,
        "footprint_sample_count": 16,
    }
    centerline = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    centerline[:, 0] = 1.0
    shifted = centerline.copy()
    shifted[:, 1] = -0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    center_cost = controller.trajectory_cost(state, centerline, goal, obstacles, costmap=costmap)
    shifted_cost = controller.trajectory_cost(state, shifted, goal, obstacles, costmap=costmap)

    assert center_cost > shifted_cost


def test_omni_mppi_local_costmap_treats_out_of_bounds_as_unknown_cost():
    controller = make_controller(
        seed=20,
        num_samples=2,
        horizon_steps=4,
        max_ax=1000.0,
        max_ay=1000.0,
        velocity_lag_beta=0.0,
    )
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    costmap = {
        "enabled": True,
        "origin": np.array([0.0, -0.1], dtype=np.float32),
        "resolution": 0.2,
        "width": 2,
        "height": 2,
        "data": np.zeros(4, dtype=np.float32),
        "weight": 10.0,
        "power": 1.0,
        "unknown_cost": 100.0,
        "max_cost": 100.0,
    }
    inside = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    outside = inside.copy()
    outside[:, 0] = 2.0
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    inside_cost = controller.trajectory_cost(state, inside, goal, obstacles, costmap=costmap)
    outside_cost = controller.trajectory_cost(state, outside, goal, obstacles, costmap=costmap)

    assert outside_cost > inside_cost


def test_omni_mppi_local_costmap_clears_only_nearby_unknown_cells():
    controller = make_controller(seed=21, num_samples=2, horizon_steps=4)
    states = np.zeros((2, 2, 6), dtype=np.float32)
    states[0, :, :2] = np.array([[0.10, 0.0], [0.20, 0.0]], dtype=np.float32)
    states[1, :, :2] = np.array([[0.10, 0.0], [0.60, 0.0]], dtype=np.float32)
    costmap = {
        "enabled": True,
        "origin": np.array([0.0, -0.1], dtype=np.float32),
        "resolution": 0.1,
        "width": 8,
        "height": 3,
        "data": np.full(24, 100.0, dtype=np.float32),
        "unknown_mask": np.ones(24, dtype=bool),
        "unknown_clear_radius": 0.35,
        "unknown_clear_value": 0.0,
        "weight": 10.0,
        "power": 1.0,
        "unknown_cost": 100.0,
        "max_cost": 100.0,
    }
    observed = {**costmap, "unknown_mask": np.zeros(24, dtype=bool)}
    initial_state = np.zeros(6, dtype=np.float32)

    unknown_cost = controller._local_costmap_cost_batch(initial_state, states, costmap)
    observed_cost = controller._local_costmap_cost_batch(initial_state, states, observed)

    assert unknown_cost[0] == pytest.approx(0.0)
    assert unknown_cost[1] > unknown_cost[0]
    assert observed_cost[0] > unknown_cost[0]


def test_omni_mppi_path_progress_reward_prefers_forward_progress():
    controller = make_controller(seed=18, num_samples=2, horizon_steps=5)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    controller.path_tracking_weight = 0.0
    controller.path_progress_weight = 2.0
    slow = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    fast = slow.copy()
    slow[:, 0] = 0.2
    fast[:, 0] = 0.8
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)
    path = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)

    assert controller.trajectory_cost(state, fast, goal, obstacles, path) < controller.trajectory_cost(
        state, slow, goal, obstacles, path
    )


def test_omni_mppi_goal_progress_reward_prefers_reducing_goal_distance():
    controller = make_controller(seed=22, num_samples=2, horizon_steps=5)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    controller.path_tracking_weight = 0.0
    controller.path_progress_weight = 0.0
    controller.goal_progress_weight = 6.0
    stop = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    forward = stop.copy()
    forward[:, 0] = 0.8
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    assert controller.trajectory_cost(state, forward, goal, obstacles) < controller.trajectory_cost(
        state, stop, goal, obstacles
    )


def test_omni_mppi_heading_to_goal_cost_penalizes_sideways_body_heading():
    controller = make_controller(seed=23, num_samples=2, horizon_steps=4)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    controller.path_tracking_weight = 0.0
    controller.path_progress_weight = 0.0
    controller.goal_progress_weight = 0.0
    controller.heading_to_goal_weight = 4.0
    controls = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    goal = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)
    aligned_state = np.zeros(6, dtype=np.float32)
    sideways_state = aligned_state.copy()
    sideways_state[2] = np.pi / 2.0

    assert controller.trajectory_cost(sideways_state, controls, goal, obstacles) > controller.trajectory_cost(
        aligned_state, controls, goal, obstacles
    )


def test_omni_mppi_update_smoothing_blends_nominal_sequence():
    controller = make_controller(seed=25, num_samples=2, horizon_steps=4, update_smoothing_alpha=0.75)
    previous = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    previous[:, 0] = 1.0
    updated = np.zeros_like(previous)
    updated[:, 0] = -1.0
    controller._has_nominal_update = True

    smoothed = controller._smooth_nominal_update(updated, previous)

    assert smoothed[:, 0] == pytest.approx(np.full(controller.horizon_steps, 0.5))


def test_omni_mppi_goal_change_resets_smoothed_nominal_sequence():
    controller = make_controller(seed=26, num_samples=2, horizon_steps=4, update_smoothing_alpha=0.75)
    controller.nominal_u[:, 0] = 1.0
    controller._has_nominal_update = True

    controller._reset_nominal_on_goal_change(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    controller._reset_nominal_on_goal_change(np.array([2.0, 0.0, 0.0], dtype=np.float32))

    assert controller._has_nominal_update is False
    assert controller.nominal_u == pytest.approx(np.zeros_like(controller.nominal_u))


def test_omni_mppi_batch_cost_matches_scalar_costs():
    controller = make_controller(seed=6, num_samples=4, horizon_steps=5)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    controls = np.zeros((4, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, 0] = np.linspace(0.2, 1.2, 4)[:, None]
    controls[:, :, 1] = np.linspace(-0.2, 0.2, 4)[:, None]

    batch_costs = controller.trajectory_cost_batch(state, controls, goal, obstacles)
    scalar_costs = np.array(
        [controller.trajectory_cost(state, control, goal, obstacles) for control in controls]
    )

    assert batch_costs == pytest.approx(scalar_costs, rel=1e-5)


def test_omni_mppi_terrain_risk_weight_zero_preserves_cost():
    terrain = TerrainField(
        enabled=True,
        slope_scale=0.0,
        roughness_scale=0.0,
        friction_base=0.8,
        friction_slope_scale=0.0,
        friction_roughness_scale=0.0,
        patches=[
            {
                "name": "risk_band",
                "type": "band",
                "center": [0.2, 0.0],
                "angle": 0.0,
                "size": [1.0, 0.1],
                "edge_width": 0.0,
                "roughness_delta": 0.7,
                "friction_delta": -0.4,
            }
        ],
    )
    base = make_controller(seed=14, num_samples=2, horizon_steps=3)
    risk_aware_disabled = make_controller(seed=14, num_samples=2, horizon_steps=3)
    risk_aware_disabled.terrain = terrain
    risk_aware_disabled.terrain_risk_weight = 0.0
    controls = np.zeros((base.horizon_steps, 3), dtype=np.float32)
    controls[:, 0] = 0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    assert risk_aware_disabled.trajectory_cost(state, controls, goal, obstacles) == pytest.approx(
        base.trajectory_cost(state, controls, goal, obstacles)
    )


def test_omni_mppi_terrain_risk_cost_penalizes_high_risk_rollout():
    terrain = TerrainField(
        enabled=True,
        slope_scale=0.0,
        roughness_scale=0.0,
        friction_base=0.8,
        friction_slope_scale=0.0,
        friction_roughness_scale=0.0,
        patches=[
            {
                "name": "risk_band",
                "type": "band",
                "center": [0.2, 0.0],
                "angle": 0.0,
                "size": [1.0, 0.1],
                "edge_width": 0.0,
                "roughness_delta": 0.7,
                "friction_delta": -0.4,
            }
        ],
    )
    controller = make_controller(seed=15, num_samples=2, horizon_steps=4)
    controller.goal_xy_weight = 0.0
    controller.yaw_weight = 0.0
    controller.control_weight = 0.0
    controller.smooth_weight = 0.0
    controller.accel_weight = 0.0
    controller.lateral_weight = 0.0
    controller.yaw_rate_weight = 0.0
    controller.jerk_weight = 0.0
    controller.terrain = terrain
    controller.terrain_risk_weight = 10.0
    controller.terrain_risk_threshold = 0.25
    controller.terrain_risk_power = 2.0
    controller.terrain_risk_mode = "excess"
    high_risk = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    high_risk[:, 0] = 0.8
    low_risk = np.zeros((controller.horizon_steps, 3), dtype=np.float32)
    low_risk[:, 1] = 0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    assert controller.trajectory_cost(state, high_risk, goal, obstacles) > controller.trajectory_cost(
        state, low_risk, goal, obstacles
    )


def test_omni_mppi_can_be_created_from_config():
    config = load_config("config/b2_omni_nominal.yaml")

    controller = MppiOmniNumpy.from_config(config, seed=4)

    assert controller.horizon_steps == 25
    assert controller.num_samples == 1024
    assert controller.max_control.tolist() == pytest.approx([1.5, 0.5, 1.0])
    assert controller.max_accel.tolist() == pytest.approx([0.8, 0.5, 1.2])
    assert controller.velocity_lag_beta == pytest.approx(0.35)
    assert controller.lateral_weight == pytest.approx(0.2)
    assert controller.yaw_rate_weight == pytest.approx(0.05)
    assert controller.accel_weight == pytest.approx(0.5)
    assert controller.jerk_weight == pytest.approx(config["mppi"].get("jerk_weight", 0.0))


def test_omni_mppi_from_config_loads_soft_obstacle_params():
    config = load_config("config/b2_omni_kinodynamic.yaml")

    controller = MppiOmniNumpy.from_config(config, seed=4)

    assert controller.obstacle_soft_weight == pytest.approx(config["mppi"]["obstacle_soft_weight"])
    assert controller.obstacle_influence_dist == pytest.approx(config["mppi"]["obstacle_influence_dist"])


def test_omni_mppi_from_config_accepts_tuning_overrides():
    config = load_config("config/b2_omni_nominal.yaml")

    controller = MppiOmniNumpy.from_config(config, seed=4, obstacle_weight=100.0)

    assert controller.obstacle_weight == pytest.approx(100.0)


def test_omni_mppi_from_config_loads_smooth_weight():
    config = load_config("config/b2_omni_nominal.yaml")

    controller = MppiOmniNumpy.from_config(config, seed=4)

    assert controller.smooth_weight == pytest.approx(1.2)


def test_omni_mppi_closed_loop_moves_toward_unobstructed_goal():
    controller = make_controller(seed=5, num_samples=256, horizon_steps=12)
    model = OmniB2(dt=0.1, max_vx=1.5, max_vy=0.5, max_wz=1.0)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    initial_distance = np.linalg.norm(goal[:2] - state[:2])
    for _ in range(20):
        control, *_ = controller.compute_control(state, [None, None, None, goal, [], 0])
        state = model.update_state(state, control)

    assert np.linalg.norm(goal[:2] - state[:2]) < initial_distance
    assert state[0] > 0.5
