import numpy as np
import pytest

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.terrain import TerrainField


def make_controller(seed=1, num_samples=64, horizon_steps=8):
    return MppiOmniNumpy(
        dt=0.1,
        horizon_steps=horizon_steps,
        num_samples=num_samples,
        lambda_=0.5,
        noise_std=np.array([0.25, 0.15, 0.25], dtype=np.float32),
        max_vx=1.5,
        max_vy=0.5,
        max_wz=1.0,
        goal_xy_weight=8.0,
        yaw_weight=0.2,
        control_weight=0.01,
        smooth_weight=0.2,
        obstacle_weight=25.0,
        obstacle_soft_weight=0.0,
        obstacle_influence_dist=0.0,
        max_ax=0.8,
        max_ay=0.5,
        max_awz=1.2,
        velocity_lag_beta=0.35,
        lateral_weight=0.2,
        yaw_rate_weight=0.05,
        accel_weight=0.5,
        jerk_weight=0.0,
        robot_radius=0.6,
        safety_dist=0.3,
        seed=seed,
    )


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
