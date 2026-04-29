import numpy as np
import pytest

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.core.omni_b2 import OmniB2


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
        obstacle_weight=25.0,
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


def test_omni_mppi_can_be_created_from_config():
    config = load_config("config/b2_omni_nominal.yaml")

    controller = MppiOmniNumpy.from_config(config, seed=4)

    assert controller.horizon_steps == 20
    assert controller.num_samples == 1024
    assert controller.max_control.tolist() == pytest.approx([1.5, 0.5, 1.0])


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
