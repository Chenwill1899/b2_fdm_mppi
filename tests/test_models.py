import numpy as np
import pytest

from b2_fdm_mppi.core.jackal import Jackal
from b2_fdm_mppi.core.obstacle import Obstacle
from b2_fdm_mppi.simulation.runner import goal_reached


def test_jackal_updates_pose_and_velocity_components():
    robot = Jackal(
        state_dim=5,
        dt=0.1,
        max_linear_velocity=1.2,
        max_angular_velocity=1.5,
        r=0.6,
        safety_dist=0.3,
        atau=0.2,
        ob_states_num=7,
    )

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    next_state = robot.update_state(state, np.array([1.0, 0.5], dtype=np.float32))

    assert next_state[0] == pytest.approx(0.1)
    assert next_state[1] == pytest.approx(0.0)
    assert next_state[2] == pytest.approx(0.05)
    assert next_state[3] == pytest.approx(np.cos(0.05), rel=1e-6)
    assert next_state[4] == pytest.approx(np.sin(0.05), rel=1e-6)


def test_obstacle_predicts_virtual_state_horizon():
    obstacle = Obstacle(
        dt=0.1,
        time_horizon=0.3,
        control_freq=10.0,
        slack_weight=10.0,
        max_slack_vari=1.0,
        num_max=2,
        virtual_obstacles=[[1.0, 2.0, 0.4, 0.0, 0.0, -0.5, 0.25]],
    )

    flat = obstacle.update_predict_state_virtual(obstacle.virtual_ob_state)

    assert flat.shape == (21,)
    assert flat[:7].tolist() == pytest.approx([1.0, 2.0, 0.4, 0.0, 0.0, -0.5, 0.25])
    assert flat[14:21].tolist() == pytest.approx([0.9, 2.05, 0.4, 0.0, 0.0, -0.5, 0.25])


def test_goal_reached_uses_xy_distance_threshold():
    assert goal_reached(
        state=np.array([0.1, 0.1, 0.0, 0.0, 0.0]),
        target=np.array([0.2, 0.2, 0.0, 0.0, 0.0]),
        minimum_distance=0.4,
    )

    assert not goal_reached(
        state=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        target=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        minimum_distance=0.4,
    )
