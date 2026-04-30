import numpy as np
import pytest

from b2_fdm_mppi.simulation.random_scenario import sample_start_goal


def scenario_config(**overrides):
    config = {
        "scenario": {
            "random_start_goal_enabled": True,
            "random_seed": 123,
            "x_range": [5.0, 95.0],
            "y_range": [5.0, 95.0],
            "distance_range": [5.0, 20.0],
            "min_obstacle_clearance": 2.0,
            "max_attempts": 5000,
            "start_yaw": 0.0,
            "goal_yaw": 0.0,
        }
    }
    config["scenario"].update(overrides)
    return config


def test_sample_start_goal_is_reproducible_for_same_seed():
    first_start, first_goal = sample_start_goal(scenario_config())
    second_start, second_goal = sample_start_goal(scenario_config())

    assert np.allclose(first_start, second_start)
    assert np.allclose(first_goal, second_goal)


def test_sample_start_goal_distance_and_bounds():
    start, goal = sample_start_goal(scenario_config())
    distance = np.linalg.norm(goal[:2] - start[:2])

    assert 5.0 <= distance <= 20.0
    assert 5.0 <= start[0] <= 95.0
    assert 5.0 <= start[1] <= 95.0
    assert 5.0 <= goal[0] <= 95.0
    assert 5.0 <= goal[1] <= 95.0


def test_sample_start_goal_respects_obstacle_clearance():
    obstacles = np.array([[50.0, 50.0, 5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    start, goal = sample_start_goal(
        scenario_config(
            random_seed=5,
            x_range=[40.0, 60.0],
            y_range=[40.0, 60.0],
            distance_range=[5.0, 8.0],
            min_obstacle_clearance=2.0,
        ),
        obstacles=obstacles,
    )

    for point in (start[:2], goal[:2]):
        assert np.linalg.norm(point - obstacles[0, :2]) > obstacles[0, 2] + 2.0


def test_sample_start_goal_raises_when_no_valid_pair_exists():
    obstacles = np.array([[50.0, 50.0, 20.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="start/goal"):
        sample_start_goal(
            scenario_config(
                x_range=[49.0, 51.0],
                y_range=[49.0, 51.0],
                distance_range=[5.0, 6.0],
                max_attempts=10,
            ),
            obstacles=obstacles,
        )
