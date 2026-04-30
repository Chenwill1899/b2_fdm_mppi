import numpy as np
import pytest

from b2_fdm_mppi.simulation.random_obstacles import generate_random_obstacles


def make_random_obstacle_config(**overrides):
    config = {
        "random_enabled": True,
        "random_seed": 123,
        "num_random": 12,
        "radius_range": [0.6, 2.0],
        "x_range": [8.0, 92.0],
        "y_range": [8.0, 92.0],
        "min_obstacle_gap": 1.0,
        "min_start_goal_clearance": 5.0,
    }
    config.update(overrides)
    return config


def test_random_obstacles_are_reproducible_and_valid():
    config = make_random_obstacle_config()
    start = np.array([5.0, 50.0], dtype=np.float32)
    goal = np.array([95.0, 50.0], dtype=np.float32)

    first = generate_random_obstacles(config, start, goal)
    second = generate_random_obstacles(config, start, goal)

    assert first.shape == (12, 7)
    assert np.allclose(first, second)
    assert np.all(first[:, 2] >= 0.6)
    assert np.all(first[:, 2] <= 2.0)
    assert np.all(first[:, 0] >= 8.0)
    assert np.all(first[:, 0] <= 92.0)
    assert np.all(first[:, 1] >= 8.0)
    assert np.all(first[:, 1] <= 92.0)
    assert np.all(first[:, 3:] == 0.0)

    for obstacle in first:
        center = obstacle[:2]
        radius = obstacle[2]
        assert np.linalg.norm(center - start) > radius + 5.0
        assert np.linalg.norm(center - goal) > radius + 5.0

    for idx, obstacle in enumerate(first):
        for other in first[idx + 1 :]:
            distance = np.linalg.norm(obstacle[:2] - other[:2])
            assert distance > obstacle[2] + other[2] + 1.0


def test_random_obstacles_raise_when_scene_is_too_crowded():
    config = make_random_obstacle_config(
        num_random=5,
        radius_range=[2.0, 2.0],
        x_range=[0.0, 1.0],
        y_range=[0.0, 1.0],
        max_attempts=20,
    )

    with pytest.raises(ValueError, match="too crowded"):
        generate_random_obstacles(
            config,
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )
