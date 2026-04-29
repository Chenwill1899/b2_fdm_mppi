from pathlib import Path

import pytest

from b2_fdm_mppi.config import load_config, validate_config


def test_default_config_loads_required_groups():
    config = load_config(Path("config/fdm_mppi.yaml"))

    assert config["simulation"]["sampling_rate"] == pytest.approx(10.0)
    assert config["mppi"]["num_trajectories"] == 2496
    assert config["robot"]["state_dim"] == 5
    assert config["results"]["root"] == "./results/sim_results"
    assert config["results"]["enable_animation"] is True


def test_short_goal_baseline_config_loads_stage0_parameters():
    config = load_config(Path("config/fdm_mppi_baseline_short.yaml"))

    assert config["simulation"]["goal"] == [3.0, 3.0, 0.0, 0.0, 0.0]
    assert config["simulation"]["max_steps"] == 400
    assert config["simulation"]["time_horizon"] == pytest.approx(2.0)
    assert config["mppi"]["num_trajectories"] == 1024
    assert config["mppi"]["lambda"] == pytest.approx(0.5)
    assert config["results"]["enable_animation"] is True


def test_straight_obstacle_baseline_config_loads_stage0_parameters():
    config = load_config(Path("config/fdm_mppi_baseline_straight_obstacle.yaml"))

    assert config["simulation"]["goal"] == [18.0, 0.0, 0.0, 0.0, 0.0]
    assert config["simulation"]["max_steps"] == 400
    assert config["simulation"]["time_horizon"] == pytest.approx(2.0)
    assert config["obstacles"]["static_enabled"] is True
    assert config["obstacles"]["virtual"] == [
        [6.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0],
        [12.0, -1.0, 0.4, 0.0, 0.0, 0.0, 0.0],
    ]
    assert config["results"]["enable_animation"] is True


def test_validate_config_rejects_bad_goal_length():
    config = load_config(Path("config/fdm_mppi.yaml"))
    config["simulation"]["goal"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="simulation.goal"):
        validate_config(config)
