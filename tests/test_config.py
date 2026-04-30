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


def test_b2_omni_nominal_config_loads_stage1_parameters():
    config = load_config(Path("config/b2_omni_nominal.yaml"))

    assert config["simulation"]["goal"] == [18.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert config["simulation"]["initial_state"] == [0.01, 0.01, 0.0, 0.0, 0.0, 0.0]
    assert config["mppi"]["state_dim"] == 6
    assert config["mppi"]["control_dim"] == 3
    assert config["mppi"]["backend"] == "cuda"
    assert config["mppi"]["std_normal"] == [0.20, 0.12, 0.14]
    assert config["mppi"]["lambda"] == pytest.approx(0.6)
    assert config["mppi"]["obstacle_weight"] == pytest.approx(300.0)
    assert config["mppi"]["cbf_weight"] == pytest.approx(0.0)
    assert config["mppi"]["control_weight"] == pytest.approx(0.02)
    assert config["mppi"]["smooth_weight"] == pytest.approx(1.2)
    assert config["mppi"]["lateral_weight"] == pytest.approx(0.2)
    assert config["mppi"]["yaw_rate_weight"] == pytest.approx(0.05)
    assert config["mppi"]["accel_weight"] == pytest.approx(0.5)
    assert config["cbf"]["enabled"] is False
    assert config["cbf"]["type"] == 0
    assert config["robot"]["max_vx"] == pytest.approx(1.5)
    assert config["robot"]["max_vy"] == pytest.approx(0.5)
    assert config["robot"]["max_wz"] == pytest.approx(1.0)
    assert config["robot"]["max_ax"] == pytest.approx(0.8)
    assert config["robot"]["max_ay"] == pytest.approx(0.5)
    assert config["robot"]["max_awz"] == pytest.approx(1.2)
    assert config["robot"]["velocity_lag_beta"] == pytest.approx(0.35)
    assert config["robot"]["safety_dist"] == pytest.approx(0.5)
    assert config["results"]["run_name"] == "b2_omni_nominal"
    assert config["results"]["timestamp_suffix"] is True
    assert config["results"]["overwrite"] is False
    assert config["execution"]["filter_enabled"] is True
    assert config["execution"]["filter_alpha"] == pytest.approx(0.6)


def test_validate_config_rejects_bad_goal_length():
    config = load_config(Path("config/fdm_mppi.yaml"))
    config["simulation"]["goal"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="simulation.goal"):
        validate_config(config)
