from pathlib import Path

import pytest

from b2_fdm_mppi.config import load_config, validate_config


def test_default_config_loads_required_groups():
    config = load_config(Path("config/fdm_mppi.yaml"))

    assert config["simulation"]["sampling_rate"] == pytest.approx(10.0)
    assert config["mppi"]["num_trajectories"] == 2496
    assert config["robot"]["state_dim"] == 5
    assert config["results"]["root"] == "./results/sim_results"


def test_validate_config_rejects_bad_goal_length():
    config = load_config(Path("config/fdm_mppi.yaml"))
    config["simulation"]["goal"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="simulation.goal"):
        validate_config(config)
