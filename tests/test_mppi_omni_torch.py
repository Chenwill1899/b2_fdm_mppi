import numpy as np
import pytest
import torch

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.controllers.mppi_omni_torch import MppiOmniTorch
from b2_fdm_mppi.core.terrain import TerrainField


def make_torch_controller(**overrides):
    params = {
        "dt": 0.1,
        "horizon_steps": 4,
        "num_samples": 5,
        "lambda_": 0.5,
        "noise_std": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "max_vx": 1.0,
        "max_vy": 0.5,
        "max_wz": 0.4,
        "max_ax": 1000.0,
        "max_ay": 1000.0,
        "max_awz": 1000.0,
        "velocity_lag_beta": 0.0,
        "robot_radius": 0.6,
        "safety_dist": 0.25,
        "draw_num_traj": 2,
        "seed": 1,
        "device": "cpu",
    }
    params.update(overrides)
    return MppiOmniTorch(**params)


def make_numpy_controller(**overrides):
    params = {
        "dt": 0.1,
        "horizon_steps": 4,
        "num_samples": 5,
        "lambda_": 0.5,
        "noise_std": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "max_vx": 1.0,
        "max_vy": 0.5,
        "max_wz": 0.4,
        "max_ax": 1000.0,
        "max_ay": 1000.0,
        "max_awz": 1000.0,
        "velocity_lag_beta": 0.0,
        "robot_radius": 0.6,
        "safety_dist": 0.25,
        "draw_num_traj": 2,
        "seed": 1,
    }
    params.update(overrides)
    return MppiOmniNumpy(**params)


def test_nominal_torch_rollout_matches_numpy_without_residual():
    torch_controller = make_torch_controller()
    numpy_controller = make_numpy_controller()
    controls = np.zeros((2, torch_controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, :] = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)

    torch_states, torch_real_controls = torch_controller._rollout_batch(state, controls, return_controls=True)
    numpy_states, numpy_real_controls = numpy_controller._rollout_batch(state, controls, return_controls=True)

    assert torch_states == pytest.approx(numpy_states, abs=1e-6)
    assert torch_real_controls == pytest.approx(numpy_real_controls, abs=1e-6)


def test_nominal_torch_batch_cost_matches_numpy_with_terrain_risk():
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
                "size": [1.0, 0.01],
                "edge_width": 0.0,
                "roughness_delta": 0.7,
                "friction_delta": -0.4,
            }
        ],
    )
    shared = {
        "terrain": terrain,
        "terrain_risk_weight": 10.0,
        "terrain_risk_threshold": 0.25,
        "terrain_risk_power": 2.0,
        "terrain_risk_mode": "excess",
    }
    torch_controller = make_torch_controller(**shared)
    numpy_controller = make_numpy_controller(**shared)
    controls = np.zeros((2, torch_controller.horizon_steps, 3), dtype=np.float32)
    controls[0, :, 0] = 0.8
    controls[1, :, 1] = 0.5
    state = np.zeros(6, dtype=np.float32)
    goal = np.zeros(6, dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    torch_costs = torch_controller.trajectory_cost_batch(state, controls, goal, obstacles)
    numpy_costs = numpy_controller.trajectory_cost_batch(state, controls, goal, obstacles)

    assert torch_costs == pytest.approx(numpy_costs, abs=1e-5)
    assert torch_costs[0] > torch_costs[1]


def test_nominal_torch_compute_control_returns_numpy_controller_outputs():
    controller = make_torch_controller()
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control, optimal_u, sample_u, normalizer, min_cost = controller.compute_control(
        state,
        [None, None, None, goal, np.empty((0, 7), dtype=np.float32), 0],
    )

    assert control.shape == (3,)
    assert optimal_u.shape == (controller.horizon_steps, 3)
    assert sample_u.shape == (controller.draw_num_traj, controller.horizon_steps, 3)
    assert np.isfinite(normalizer)
    assert np.isfinite(min_cost)


def test_nominal_torch_can_be_created_from_config_cpu_device():
    config = load_config("config/b2_omni_oracle_risk_band.yaml")
    config["mppi"]["backend"] = "torch"
    config["mppi"]["device"] = "cpu"
    config["mppi"]["num_trajectories"] = 8
    config["simulation"]["time_horizon"] = 0.3

    controller = MppiOmniTorch.from_config(config, seed=4)

    assert controller.torch_device.type == "cpu"
    assert controller.horizon_steps == 3
    assert controller.num_samples == 8


def test_nominal_torch_profile_records_runtime_buckets():
    controller = make_torch_controller(profile_enabled=True)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    controller.compute_control(
        state,
        [None, None, None, goal, np.empty((0, 7), dtype=np.float32), 0],
    )

    profile = controller.profile_summary()
    assert profile["enabled"] is True
    assert profile["total_calls"] == 1
    for bucket in ("sample_candidates_ms", "rollout_total_ms", "cost_terms_ms", "cpu_transfer_ms"):
        assert bucket in profile["totals_ms"]
        assert profile["totals_ms"][bucket] >= 0.0


def test_nominal_torch_bilinear_sample_many_matches_individual_samples():
    terrain = TerrainField(
        enabled=True,
        noise_enabled=True,
        noise_seed=7,
        noise_grid_size=(5, 6),
        noise_x_range=(-1.0, 2.0),
        noise_y_range=(-2.0, 3.0),
    )
    controller = make_torch_controller(terrain=terrain)
    x = torch.tensor([-2.0, -0.25, 0.5, 1.5, 3.0], dtype=torch.float32)
    y = torch.tensor([-3.0, -1.0, 0.5, 2.5, 4.0], dtype=torch.float32)
    grids = torch.stack([controller.noise_grid_t, controller.noise_grad_x_t, controller.noise_grad_y_t], dim=0)

    batched = controller._bilinear_sample_many_torch(grids, x, y)
    individual = torch.stack(
        [
            controller._bilinear_sample_torch(controller.noise_grid_t, x, y),
            controller._bilinear_sample_torch(controller.noise_grad_x_t, x, y),
            controller._bilinear_sample_torch(controller.noise_grad_y_t, x, y),
        ],
        dim=0,
    )

    assert batched.shape == individual.shape
    assert torch.allclose(batched, individual, atol=1e-6)
