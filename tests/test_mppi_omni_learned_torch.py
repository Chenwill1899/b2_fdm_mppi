import numpy as np
import pytest
import torch

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_learned_torch import LearnedFdmMppiOmniTorch, MppiOmniSequenceFdmTorch
from b2_fdm_mppi.core.terrain import TerrainField


class ConstantTorchResidualDynamics:
    checkpoint_path = "stub/best_model.pt"
    normalization_path = "stub/normalization.npz"
    device = "cpu"

    def predict_residual_torch(self, states, commands):
        residuals = torch.zeros((states.shape[0], 3), dtype=states.dtype, device=states.device)
        residuals[:, 0] = 0.2
        residuals[:, 1] = -0.1
        residuals[:, 2] = 0.05
        return residuals


class DummySequenceDynamics:
    sequence_horizon = 3
    include_history_controls = True
    history_steps = 1
    device = "cpu"


def make_controller(*, residual_gain=1.0, profile_enabled=False):
    return LearnedFdmMppiOmniTorch(
        dt=0.1,
        horizon_steps=3,
        num_samples=4,
        lambda_=0.5,
        noise_std=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=0.5,
        max_wz=0.4,
        max_ax=1000.0,
        max_ay=1000.0,
        max_awz=1000.0,
        velocity_lag_beta=0.0,
        robot_radius=0.6,
        safety_dist=0.25,
        draw_num_traj=2,
        seed=1,
        learned_dynamics=ConstantTorchResidualDynamics(),
        device="cpu",
        residual_gain=residual_gain,
        profile_enabled=profile_enabled,
    )


def make_sequence_controller():
    return MppiOmniSequenceFdmTorch(
        dt=0.1,
        horizon_steps=3,
        num_samples=4,
        lambda_=0.5,
        noise_std=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=0.5,
        max_wz=0.4,
        max_ax=1000.0,
        max_ay=1000.0,
        max_awz=1000.0,
        velocity_lag_beta=0.0,
        robot_radius=0.6,
        safety_dist=0.25,
        draw_num_traj=2,
        seed=1,
        learned_dynamics=DummySequenceDynamics(),
        device="cpu",
    )


def test_sequence_fdm_relative_trajectory_is_not_cumulatively_integrated():
    controller = make_sequence_controller()
    initial_state = np.array([10.0, -2.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    rel_traj = torch.tensor(
        [
            [
                [1.0, 0.1, 0.01],
                [2.0, 0.2, 0.02],
                [3.0, 0.3, 0.03],
            ]
        ],
        dtype=torch.float32,
    )

    states = controller._sequence_to_states_torch(initial_state, rel_traj).detach().cpu().numpy()

    assert states.shape == (1, 4, 6)
    assert states[0, 1:, 0] == pytest.approx([11.0, 12.0, 13.0], abs=1e-6)
    assert states[0, 1:, 1] == pytest.approx([-1.9, -1.8, -1.7], abs=1e-6)
    assert states[0, 1:, 2] == pytest.approx([0.51, 0.52, 0.53], abs=1e-6)


def test_learned_torch_rollout_applies_residual_to_response_limited_command():
    controller = make_controller()
    controls = np.zeros((1, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, :] = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)

    states, real_controls = controller._rollout_batch(state, controls, return_controls=True)

    assert states.shape == (1, controller.horizon_steps + 1, 6)
    assert real_controls.shape == (1, controller.horizon_steps, 3)
    assert real_controls[0, 0] == pytest.approx([0.7, 0.1, 0.15], abs=1e-6)
    assert states[0, 1, 3:] == pytest.approx([0.7, 0.1, 0.15], abs=1e-6)
    assert np.all(np.isfinite(states))


def test_learned_torch_rollout_scales_residual_with_gain():
    controller = make_controller(residual_gain=0.5)
    controls = np.zeros((1, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, :] = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)

    _states, real_controls = controller._rollout_batch(state, controls, return_controls=True)

    assert controller.residual_gain == pytest.approx(0.5)
    assert real_controls[0, 0] == pytest.approx([0.6, 0.15, 0.125], abs=1e-6)


def test_learned_torch_rollout_zero_gain_matches_response_limited_command():
    controller = make_controller(residual_gain=0.0)
    controls = np.zeros((1, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, :] = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)

    _states, real_controls = controller._rollout_batch(state, controls, return_controls=True)

    assert real_controls[0, 0] == pytest.approx([0.5, 0.2, 0.1], abs=1e-6)


def test_learned_torch_batch_cost_is_finite_and_shape_compatible():
    controller = make_controller()
    controls = np.zeros((controller.num_samples, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, 0] = np.linspace(0.0, 0.5, controller.num_samples)[:, None]
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    costs = controller.trajectory_cost_batch(state, controls, goal, obstacles)

    assert costs.shape == (controller.num_samples,)
    assert np.all(np.isfinite(costs))


def test_learned_torch_compute_control_returns_numpy_controller_outputs():
    controller = make_controller()
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


def test_learned_torch_profile_records_runtime_buckets():
    controller = make_controller(profile_enabled=True)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    controller.compute_control(
        state,
        [None, None, None, goal, np.empty((0, 7), dtype=np.float32), 0],
    )

    profile = controller.profile_summary()
    assert profile["enabled"] is True
    assert profile["total_calls"] == 1
    for bucket in (
        "sample_candidates_ms",
        "rollout_total_ms",
        "fdm_inference_ms",
        "cost_terms_ms",
        "cpu_transfer_ms",
    ):
        assert bucket in profile["totals_ms"]
        assert profile["totals_ms"][bucket] >= 0.0


def test_learned_torch_terrain_features_match_numpy_terrain_with_noise():
    config = load_config("config/b2_omni_oracle_random100_dataset.yaml")
    terrain = TerrainField.from_config(config["terrain"])
    controller = make_controller()
    controller.terrain = terrain
    controller._setup_terrain_tensors()
    states = torch.tensor(
        [
            [10.0, 12.0, 0.0, 0.1, 0.0, 0.0],
            [42.5, 50.0, 0.2, 0.3, -0.1, 0.1],
        ],
        dtype=torch.float32,
    )

    features_t, risks_t = controller._terrain_features_torch(states)

    expected_features = np.asarray([terrain.feature(float(s[0]), float(s[1])) for s in states], dtype=np.float32)
    expected_risks = np.asarray(
        [
            terrain.risk_cost(float(state[0]), float(state[1]), features=expected_features[idx])
            for idx, state in enumerate(states)
        ],
        dtype=np.float32,
    )
    assert features_t.detach().cpu().numpy() == pytest.approx(expected_features, abs=1e-6)
    assert risks_t.detach().cpu().numpy() == pytest.approx(expected_risks, abs=1e-6)


def test_learned_torch_terrain_features_match_numpy_terrain_with_patches():
    terrain = TerrainField(
        enabled=True,
        slope_scale=0.02,
        roughness_scale=0.05,
        friction_base=0.8,
        friction_slope_scale=0.05,
        friction_roughness_scale=0.05,
        patches=[
            {
                "name": "risk_band",
                "type": "band",
                "center": [10.0, 0.0],
                "angle": 90.0,
                "size": [8.0, 2.0],
                "edge_width": 0.5,
                "slope_f_delta": 0.06,
                "roughness_delta": 0.4,
                "friction_delta": -0.25,
            },
            {
                "name": "risk_island",
                "type": "ellipse",
                "center": [14.0, 1.5],
                "angle": 25.0,
                "size": [3.0, 1.5],
                "edge_width": 0.5,
                "roughness_delta": 0.3,
                "friction_delta": -0.2,
            },
        ],
    )
    controller = make_controller()
    controller.terrain = terrain
    controller._setup_terrain_tensors()
    states = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [14.0, 1.5, 0.2, 0.3, -0.1, 0.1],
            [18.0, 5.5, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    features_t, risks_t = controller._terrain_features_torch(states)

    expected_features = np.asarray([terrain.feature(float(s[0]), float(s[1])) for s in states], dtype=np.float32)
    expected_risks = np.asarray(
        [
            terrain.risk_cost(float(state[0]), float(state[1]), features=expected_features[idx])
            for idx, state in enumerate(states)
        ],
        dtype=np.float32,
    )
    assert features_t.detach().cpu().numpy() == pytest.approx(expected_features, abs=1e-6)
    assert risks_t.detach().cpu().numpy() == pytest.approx(expected_risks, abs=1e-6)


def test_learned_torch_terrain_risk_cost_matches_numpy_on_patch_map():
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
                "angle": 90.0,
                "size": [1.0, 1.0],
                "roughness_delta": 0.7,
                "friction_delta": -0.4,
            }
        ],
    )
    controller = make_controller()
    controller.terrain = terrain
    controller.terrain_risk_weight = 10.0
    controller.terrain_risk_threshold = 0.25
    controller.terrain_risk_power = 2.0
    controller.terrain_risk_mode = "excess"
    controller._setup_terrain_tensors()
    states = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.4, 0.0, 0.0, 0.0, 0.0],
                [0.1, 1.4, 0.0, 0.0, 0.0, 0.0],
                [0.2, 1.4, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )

    costs_t = controller._terrain_risk_cost_batch_torch(states)
    costs_np = controller._terrain_risk_cost_batch(states.detach().cpu().numpy())

    assert costs_t.detach().cpu().numpy() == pytest.approx(costs_np, abs=1e-5)
    assert costs_t[0].item() > costs_t[1].item()
