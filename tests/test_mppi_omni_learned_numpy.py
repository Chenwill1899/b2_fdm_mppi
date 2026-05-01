import numpy as np
import pytest

from b2_fdm_mppi.controllers.mppi_omni_learned_numpy import LearnedFdmMppiOmniNumpy


class ConstantResidualDynamics:
    checkpoint_path = "stub/best_model.pt"
    normalization_path = "stub/normalization.npz"
    device = "cpu"

    def predict_residual_batch(self, states, commands, terrain_features=None, terrain_risk=None):
        states = np.asarray(states, dtype=np.float32)
        residuals = np.zeros((states.shape[0], 3), dtype=np.float32)
        residuals[:, 0] = 0.2
        residuals[:, 1] = -0.1
        residuals[:, 2] = 0.05
        return residuals


def make_learned_controller():
    return LearnedFdmMppiOmniNumpy(
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
        learned_dynamics=ConstantResidualDynamics(),
    )


def test_learned_numpy_rollout_applies_residual_to_response_limited_command():
    controller = make_learned_controller()
    controls = np.zeros((1, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, :] = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)

    states, real_controls = controller._rollout_batch(state, controls, return_controls=True)

    assert states.shape == (1, controller.horizon_steps + 1, 6)
    assert real_controls.shape == (1, controller.horizon_steps, 3)
    assert real_controls[0, 0] == pytest.approx([0.7, 0.1, 0.15], abs=1e-6)
    assert states[0, 1, 3:] == pytest.approx([0.7, 0.1, 0.15], abs=1e-6)
    assert np.all(np.isfinite(states))


def test_learned_numpy_batch_cost_is_finite_and_shape_compatible():
    controller = make_learned_controller()
    controls = np.zeros((controller.num_samples, controller.horizon_steps, 3), dtype=np.float32)
    controls[:, :, 0] = np.linspace(0.0, 0.5, controller.num_samples)[:, None]
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.empty((0, 7), dtype=np.float32)

    costs = controller.trajectory_cost_batch(state, controls, goal, obstacles)

    assert costs.shape == (controller.num_samples,)
    assert np.all(np.isfinite(costs))
