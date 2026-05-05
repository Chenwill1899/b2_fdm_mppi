from pathlib import Path

import numpy as np
import pytest
import torch

from b2_fdm_mppi.core.learned_residual_dynamics import LearnedResidualDynamics, LearnedSequenceResidualDynamics
from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.residual_fdm_model import FEATURE_NAMES, TARGET_NAMES, ResidualFdmMlp
from b2_fdm_mppi.core.terrain import TerrainField


def write_fdm_artifacts(
    model_dir: Path,
    *,
    residual: tuple[float, float, float] = (0.2, -0.1, 0.05),
    input_dim: int | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_count = len(FEATURE_NAMES) if input_dim is None else int(input_dim)
    model = ResidualFdmMlp(input_dim=feature_count, hidden_dim=8)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": feature_count,
            "hidden_dim": 8,
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "epoch": 1,
            "val_loss": 0.0,
            "checkpoint_type": "best",
        },
        model_dir / "best_model.pt",
    )
    np.savez(
        model_dir / "normalization.npz",
        feature_mean=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        feature_std=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        target_mean=np.asarray(residual, dtype=np.float32),
        target_std=np.ones(len(TARGET_NAMES), dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
    )


def test_learned_residual_dynamics_predicts_and_steps_with_artifacts(tmp_path):
    write_fdm_artifacts(tmp_path)
    robot = OmniB2(dt=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.4)
    terrain = TerrainField(enabled=True)
    dynamics = LearnedResidualDynamics.from_artifacts(tmp_path, robot=robot, terrain=terrain, device="cpu")
    state = np.zeros(6, dtype=np.float32)
    command = np.array([0.9, 0.0, 0.1], dtype=np.float32)

    residual = dynamics.predict_residual(state, command)
    step = dynamics.step(state, command)

    assert residual.shape == (3,)
    assert residual == pytest.approx([0.2, -0.1, 0.05], abs=1e-6)
    assert step.predicted_residual == pytest.approx(residual, abs=1e-6)
    assert step.real_control == pytest.approx([1.0, -0.1, 0.15], abs=1e-6)
    assert step.next_state.shape == (6,)
    assert step.next_state[3:] == pytest.approx(step.real_control, abs=1e-6)
    assert np.all(np.isfinite(step.next_state))
    assert step.terrain_features.shape == (4,)
    assert np.isfinite(step.terrain_risk)
    assert dynamics.checkpoint_path == tmp_path / "best_model.pt"
    assert dynamics.normalization_path == tmp_path / "normalization.npz"


def test_learned_residual_dynamics_batch_prediction_is_finite(tmp_path):
    write_fdm_artifacts(tmp_path, residual=(0.1, 0.2, -0.05))
    robot = OmniB2(dt=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.4)
    terrain = TerrainField(enabled=True)
    dynamics = LearnedResidualDynamics.from_artifacts(tmp_path, robot=robot, terrain=terrain, device="cpu")
    states = np.zeros((3, 6), dtype=np.float32)
    states[:, 0] = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    commands = np.zeros((3, 3), dtype=np.float32)

    residuals = dynamics.predict_residual_batch(states, commands)

    assert residuals.shape == (3, 3)
    assert np.all(np.isfinite(residuals))
    assert residuals[:, 0] == pytest.approx([0.1, 0.1, 0.1], abs=1e-6)
    assert residuals[:, 1] == pytest.approx([0.2, 0.2, 0.2], abs=1e-6)
    assert residuals[:, 2] == pytest.approx([-0.05, -0.05, -0.05], abs=1e-6)


def test_learned_residual_dynamics_rejects_schema_mismatch(tmp_path):
    write_fdm_artifacts(tmp_path, input_dim=len(FEATURE_NAMES) - 1)
    robot = OmniB2(dt=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.4)
    terrain = TerrainField(enabled=True)

    with pytest.raises(ValueError, match="input_dim"):
        LearnedResidualDynamics.from_artifacts(tmp_path, robot=robot, terrain=terrain, device="cpu")


class CaptureSequenceModel(torch.nn.Module):
    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.captured: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.captured = features.detach().cpu()
        return torch.zeros((features.shape[0], self.horizon * 4), dtype=features.dtype, device=features.device)


def test_learned_sequence_dynamics_builds_interleaved_runtime_features(tmp_path):
    horizon = 2
    model = CaptureSequenceModel(horizon)
    dynamics = LearnedSequenceResidualDynamics(
        model=model,
        robot=OmniB2(dt=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.4),
        terrain=TerrainField(enabled=False),
        feature_mean=np.zeros(6 + 3 + horizon * 8, dtype=np.float32),
        feature_std=np.ones(6 + 3 + horizon * 8, dtype=np.float32),
        target_mean=np.zeros(horizon * 4, dtype=np.float32),
        target_std=np.ones(horizon * 4, dtype=np.float32),
        sequence_horizon=horizon,
        include_history_controls=True,
        history_steps=1,
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "best_model.pt",
        normalization_path=tmp_path / "normalization.npz",
    )
    states = torch.tensor([[1.0, 2.0, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)
    commands = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=torch.float32)
    history = torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32)
    terrain_features = torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float32)
    terrain_risk = torch.tensor([0.25], dtype=torch.float32)

    dynamics.predict_sequence_batch_torch(
        states=states,
        command_sequences=commands,
        history=history,
        terrain_features=terrain_features,
        terrain_risk=terrain_risk,
    )

    assert model.captured is not None
    expected = torch.tensor(
        [
            [
                1.0,
                2.0,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.1,
                0.2,
                0.3,
                0.01,
                0.02,
                0.03,
                0.04,
                0.25,
                0.4,
                0.5,
                0.6,
                0.01,
                0.02,
                0.03,
                0.04,
                0.25,
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(model.captured, expected, atol=1e-6)
