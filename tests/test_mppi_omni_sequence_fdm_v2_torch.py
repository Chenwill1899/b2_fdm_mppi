import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from b2_fdm_mppi.controllers.mppi_omni_sequence_fdm_v2_torch import MppiOmniSequenceFdmV2Torch
from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics
from b2_fdm_mppi.core.terrain import TerrainField


def _make_dummy_dynamics(tmp_path: Path, horizon_steps: int = 5):
    model = SequenceFdmMlpV2(horizon_steps=horizon_steps, hidden_dims=[32])
    norm = {
        "state_mean": np.zeros(6, dtype=np.float32),
        "state_std": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(3, dtype=np.float32),
        "control_std": np.ones(3, dtype=np.float32),
        "target_mean": np.zeros(horizon_steps * 7, dtype=np.float32),
        "target_std": np.ones(horizon_steps * 7, dtype=np.float32),
    }
    ckpt = {
        "model_state_dict": model.state_dict(),
        "horizon_steps": horizon_steps,
        "hidden_dims": [32],
        "input_dim": 6 + 3 * horizon_steps + 81,
        "target_dim": horizon_steps * 7,
    }
    torch.save(ckpt, tmp_path / "best_model.pt")
    np.savez(tmp_path / "normalization.npz", **norm)
    return SequenceFdmDynamics.from_artifacts(tmp_path, device="cpu")


def test_controller_instantiation():
    with tempfile.TemporaryDirectory() as tmpdir:
        dyn = _make_dummy_dynamics(Path(tmpdir), horizon_steps=5)
        terrain = TerrainField(enabled=True, noise_enabled=False)
        controller = MppiOmniSequenceFdmV2Torch(
            dt=0.1,
            horizon_steps=5,
            num_samples=16,
            lambda_=0.7,
            noise_std=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            max_vx=1.5,
            max_vy=0.5,
            max_wz=1.0,
            sequence_dynamics=dyn,
            terrain=terrain,
            device="cpu",
            fdm_risk_weight=5.0,
        )
        assert controller.horizon_steps == 5
        assert controller.fdm_risk_weight == 5.0


def test_controller_cost_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        dyn = _make_dummy_dynamics(Path(tmpdir), horizon_steps=5)
        terrain = TerrainField(enabled=True, noise_enabled=False)
        controller = MppiOmniSequenceFdmV2Torch(
            dt=0.1,
            horizon_steps=5,
            num_samples=8,
            lambda_=0.7,
            noise_std=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            max_vx=1.5,
            max_vy=0.5,
            max_wz=1.0,
            sequence_dynamics=dyn,
            terrain=terrain,
            device="cpu",
            fdm_risk_weight=5.0,
        )
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        controls = torch.randn(8, 5, 3, dtype=torch.float32)
        goal = torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        obstacles = torch.zeros(0, 3, dtype=torch.float32)

        costs = controller._trajectory_cost_batch_torch(initial_state, controls, goal, obstacles)
        assert costs.shape == (8,)
        assert torch.isfinite(costs).all()
