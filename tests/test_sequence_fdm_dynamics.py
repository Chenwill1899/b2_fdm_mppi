import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile

from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics


def _create_dummy_checkpoint(tmp_path: Path, horizon_steps: int = 5):
    model = SequenceFdmMlpV2(horizon_steps=horizon_steps, hidden_dims=[32])
    norm = {
        "state_mean": np.zeros(6, dtype=np.float32),
        "state_std": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(3, dtype=np.float32),
        "control_std": np.ones(3, dtype=np.float32),
        "target_mean": np.zeros(horizon_steps * 7, dtype=np.float32),
        "target_std": np.ones(horizon_steps * 7, dtype=np.float32),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "horizon_steps": horizon_steps,
        "hidden_dims": [32],
        "input_dim": 6 + 3 * horizon_steps + 81,
        "target_dim": horizon_steps * 7,
        "feature_names": [],
        "target_names": [],
    }
    ckpt_path = tmp_path / "best_model.pt"
    norm_path = tmp_path / "normalization.npz"
    torch.save(checkpoint, ckpt_path)
    np.savez(norm_path, **norm)
    return tmp_path


def test_from_artifacts(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=5)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")
    assert dyn.horizon_steps == 5
    assert dyn.model is not None


def test_predict_shapes(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=5)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")

    state = np.zeros(6, dtype=np.float32)
    controls = np.zeros((5, 3), dtype=np.float32)
    terrain = np.zeros(81, dtype=np.float32)

    states_pred, risk_logits = dyn.predict(state, controls, terrain)
    assert states_pred.shape == (5, 6)
    assert risk_logits.shape == (5,)
    assert np.isfinite(states_pred).all()
    assert np.isfinite(risk_logits).all()


def test_predict_batch(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=8)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")

    states = np.zeros((3, 6), dtype=np.float32)
    controls = np.zeros((3, 8, 3), dtype=np.float32)
    terrains = np.zeros((3, 81), dtype=np.float32)

    states_pred, risk_logits = dyn.predict_batch(states, controls, terrains)
    assert states_pred.shape == (3, 8, 6)
    assert risk_logits.shape == (3, 8)


def test_predict_torch(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=5)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")

    state = torch.zeros(6)
    controls = torch.zeros(5, 3)
    terrain = torch.zeros(81)

    states_pred, risk_logits = dyn.predict_torch(state, controls, terrain)
    assert states_pred.shape == (5, 6)
    assert risk_logits.shape == (5,)
