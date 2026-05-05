import numpy as np
import pytest
import tempfile
from pathlib import Path
import torch

from b2_fdm_mppi.training.sequence_fdm_v2 import SequenceFdmDataset, compute_normalization


def test_dataset_length_and_getitem():
    windows = []
    for i in range(20):
        windows.append({
            "state": np.zeros(6, dtype=np.float32),
            "controls": np.zeros((10, 3), dtype=np.float32),
            "terrain_grid": np.zeros(81, dtype=np.float32),
            "target_states": np.ones((10, 6), dtype=np.float32),
            "target_risk": np.zeros(10, dtype=np.float32),
        })
    dataset = SequenceFdmDataset(windows, horizon_steps=10)
    assert len(dataset) == 20
    x_state, x_ctrl, x_grid, y_state, y_risk = dataset[0]
    assert x_state.shape == (6,)
    assert x_ctrl.shape == (10, 3)
    assert x_grid.shape == (81,)
    assert y_state.shape == (10, 6)
    assert y_risk.shape == (10,)


def test_compute_normalization():
    windows = []
    for i in range(50):
        windows.append({
            "state": np.random.randn(6).astype(np.float32),
            "controls": np.random.randn(10, 3).astype(np.float32),
            "terrain_grid": np.random.rand(81).astype(np.float32),
            "target_states": np.random.randn(10, 6).astype(np.float32),
            "target_risk": np.random.randint(0, 2, size=10).astype(np.float32),
        })
    norm = compute_normalization(windows, horizon_steps=10)
    assert "state_mean" in norm
    assert "state_std" in norm
    assert "control_mean" in norm
    assert "control_std" in norm
    assert "target_mean" in norm
    assert "target_std" in norm
    assert norm["state_mean"].shape == (6,)
    assert norm["state_std"].shape == (6,)
    assert norm["control_mean"].shape == (3,)
    assert norm["control_std"].shape == (3,)
    assert norm["target_mean"].shape == (10 * 7,)
    assert norm["target_std"].shape == (10 * 7,)


def test_train_smoke():
    """Smoke test: training runs for a few epochs without crashing."""
    from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2
    from b2_fdm_mppi.training.sequence_fdm_v2 import train_sequence_fdm_v2

    windows = []
    for i in range(30):
        windows.append({
            "state": np.random.randn(6).astype(np.float32) * 0.1,
            "controls": np.random.randn(5, 3).astype(np.float32) * 0.1,
            "terrain_grid": np.random.rand(81).astype(np.float32),
            "target_states": np.random.randn(5, 6).astype(np.float32) * 0.1,
            "target_risk": np.random.randint(0, 2, size=5).astype(np.float32),
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        metrics = train_sequence_fdm_v2(
            windows=windows,
            output_dir=output_dir,
            hidden_dims=[32, 32],
            curriculum_phases=[(5, 3, 1e-3)],
            batch_size=8,
            use_tensorboard=False,
        )
        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "normalization.npz").exists()
        assert "best_val_loss" in metrics
