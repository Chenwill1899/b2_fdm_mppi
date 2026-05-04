import numpy as np
import pytest
import tempfile
from pathlib import Path

from b2_fdm_mppi.data.sequence_fdm_collector import build_sequence_fdm_windows


def test_build_windows_shapes():
    T = 50
    states = np.zeros((T, 6), dtype=np.float32)
    states[:, 0] = np.arange(T, dtype=np.float32)
    controls = np.ones((T, 3), dtype=np.float32)
    terrain_risk = np.zeros(T, dtype=np.float32)
    binary_risk = np.zeros(T, dtype=np.float32)
    terrain_seed = np.array([42], dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode_000000.npz"
        np.savez(path, states=states, cmd_controls=controls, terrain_risk=terrain_risk,
                 binary_risk=binary_risk, terrain_seed=terrain_seed)

        windows = build_sequence_fdm_windows(
            episode_path=path,
            horizon_steps=10,
            stride=5,
            map_bounds=(-20, 20, -20, 20),
        )
        assert len(windows) > 0
        w = windows[0]
        assert w["state"].shape == (6,)
        assert w["controls"].shape == (10, 3)
        assert w["terrain_grid"].shape == (81,)
        assert w["target_states"].shape == (10, 6)
        assert w["target_risk"].shape == (10,)


def test_build_windows_stride():
    T = 30
    states = np.zeros((T, 6), dtype=np.float32)
    controls = np.ones((T, 3), dtype=np.float32)
    terrain_risk = np.zeros(T, dtype=np.float32)
    binary_risk = np.zeros(T, dtype=np.float32)
    terrain_seed = np.array([1], dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode_000000.npz"
        np.savez(path, states=states, cmd_controls=controls, terrain_risk=terrain_risk,
                 binary_risk=binary_risk, terrain_seed=terrain_seed)

        windows_stride1 = build_sequence_fdm_windows(path, horizon_steps=5, stride=1, map_bounds=(-10, 10, -10, 10))
        windows_stride5 = build_sequence_fdm_windows(path, horizon_steps=5, stride=5, map_bounds=(-10, 10, -10, 10))
        assert len(windows_stride5) < len(windows_stride1)
