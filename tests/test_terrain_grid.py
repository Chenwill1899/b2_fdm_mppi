import numpy as np
import pytest
import torch

from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.core.terrain_grid import sample_terrain_risk_grid_np, sample_terrain_risk_grid_torch


def test_grid_np_shape_and_bounds():
    terrain = TerrainField(enabled=True, noise_enabled=False)
    grid = sample_terrain_risk_grid_np(terrain, x=0.0, y=0.0, size=9, span=18.0)
    assert grid.shape == (81,)
    assert grid.dtype == np.float32
    assert np.all(grid >= 0.0)


def test_grid_np_centered():
    terrain = TerrainField(enabled=True, noise_enabled=False, patches=[
        {"type": "ellipse", "center": [5.0, 0.0], "angle": 0.0, "size": [2.0, 2.0],
         "edge_width": 0.5, "slope_f_delta": 1.0, "slope_l_delta": 0.0, "roughness_delta": 0.0, "friction_delta": 0.0}
    ])
    grid_center = sample_terrain_risk_grid_np(terrain, x=0.0, y=0.0, size=9, span=18.0)
    grid_shifted = sample_terrain_risk_grid_np(terrain, x=5.0, y=0.0, size=9, span=18.0)
    # The grid centered at (5,0) should have higher risk values near the patch center
    center_idx = (9 * 9) // 2
    assert grid_shifted[center_idx] > grid_center[center_idx]


def test_grid_torch_shape():
    terrain = TerrainField(enabled=True, noise_enabled=False)
    grid = sample_terrain_risk_grid_torch(terrain, x=0.0, y=0.0, size=9, span=18.0)
    assert grid.shape == (81,)
    assert grid.dtype == torch.float32
    assert torch.all(grid >= 0.0)


def test_grid_torch_matches_np():
    terrain = TerrainField(enabled=True, noise_enabled=True, noise_seed=42)
    grid_np = sample_terrain_risk_grid_np(terrain, x=3.0, y=-2.0, size=9, span=18.0)
    grid_torch = sample_terrain_risk_grid_torch(terrain, x=3.0, y=-2.0, size=9, span=18.0)
    assert np.allclose(grid_np, grid_torch.numpy(), atol=1e-5)
