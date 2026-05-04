"""Terrain risk grid sampling utilities for sequence FDM."""

from __future__ import annotations

import numpy as np
import torch

from b2_fdm_mppi.core.terrain import TerrainField


def sample_terrain_risk_grid_np(
    terrain: TerrainField,
    x: float,
    y: float,
    size: int = 9,
    span: float = 18.0,
) -> np.ndarray:
    """Sample a flat risk grid centered on (x, y) using numpy."""
    half = span / 2.0
    xs = np.linspace(x - half, x + half, size, dtype=np.float32)
    ys = np.linspace(y - half, y + half, size, dtype=np.float32)
    grid = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = float(terrain.risk_cost(float(xs[i]), float(ys[j])))
    return grid.reshape(-1)


def sample_terrain_risk_grid_torch(
    terrain: TerrainField,
    x: float,
    y: float,
    size: int = 9,
    span: float = 18.0,
) -> torch.Tensor:
    """Sample a flat risk grid centered on (x, y) using torch."""
    half = span / 2.0
    xs = torch.linspace(x - half, x + half, size, dtype=torch.float32)
    ys = torch.linspace(y - half, y + half, size, dtype=torch.float32)
    grid = torch.zeros((size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = float(terrain.risk_cost(float(xs[i].item()), float(ys[j].item())))
    return grid.reshape(-1)
