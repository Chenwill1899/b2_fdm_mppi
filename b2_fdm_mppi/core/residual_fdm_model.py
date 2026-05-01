"""Shared residual FDM model definition and feature helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = [
    "state_x",
    "state_y",
    "state_theta",
    "state_vx",
    "state_vy",
    "state_wz",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "terrain_slope_f",
    "terrain_slope_l",
    "terrain_roughness",
    "terrain_friction",
    "terrain_risk",
]
TARGET_NAMES = ["exec_du_vx", "exec_du_vy", "exec_du_wz"]
TARGET_AXES = ("vx", "vy", "wz")


class ResidualFdmMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(TARGET_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_feature_vector(
    state: np.ndarray,
    command: np.ndarray,
    terrain_features: np.ndarray,
    terrain_risk: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float32).reshape(6),
            np.asarray(command, dtype=np.float32).reshape(3),
            np.asarray(terrain_features, dtype=np.float32).reshape(4),
            np.asarray([terrain_risk], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
