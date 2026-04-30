"""Analytic terrain field for oracle residual simulations."""

from __future__ import annotations

import numpy as np


class TerrainField:
    """Deterministic analytic terrain model for residual-world simulation."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        slope_scale: float = 0.15,
        slope_wave: float = 0.6,
        roughness_scale: float = 0.4,
        roughness_wave: float = 0.45,
        friction_base: float = 0.8,
        friction_slope_scale: float = 0.3,
        friction_roughness_scale: float = 0.2,
        risk_weights: tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.8),
        goal_relief: dict | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.slope_scale = float(slope_scale)
        self.slope_wave = float(slope_wave)
        self.roughness_scale = float(roughness_scale)
        self.roughness_wave = float(roughness_wave)
        self.friction_base = float(friction_base)
        self.friction_slope_scale = float(friction_slope_scale)
        self.friction_roughness_scale = float(friction_roughness_scale)
        self.risk_weights = tuple(float(w) for w in risk_weights)
        self.goal_relief = goal_relief or {}

    @classmethod
    def from_config(cls, config: dict | None) -> "TerrainField":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", False)),
            slope_scale=float(config.get("slope_scale", 0.15)),
            slope_wave=float(config.get("slope_wave", 0.6)),
            roughness_scale=float(config.get("roughness_scale", 0.4)),
            roughness_wave=float(config.get("roughness_wave", 0.45)),
            friction_base=float(config.get("friction_base", 0.8)),
            friction_slope_scale=float(config.get("friction_slope_scale", 0.3)),
            friction_roughness_scale=float(config.get("friction_roughness_scale", 0.2)),
            risk_weights=tuple(config.get("risk_weights", (1.0, 1.0, 0.5, 0.8))),
            goal_relief=dict(config.get("goal_relief", {})),
        )

    def feature(self, x: float, y: float) -> np.ndarray:
        if not self.enabled:
            return np.zeros(4, dtype=np.float32)
        slope_f = self.slope_scale * (
            np.sin(self.slope_wave * x) + 0.35 * np.cos(0.5 * y)
        )
        slope_l = self.slope_scale * (
            np.cos(self.slope_wave * y) + 0.35 * np.sin(0.5 * x)
        )
        roughness = self.roughness_scale * (0.5 + 0.5 * np.sin(self.roughness_wave * (x + y)))
        friction = self.friction_base - self.friction_slope_scale * (
            0.5 * (abs(slope_f) + abs(slope_l))
        )
        friction -= self.friction_roughness_scale * roughness
        friction = float(np.clip(friction, 0.2, 1.0))
        attenuation = self._goal_relief_attenuation(x, y)
        slope_f *= attenuation
        slope_l *= attenuation
        roughness *= attenuation
        friction = 1.0 - attenuation * (1.0 - friction)
        return np.array([slope_f, slope_l, roughness, friction], dtype=np.float32)

    def risk_cost(self, x: float, y: float, *, features: np.ndarray | None = None) -> float:
        if not self.enabled:
            return 0.0
        if features is None:
            features = self.feature(x, y)
        slope_f, slope_l, roughness, friction = (float(val) for val in features)
        w0, w1, w2, w3 = self.risk_weights
        return float(w0 * abs(slope_f) + w1 * abs(slope_l) + w2 * roughness + w3 * (1.0 - friction))

    def _goal_relief_attenuation(self, x: float, y: float) -> float:
        relief = self.goal_relief
        if not bool(relief.get("enabled", False)):
            return 1.0
        center = relief.get("center", [0.0, 0.0])
        sigma = relief.get("sigma", [2.0, 1.2])
        sigma_x = max(1e-6, float(sigma[0]))
        sigma_y = max(1e-6, float(sigma[1]))
        strength = float(np.clip(relief.get("strength", 0.75), 0.0, 1.0))
        floor = float(np.clip(relief.get("floor", 0.25), 0.0, 1.0))
        dx = (x - float(center[0])) / sigma_x
        dy = (y - float(center[1])) / sigma_y
        gaussian = float(np.exp(-0.5 * (dx * dx + dy * dy)))
        return max(floor, 1.0 - strength * gaussian)
