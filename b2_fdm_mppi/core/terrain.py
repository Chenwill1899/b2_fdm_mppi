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
        noise_enabled: bool = False,
        noise_seed: int = 123,
        noise_grid_size: tuple[int, int] = (32, 32),
        noise_scale: float = 0.35,
        noise_smooth_passes: int = 3,
        noise_roughness_weight: float = 0.25,
        noise_friction_weight: float = 0.18,
        noise_slope_weight: float = 0.08,
        noise_x_range: tuple[float, float] = (0.0, 100.0),
        noise_y_range: tuple[float, float] = (0.0, 100.0),
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
        self.noise_enabled = bool(noise_enabled)
        self.noise_seed = int(noise_seed)
        self.noise_grid_size = tuple(int(v) for v in noise_grid_size)
        self.noise_scale = float(noise_scale)
        self.noise_smooth_passes = int(noise_smooth_passes)
        self.noise_roughness_weight = float(noise_roughness_weight)
        self.noise_friction_weight = float(noise_friction_weight)
        self.noise_slope_weight = float(noise_slope_weight)
        self.noise_x_range = tuple(float(v) for v in noise_x_range)
        self.noise_y_range = tuple(float(v) for v in noise_y_range)
        self._noise_grid = self._build_noise_grid() if self.noise_enabled else None
        if self._noise_grid is not None:
            grad_y, grad_x = np.gradient(self._noise_grid)
            self._noise_grad_x = grad_x
            self._noise_grad_y = grad_y
        else:
            self._noise_grad_x = None
            self._noise_grad_y = None

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
            noise_enabled=bool(config.get("noise_enabled", False)),
            noise_seed=int(config.get("noise_seed", 123)),
            noise_grid_size=tuple(config.get("noise_grid_size", (32, 32))),
            noise_scale=float(config.get("noise_scale", 0.35)),
            noise_smooth_passes=int(config.get("noise_smooth_passes", 3)),
            noise_roughness_weight=float(config.get("noise_roughness_weight", 0.25)),
            noise_friction_weight=float(config.get("noise_friction_weight", 0.18)),
            noise_slope_weight=float(config.get("noise_slope_weight", 0.08)),
            noise_x_range=tuple(config.get("noise_x_range", (0.0, 100.0))),
            noise_y_range=tuple(config.get("noise_y_range", (0.0, 100.0))),
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
        if self._noise_grid is not None:
            noise, gradient_x, gradient_y = self._sample_noise(x, y)
            roughness += self.noise_roughness_weight * noise
            friction -= self.noise_friction_weight * noise
            slope_f += self.noise_slope_weight * gradient_x
            slope_l += self.noise_slope_weight * gradient_y
        roughness = float(np.clip(roughness, 0.0, 1.0))
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

    def _build_noise_grid(self) -> np.ndarray:
        rows = max(2, int(self.noise_grid_size[0]))
        cols = max(2, int(self.noise_grid_size[1]))
        rng = np.random.default_rng(self.noise_seed)
        grid = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float32)
        for _ in range(max(0, self.noise_smooth_passes)):
            padded = np.pad(grid, 1, mode="edge")
            grid = (
                padded[1:-1, 1:-1] * 4.0
                + padded[:-2, 1:-1] * 2.0
                + padded[2:, 1:-1] * 2.0
                + padded[1:-1, :-2] * 2.0
                + padded[1:-1, 2:] * 2.0
                + padded[:-2, :-2]
                + padded[:-2, 2:]
                + padded[2:, :-2]
                + padded[2:, 2:]
            ) / 16.0
        max_abs = float(np.max(np.abs(grid)))
        if max_abs > 1e-6:
            grid = grid / max_abs
        return (grid * self.noise_scale).astype(np.float32)

    def _sample_noise(self, x: float, y: float) -> tuple[float, float, float]:
        assert self._noise_grid is not None
        assert self._noise_grad_x is not None
        assert self._noise_grad_y is not None
        return (
            self._bilinear_sample(self._noise_grid, x, y),
            self._bilinear_sample(self._noise_grad_x, x, y),
            self._bilinear_sample(self._noise_grad_y, x, y),
        )

    def _bilinear_sample(self, grid: np.ndarray, x: float, y: float) -> float:
        x_min, x_max = self.noise_x_range
        y_min, y_max = self.noise_y_range
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)
        cols = grid.shape[1]
        rows = grid.shape[0]
        gx = np.clip((float(x) - x_min) / x_span, 0.0, 1.0) * (cols - 1)
        gy = np.clip((float(y) - y_min) / y_span, 0.0, 1.0) * (rows - 1)
        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, cols - 1)
        y1 = min(y0 + 1, rows - 1)
        tx = float(gx - x0)
        ty = float(gy - y0)
        top = (1.0 - tx) * grid[y0, x0] + tx * grid[y0, x1]
        bottom = (1.0 - tx) * grid[y1, x0] + tx * grid[y1, x1]
        return float((1.0 - ty) * top + ty * bottom)

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
