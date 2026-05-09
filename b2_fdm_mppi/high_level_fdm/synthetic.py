"""Numpy-only synthetic dataset generator for the High-Level FDM.

This generator exists so the training pipeline can be exercised without
MuJoCo or any ROS topic stream. It reuses the existing nominal and
residual models:

- `b2_fdm_mppi.core.omni_b2.OmniB2`         nominal SE(2) integrator
- `b2_fdm_mppi.core.residual_world.ResidualWorld` oracle residuals
- `b2_fdm_mppi.core.terrain.TerrainField`   traversability field

and adds:

- a simple circular-disk obstacle sampler,
- body-frame local map rasterization (occupancy + traversability +
  height mean/std mock),
- per-step risk labeling consistent with the schema's four channels.

The output of `generate_sample` matches the layout in
`high_level_fdm.schema.HighLevelFdmSample`. `generate_dataset` emits
train/val/test npz shards plus a `manifest.json` that the trainer loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.residual_world import ResidualWorld
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.high_level_fdm.schema import (
    CONTROL_DIM,
    DEFAULT_MAP_CHANNELS,
    HighLevelFdmSample,
    HighLevelFdmSchema,
    NUM_RISK_CHANNELS,
    RISK_CHANNELS,
    STATE_DIM,
    pose_delta_to_param,
    wrap_angle,
    world_to_body_xy,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ObstacleConfig:
    num_range: tuple[int, int] = (2, 6)
    radius_range: tuple[float, float] = (0.25, 0.55)
    spawn_radius_range: tuple[float, float] = (0.6, 3.5)
    min_obstacle_gap: float = 0.2


@dataclass
class SyntheticConfig:
    # Control-loop dt (seconds). Matches the residual FDM pipeline default.
    dt: float = 0.1
    # Max velocities (m/s, rad/s). Same meaning as OmniB2.
    max_vx: float = 0.8
    max_vy: float = 0.4
    max_wz: float = 1.2
    # Robot disk radius used for collision/swept-distance labels.
    robot_radius: float = 0.3
    # History and horizon lengths.
    history_length: int = 8
    horizon: int = 16
    # Map patch (body frame, meters per cell).
    map_channels: tuple[str, ...] = DEFAULT_MAP_CHANNELS
    map_height: int = 48
    map_width: int = 48
    map_resolution: float = 0.1
    # Command sampling
    cmd_low_pass_tau: float = 0.35
    cmd_target_change_prob: float = 0.08
    # Risk thresholds
    high_cost_threshold: float = 0.35
    stuck_velocity_ratio: float = 0.25  # |u_real - u_cmd| / |u_cmd| threshold
    stuck_speed_floor: float = 0.05
    out_of_map_margin: float = 0.05  # meters outside patch counts as untraversable
    # Obstacles
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    # Terrain enable (ResidualWorld is only meaningful when terrain is on)
    terrain_noise_enabled: bool = True
    residual_world_enabled: bool = True


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    obstacles: np.ndarray  # (M, 3) [x, y, radius]
    terrain: TerrainField
    residual_world: ResidualWorld
    initial_state: np.ndarray  # (6,)
    history: np.ndarray        # (H, 6) world-frame states preceding t
    history_controls: np.ndarray  # (H, 3)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HighLevelFdmSyntheticGenerator:
    """Deterministic synthetic generator, seed-indexed per sample."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self._robot = OmniB2(
            dt=self.config.dt,
            max_vx=self.config.max_vx,
            max_vy=self.config.max_vy,
            max_wz=self.config.max_wz,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schema(self) -> HighLevelFdmSchema:
        cfg = self.config
        return HighLevelFdmSchema(
            history_length=int(cfg.history_length),
            horizon=int(cfg.horizon),
            map_channels=int(len(cfg.map_channels)),
            map_height=int(cfg.map_height),
            map_width=int(cfg.map_width),
            map_resolution=float(cfg.map_resolution),
            risk_channels=RISK_CHANNELS,
        )

    def generate_sample(self, seed: int) -> HighLevelFdmSample:
        rng = np.random.default_rng(int(seed))
        scenario = self._build_scenario(rng)
        control_sequence = self._sample_control_sequence(rng, scenario.history_controls[-1])
        traj, u_real, collisions, high_cost, stuck, out_of_map = self._rollout_ground_truth(
            scenario, control_sequence
        )
        pose_target = self._build_pose_target(scenario.initial_state, traj)
        map_patch = self._rasterize_map_patch(scenario)
        history_body = self._history_body_frame(scenario)
        risk_target, risk_mask = self._build_risk_labels(
            collisions=collisions,
            high_cost=high_cost,
            stuck=stuck,
            out_of_map=out_of_map,
            u_real=u_real,
            u_cmd=control_sequence,
        )
        return HighLevelFdmSample(
            state_history=history_body.astype(np.float32),
            map_patch=map_patch.astype(np.float32),
            control_sequence=control_sequence.astype(np.float32),
            pose_target=pose_target.astype(np.float32),
            risk_target=risk_target.astype(np.float32),
            risk_mask=risk_mask.astype(np.float32),
            meta={
                "seed": int(seed),
                "num_obstacles": int(scenario.obstacles.shape[0]),
            },
        )

    def iter_samples(self, seeds: Iterable[int]) -> Iterable[HighLevelFdmSample]:
        for seed in seeds:
            yield self.generate_sample(int(seed))

    # ------------------------------------------------------------------
    # Scenario building
    # ------------------------------------------------------------------

    def _build_scenario(self, rng: np.random.Generator) -> Scenario:
        cfg = self.config
        terrain = self._make_terrain(rng)
        residual_world = ResidualWorld(
            model=OmniB2(
                dt=cfg.dt,
                max_vx=cfg.max_vx,
                max_vy=cfg.max_vy,
                max_wz=cfg.max_wz,
            ),
            terrain=terrain,
            enabled=bool(cfg.residual_world_enabled),
            alpha=float(rng.uniform(0.2, 0.5)),
            residual_scale=float(rng.uniform(0.35, 0.7)),
            noise_std=float(rng.uniform(0.01, 0.03)),
            max_residual_ratio=0.4,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        initial_state = np.zeros(STATE_DIM, dtype=np.float32)
        initial_state[2] = float(rng.uniform(-np.pi, np.pi))
        obstacles = self._sample_obstacles(rng, initial_state[:2])
        history, history_controls = self._sample_history(rng, initial_state, residual_world)
        return Scenario(
            obstacles=obstacles,
            terrain=terrain,
            residual_world=residual_world,
            initial_state=initial_state,
            history=history,
            history_controls=history_controls,
        )

    def _make_terrain(self, rng: np.random.Generator) -> TerrainField:
        return TerrainField(
            enabled=True,
            slope_scale=float(rng.uniform(0.05, 0.2)),
            slope_wave=float(rng.uniform(0.4, 0.9)),
            roughness_scale=float(rng.uniform(0.2, 0.5)),
            roughness_wave=float(rng.uniform(0.3, 0.8)),
            friction_base=float(rng.uniform(0.55, 0.85)),
            friction_slope_scale=0.3,
            friction_roughness_scale=0.2,
            risk_weights=(1.0, 1.0, 0.5, 0.8),
            noise_enabled=bool(self.config.terrain_noise_enabled),
            noise_seed=int(rng.integers(0, 2**31 - 1)),
            noise_grid_size=(16, 16),
            noise_scale=float(rng.uniform(0.2, 0.4)),
            noise_roughness_weight=0.25,
            noise_friction_weight=0.18,
            noise_slope_weight=0.08,
            noise_x_range=(-10.0, 10.0),
            noise_y_range=(-10.0, 10.0),
        )

    def _sample_obstacles(
        self,
        rng: np.random.Generator,
        origin_xy: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config.obstacles
        num = int(rng.integers(cfg.num_range[0], cfg.num_range[1] + 1))
        if num <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        obstacles: list[tuple[float, float, float]] = []
        attempts = 0
        while len(obstacles) < num and attempts < num * 20:
            attempts += 1
            radius = float(rng.uniform(cfg.radius_range[0], cfg.radius_range[1]))
            dist = float(rng.uniform(cfg.spawn_radius_range[0], cfg.spawn_radius_range[1]))
            angle = float(rng.uniform(-np.pi, np.pi))
            x = float(origin_xy[0]) + dist * np.cos(angle)
            y = float(origin_xy[1]) + dist * np.sin(angle)
            ok = True
            for ox, oy, orad in obstacles:
                if np.hypot(x - ox, y - oy) < radius + orad + cfg.min_obstacle_gap:
                    ok = False
                    break
            if ok:
                obstacles.append((x, y, radius))
        return np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)

    def _sample_history(
        self,
        rng: np.random.Generator,
        initial_state: np.ndarray,
        residual_world: ResidualWorld,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        # Roll *backward* from the current state by stepping forward from a
        # past state. Easier to just sample some commands forward starting
        # from a random past pose and let the last state be the "current".
        history_states = np.zeros((cfg.history_length, STATE_DIM), dtype=np.float32)
        history_controls = np.zeros((cfg.history_length, CONTROL_DIM), dtype=np.float32)

        past_state = initial_state.copy()
        past_state[0] -= float(rng.uniform(0.0, 0.5))
        past_state[1] -= float(rng.uniform(-0.25, 0.25))
        past_state[2] = float(rng.uniform(-np.pi, np.pi))

        cmd = self._random_command(rng)
        target = cmd.copy()
        residual_world.reset()
        for step in range(cfg.history_length):
            if rng.random() < cfg.cmd_target_change_prob:
                target = self._random_command(rng)
            cmd = self._smooth_command(cmd, target)
            history_controls[step] = cmd
            past_state, _u_real, _delta, _terrain_features = residual_world.update_state(
                past_state, cmd
            )
            history_states[step] = past_state
        # Use the very last state as "current" to guarantee history ends at
        # state t. We then overwrite initial_state in the caller to this
        # final state so the history and current frame are consistent.
        initial_state[:] = history_states[-1]
        return history_states, history_controls

    # ------------------------------------------------------------------
    # Control sampling
    # ------------------------------------------------------------------

    def _sample_control_sequence(
        self,
        rng: np.random.Generator,
        last_cmd: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        horizon = int(cfg.horizon)
        controls = np.zeros((horizon, CONTROL_DIM), dtype=np.float32)
        cmd = np.asarray(last_cmd, dtype=np.float32).copy()
        target = self._random_command(rng)
        for step in range(horizon):
            if rng.random() < cfg.cmd_target_change_prob:
                target = self._random_command(rng)
            cmd = self._smooth_command(cmd, target)
            controls[step] = cmd
        return controls

    def _random_command(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        return np.array(
            [
                rng.uniform(-cfg.max_vx, cfg.max_vx),
                rng.uniform(-cfg.max_vy, cfg.max_vy),
                rng.uniform(-cfg.max_wz, cfg.max_wz),
            ],
            dtype=np.float32,
        )

    def _smooth_command(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        cfg = self.config
        alpha = float(np.exp(-float(cfg.dt) / max(1e-3, float(cfg.cmd_low_pass_tau))))
        return alpha * np.asarray(current, dtype=np.float32) + (1.0 - alpha) * np.asarray(
            target, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Ground-truth rollout and labels
    # ------------------------------------------------------------------

    def _rollout_ground_truth(
        self,
        scenario: Scenario,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        horizon = int(controls.shape[0])
        traj = np.zeros((horizon, STATE_DIM), dtype=np.float32)
        u_real = np.zeros((horizon, CONTROL_DIM), dtype=np.float32)
        collisions = np.zeros(horizon, dtype=np.float32)
        high_cost = np.zeros(horizon, dtype=np.float32)
        stuck = np.zeros(horizon, dtype=np.float32)
        out_of_map = np.zeros(horizon, dtype=np.float32)

        state = scenario.initial_state.copy()
        scenario.residual_world.reset()
        for step, u in enumerate(controls):
            next_state, u_r, _delta, terrain_features = scenario.residual_world.update_state(
                state, u
            )
            traj[step] = next_state
            u_real[step] = u_r
            # Risk labels
            if self._collides(next_state[:2], scenario.obstacles):
                collisions[step] = 1.0
            risk = scenario.terrain.risk_cost(
                float(next_state[0]), float(next_state[1]), features=terrain_features
            )
            if risk >= float(self.config.high_cost_threshold):
                high_cost[step] = 1.0
            if self._is_stuck(u, u_r):
                stuck[step] = 1.0
            if self._out_of_map(next_state[:2], scenario.initial_state):
                out_of_map[step] = 1.0
            state = next_state
        return traj, u_real, collisions, high_cost, stuck, out_of_map

    def _collides(self, xy: np.ndarray, obstacles: np.ndarray) -> bool:
        if obstacles.size == 0:
            return False
        dx = obstacles[:, 0] - float(xy[0])
        dy = obstacles[:, 1] - float(xy[1])
        dist = np.sqrt(dx * dx + dy * dy)
        clearance = obstacles[:, 2] + float(self.config.robot_radius)
        return bool(np.any(dist <= clearance))

    def _is_stuck(self, u_cmd: np.ndarray, u_real: np.ndarray) -> bool:
        cmd_mag = float(np.linalg.norm(u_cmd[:2]))
        if cmd_mag < float(self.config.stuck_speed_floor):
            # Low-speed segments are not labeled as stuck; it's intentional.
            return False
        diff = float(np.linalg.norm(u_cmd[:2] - u_real[:2]))
        return diff / (cmd_mag + 1e-6) >= float(self.config.stuck_velocity_ratio)

    def _out_of_map(self, xy: np.ndarray, origin_state: np.ndarray) -> bool:
        cfg = self.config
        half_w = 0.5 * cfg.map_width * cfg.map_resolution
        half_h = 0.5 * cfg.map_height * cfg.map_resolution
        body = world_to_body_xy(
            np.asarray(xy, dtype=np.float32).reshape(1, 2),
            np.asarray(origin_state[:2], dtype=np.float32),
            float(origin_state[2]),
        )[0]
        return bool(
            abs(float(body[0])) > half_w + cfg.out_of_map_margin
            or abs(float(body[1])) > half_h + cfg.out_of_map_margin
        )

    # ------------------------------------------------------------------
    # Tensor packing
    # ------------------------------------------------------------------

    def _build_pose_target(
        self,
        initial_state: np.ndarray,
        traj: np.ndarray,
    ) -> np.ndarray:
        origin_xy = initial_state[:2]
        origin_theta = float(initial_state[2])
        xy_body = world_to_body_xy(traj[:, :2], origin_xy, origin_theta)
        dtheta = wrap_angle(traj[:, 2] - origin_theta)
        pose_delta = np.concatenate([xy_body, dtheta[:, None]], axis=1).astype(np.float32)
        return pose_delta_to_param(pose_delta)

    def _history_body_frame(self, scenario: Scenario) -> np.ndarray:
        cfg = self.config
        history = np.asarray(scenario.history, dtype=np.float32)
        out = np.zeros((cfg.history_length, STATE_DIM), dtype=np.float32)
        origin_xy = scenario.initial_state[:2]
        origin_theta = float(scenario.initial_state[2])
        xy_body = world_to_body_xy(history[:, :2], origin_xy, origin_theta)
        theta_body = wrap_angle(history[:, 2] - origin_theta)
        out[:, 0] = xy_body[:, 0]
        out[:, 1] = xy_body[:, 1]
        out[:, 2] = theta_body
        # Velocities are body-frame already in the OmniB2 convention.
        out[:, 3:6] = history[:, 3:6]
        return out

    def _rasterize_map_patch(self, scenario: Scenario) -> np.ndarray:
        cfg = self.config
        channels = len(cfg.map_channels)
        patch = np.zeros((channels, cfg.map_height, cfg.map_width), dtype=np.float32)
        origin_xy = scenario.initial_state[:2]
        origin_theta = float(scenario.initial_state[2])

        # Build a (Gh, Gw, 2) grid of body-frame cell centers and then rotate
        # into the world frame once.
        gh = int(cfg.map_height)
        gw = int(cfg.map_width)
        res = float(cfg.map_resolution)
        half_w = 0.5 * gw * res
        half_h = 0.5 * gh * res
        # Body-frame x increases forward (rows go from front to back).
        body_x = np.linspace(half_w - 0.5 * res, -half_w + 0.5 * res, gh, dtype=np.float32)
        body_y = np.linspace(-half_h + 0.5 * res, half_h - 0.5 * res, gw, dtype=np.float32)
        bx, by = np.meshgrid(body_x, body_y, indexing="ij")
        cos_t = float(np.cos(origin_theta))
        sin_t = float(np.sin(origin_theta))
        wx = origin_xy[0] + cos_t * bx - sin_t * by
        wy = origin_xy[1] + sin_t * bx + cos_t * by

        channel_index = {name: idx for idx, name in enumerate(cfg.map_channels)}

        # Occupancy from obstacles
        if "occupancy" in channel_index and scenario.obstacles.size > 0:
            occ = np.zeros((gh, gw), dtype=np.float32)
            obs = scenario.obstacles
            for k in range(obs.shape[0]):
                ox = float(obs[k, 0])
                oy = float(obs[k, 1])
                orad = float(obs[k, 2])
                mask = ((wx - ox) ** 2 + (wy - oy) ** 2) <= (orad ** 2)
                occ[mask] = 1.0
            patch[channel_index["occupancy"]] = occ

        # Terrain-driven channels. We sample the terrain field at each cell
        # center. This is O(Gh * Gw) per sample; tiny patch sizes keep this
        # fine on CPU.
        if any(
            name in channel_index
            for name in ("traversability_risk", "height_mean", "height_std")
        ):
            risk = np.zeros((gh, gw), dtype=np.float32)
            height_mean = np.zeros((gh, gw), dtype=np.float32)
            height_std = np.zeros((gh, gw), dtype=np.float32)
            for i in range(gh):
                for j in range(gw):
                    feats = scenario.terrain.feature(float(wx[i, j]), float(wy[i, j]))
                    risk[i, j] = scenario.terrain.risk_cost(
                        float(wx[i, j]), float(wy[i, j]), features=feats
                    )
                    # Surrogate height statistics from terrain features. The
                    # values are purely synthetic placeholders; real data
                    # will provide true elevation statistics.
                    height_mean[i, j] = float(feats[0])  # slope_f as proxy
                    height_std[i, j] = float(feats[2])   # roughness as proxy
            if "traversability_risk" in channel_index:
                patch[channel_index["traversability_risk"]] = risk
            if "height_mean" in channel_index:
                patch[channel_index["height_mean"]] = height_mean
            if "height_std" in channel_index:
                patch[channel_index["height_std"]] = height_std
        return patch

    def _build_risk_labels(
        self,
        *,
        collisions: np.ndarray,
        high_cost: np.ndarray,
        stuck: np.ndarray,
        out_of_map: np.ndarray,
        u_real: np.ndarray,
        u_cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        horizon = int(collisions.shape[0])
        risk = np.zeros((horizon, NUM_RISK_CHANNELS), dtype=np.float32)
        mask = np.ones((horizon, NUM_RISK_CHANNELS), dtype=np.float32)
        risk[:, 0] = collisions
        risk[:, 1] = high_cost
        risk[:, 2] = stuck
        risk[:, 3] = np.maximum(collisions, out_of_map)
        # Mask out "stuck" when the command itself is near zero (no signal).
        cmd_speed = np.linalg.norm(u_cmd[:, :2], axis=1)
        mask[:, 2] = (cmd_speed >= float(self.config.stuck_speed_floor)).astype(np.float32)
        # Once collision occurs, later steps are undefined; we still keep a
        # mask of 1 for now so the model learns that collisions persist.
        return risk, mask


# ---------------------------------------------------------------------------
# Dataset writing (raw shards, dataset.py does the final npz)
# ---------------------------------------------------------------------------

def save_sample(sample: HighLevelFdmSample, path: Path) -> None:
    np.savez_compressed(
        str(path),
        state_history=sample.state_history,
        map_patch=sample.map_patch,
        control_sequence=sample.control_sequence,
        pose_target=sample.pose_target,
        risk_target=sample.risk_target,
        risk_mask=sample.risk_mask,
        meta=json.dumps(sample.meta),
    )


def synthetic_config_to_dict(cfg: SyntheticConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    # dataclasses.asdict turns ObstacleConfig into a dict already; just ensure
    # tuples survive round-trips when dumped to JSON.
    payload["map_channels"] = list(cfg.map_channels)
    return payload


def synthetic_config_from_dict(payload: dict[str, Any]) -> SyntheticConfig:
    obstacles_payload = dict(payload.get("obstacles", {}))
    obstacles = ObstacleConfig(
        num_range=tuple(obstacles_payload.get("num_range", (2, 6))),
        radius_range=tuple(obstacles_payload.get("radius_range", (0.25, 0.55))),
        spawn_radius_range=tuple(obstacles_payload.get("spawn_radius_range", (0.6, 3.5))),
        min_obstacle_gap=float(obstacles_payload.get("min_obstacle_gap", 0.2)),
    )
    cfg = SyntheticConfig(
        dt=float(payload.get("dt", 0.1)),
        max_vx=float(payload.get("max_vx", 0.8)),
        max_vy=float(payload.get("max_vy", 0.4)),
        max_wz=float(payload.get("max_wz", 1.2)),
        robot_radius=float(payload.get("robot_radius", 0.3)),
        history_length=int(payload.get("history_length", 8)),
        horizon=int(payload.get("horizon", 16)),
        map_channels=tuple(payload.get("map_channels", DEFAULT_MAP_CHANNELS)),
        map_height=int(payload.get("map_height", 48)),
        map_width=int(payload.get("map_width", 48)),
        map_resolution=float(payload.get("map_resolution", 0.1)),
        cmd_low_pass_tau=float(payload.get("cmd_low_pass_tau", 0.35)),
        cmd_target_change_prob=float(payload.get("cmd_target_change_prob", 0.08)),
        high_cost_threshold=float(payload.get("high_cost_threshold", 0.35)),
        stuck_velocity_ratio=float(payload.get("stuck_velocity_ratio", 0.25)),
        stuck_speed_floor=float(payload.get("stuck_speed_floor", 0.05)),
        out_of_map_margin=float(payload.get("out_of_map_margin", 0.05)),
        obstacles=obstacles,
        terrain_noise_enabled=bool(payload.get("terrain_noise_enabled", True)),
        residual_world_enabled=bool(payload.get("residual_world_enabled", True)),
    )
    return cfg
