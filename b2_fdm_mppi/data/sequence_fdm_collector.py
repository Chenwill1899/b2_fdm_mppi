"""Collect a single sequence FDM episode on random terrain."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import yaml

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.data.collect_oracle_episode import collect_oracle_episode
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator


def _mark_binary_risk(terrain_risk: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Return binary array where once threshold exceeded, all subsequent are 1."""
    binary = np.zeros_like(terrain_risk, dtype=np.float32)
    if terrain_risk.size == 0:
        return binary
    idx = int(np.argmax(terrain_risk > threshold))
    if terrain_risk[idx] > threshold:
        binary[idx:] = 1.0
    return binary


def _sample_start_goal(
    rng: np.random.Generator,
    map_bounds: tuple[float, float, float, float],
    min_distance: float = 10.0,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample start and goal positions at least min_distance apart."""
    x_min, x_max, y_min, y_max = map_bounds
    for _ in range(max_attempts):
        start = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32
        )
        goal = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32
        )
        if np.linalg.norm(goal - start) >= min_distance:
            return start, goal
    raise RuntimeError(
        f"Could not sample start/goal pair within {max_attempts} attempts"
    )


def _terrain_to_config(terrain: TerrainField) -> dict:
    """Build a config dict from a TerrainField's public attributes."""
    return {
        "enabled": terrain.enabled,
        "slope_scale": terrain.slope_scale,
        "slope_wave": terrain.slope_wave,
        "roughness_scale": terrain.roughness_scale,
        "roughness_wave": terrain.roughness_wave,
        "friction_base": terrain.friction_base,
        "friction_slope_scale": terrain.friction_slope_scale,
        "friction_roughness_scale": terrain.friction_roughness_scale,
        "risk_weights": terrain.risk_weights,
        "goal_relief": terrain.goal_relief,
        "noise_enabled": terrain.noise_enabled,
        "noise_seed": terrain.noise_seed,
        "noise_grid_size": terrain.noise_grid_size,
        "noise_scale": terrain.noise_scale,
        "noise_smooth_passes": terrain.noise_smooth_passes,
        "noise_roughness_weight": terrain.noise_roughness_weight,
        "noise_friction_weight": terrain.noise_friction_weight,
        "noise_slope_weight": terrain.noise_slope_weight,
        "noise_x_range": terrain.noise_x_range,
        "noise_y_range": terrain.noise_y_range,
        "patches": terrain.patches,
    }


def _convert_tuples_to_lists(obj):
    """Recursively convert tuples to lists for YAML serialization."""
    if isinstance(obj, tuple):
        return [_convert_tuples_to_lists(v) for v in obj]
    if isinstance(obj, list):
        return [_convert_tuples_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _convert_tuples_to_lists(v) for k, v in obj.items()}
    return obj


def collect_sequence_fdm_episode(
    *,
    base_config_path: str | Path,
    episode_id: int,
    terrain_seed: int,
    output_dir: str | Path,
    map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    min_start_goal_distance: float = 10.0,
    risk_threshold: float = 0.6,
    num_patches_range: tuple[int, int] = (3, 6),
) -> dict:
    """Generate random terrain, run MPPI, and save an episode with binary risk labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{int(episode_id):06d}.npz"

    # Generate random terrain
    rng = np.random.default_rng(terrain_seed)
    terrain_gen = RandomTerrainGenerator(
        map_bounds=map_bounds,
        num_patches_range=num_patches_range,
    )
    terrain = terrain_gen.generate(seed=terrain_seed)

    # Sample start and goal
    start_xy, goal_xy = _sample_start_goal(
        rng, map_bounds, min_distance=min_start_goal_distance
    )

    # Load and override base config
    config = load_config(base_config_path)
    config["terrain"] = _terrain_to_config(terrain)
    config.setdefault("mppi", {})["backend"] = "torch"
    config["simulation"]["max_steps"] = 500
    # Disable expensive visualization to speed up collection
    config.setdefault("results", {})["enable_plots"] = False
    config.setdefault("results", {})["enable_animation"] = False
    # Enable terrain risk avoidance so MPPI avoids high-risk patches
    mppi_cfg = config.setdefault("mppi", {})
    mppi_cfg["terrain_risk_weight"] = 100.0
    mppi_cfg["terrain_risk_threshold"] = risk_threshold
    mppi_cfg["terrain_risk_mode"] = "excess"
    # Add random static obstacles
    rng_obs = np.random.default_rng(terrain_seed + 50000)
    num_obs = int(rng_obs.integers(3, 11))
    obs_list = []
    for _ in range(num_obs):
        ox = float(rng_obs.uniform(map_bounds[0] + 2.0, map_bounds[1] - 2.0))
        oy = float(rng_obs.uniform(map_bounds[2] + 2.0, map_bounds[3] - 2.0))
        radius = float(rng_obs.uniform(0.8, 2.0))
        obs_list.append([ox, oy, radius, 0.0, 0.0, 0.0, 0.0])
    config["obstacles"] = {
        "num_max": num_obs,
        "static_enabled": True,
        "virtual": obs_list,
    }

    initial_state = list(config["simulation"]["initial_state"])
    goal_state = list(config["simulation"]["goal"])
    initial_state[0] = float(start_xy[0])
    initial_state[1] = float(start_xy[1])
    goal_state[0] = float(goal_xy[0])
    goal_state[1] = float(goal_xy[1])
    config["simulation"]["initial_state"] = initial_state
    config["simulation"]["goal"] = goal_state

    # Write temporary config (convert tuples to lists for YAML compatibility)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        yaml.dump(_convert_tuples_to_lists(config), temp_file)
        temp_config_path = Path(temp_file.name)

    try:
        # Collect oracle episode
        metadata = collect_oracle_episode(
            config_path=temp_config_path,
            episode_id=episode_id,
            seed=terrain_seed,
            output_path=output_path,
        )

        # Load saved NPZ, augment, and re-save
        data = dict(np.load(output_path, allow_pickle=True))
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].dtype == object:
                data[key] = data[key].item()

        terrain_risk = data["terrain_risk"]
        binary_risk = _mark_binary_risk(terrain_risk, threshold=risk_threshold)
        data["binary_risk"] = binary_risk
        data["terrain_seed"] = np.asarray(int(terrain_seed), dtype=np.int64)
        data["start_xy"] = start_xy.astype(np.float32)
        data["goal_xy"] = goal_xy.astype(np.float32)

        np.savez_compressed(output_path, **data)

        metadata["binary_risk"] = binary_risk.tolist()
        metadata["terrain_seed"] = int(terrain_seed)
        metadata["start_xy"] = start_xy.tolist()
        metadata["goal_xy"] = goal_xy.tolist()
        metadata["output_path"] = str(output_path)

        return metadata
    finally:
        temp_config_path.unlink(missing_ok=True)
