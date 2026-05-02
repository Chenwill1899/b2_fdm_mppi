#!/usr/bin/env python3
"""Inspect configured terrain patch maps for Stage 5-E0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.visualization.utils import map_axis_limits_from_config


def inspect_configs(
    *,
    config_paths: Sequence[Path],
    output_dir: Path,
    grid_resolution: int = 100,
    path_samples: int = 200,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = [
        inspect_config(
            config_path=Path(config_path),
            output_dir=output_dir / Path(config_path).stem,
            grid_resolution=grid_resolution,
            path_samples=path_samples,
        )
        for config_path in config_paths
    ]
    summary = {
        "output_dir": str(output_dir),
        "grid_resolution": int(grid_resolution),
        "path_samples": int(path_samples),
        "maps": maps,
    }
    (output_dir / "terrain_map_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def inspect_config(
    *,
    config_path: Path,
    output_dir: Path,
    grid_resolution: int,
    path_samples: int,
) -> dict:
    config = load_config(config_path)
    terrain = TerrainField.from_config(config.get("terrain"))
    xlim, ylim = map_axis_limits_from_config(config)
    grid = terrain_grid(terrain, xlim, ylim, grid_resolution)
    start = np.asarray(config["simulation"]["initial_state"][:2], dtype=np.float32)
    goal = np.asarray(config["simulation"]["goal"][:2], dtype=np.float32)
    path_summary = summarize_path_risks(
        terrain=terrain,
        start=start,
        goal=goal,
        xlim=xlim,
        ylim=ylim,
        path_samples=path_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_png = output_dir / "terrain_risk_map.png"
    feature_png = output_dir / "terrain_feature_maps.png"
    plot_risk_map(
        grid=grid,
        xlim=xlim,
        ylim=ylim,
        start=start,
        goal=goal,
        best_detour=np.asarray(path_summary["best_detour_path"], dtype=np.float32),
        output_path=risk_png,
        title=config_path.stem,
    )
    plot_feature_maps(grid=grid, xlim=xlim, ylim=ylim, output_path=feature_png, title=config_path.stem)
    summary = {
        "scenario": config_path.stem,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "terrain_risk_map": str(risk_png),
        "terrain_feature_maps": str(feature_png),
        "risk_min": float(np.min(grid["risk"])),
        "risk_max": float(np.max(grid["risk"])),
        "risk_mean": float(np.mean(grid["risk"])),
        "patch_coverage_ratio": float(np.mean(grid["patch_influence"] > 0.05)),
        "patch_coverage_by_name": grid["patch_coverage_by_name"],
        **path_summary,
    }
    (output_dir / "terrain_map_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def terrain_grid(
    terrain: TerrainField,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: int,
) -> dict:
    resolution = max(2, int(resolution))
    xs = np.linspace(float(xlim[0]), float(xlim[1]), resolution)
    ys = np.linspace(float(ylim[0]), float(ylim[1]), resolution)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    features = np.zeros((resolution, resolution, 4), dtype=np.float32)
    risk = np.zeros((resolution, resolution), dtype=np.float32)
    influence = np.zeros((resolution, resolution), dtype=np.float32)
    patch_hits = {str(patch["name"]): np.zeros((resolution, resolution), dtype=np.float32) for patch in terrain.patches}
    for row in range(resolution):
        for col in range(resolution):
            x = float(mesh_x[row, col])
            y = float(mesh_y[row, col])
            feature = terrain.feature(x, y)
            features[row, col] = feature
            risk[row, col] = terrain.risk_cost(x, y, features=feature)
            influences = terrain.patch_influences(x, y)
            if influences:
                influence[row, col] = max(influences.values())
                for name, value in influences.items():
                    patch_hits[name][row, col] = float(value)
    return {
        "x": mesh_x,
        "y": mesh_y,
        "features": features,
        "risk": risk,
        "patch_influence": influence,
        "patch_coverage_by_name": {
            name: float(np.mean(values > 0.05))
            for name, values in patch_hits.items()
        },
    }


def summarize_path_risks(
    *,
    terrain: TerrainField,
    start: np.ndarray,
    goal: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    path_samples: int,
) -> dict:
    straight_path = np.vstack([start, goal]).astype(np.float32)
    straight_risk = path_cumulative_risk(terrain, straight_path, path_samples)
    detours = []
    for offset in (2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0):
        path = detour_path(start, goal, offset)
        waypoint = path[1]
        if not (xlim[0] <= waypoint[0] <= xlim[1] and ylim[0] <= waypoint[1] <= ylim[1]):
            continue
        detours.append(
            {
                "offset": float(offset),
                "path": path.tolist(),
                "cumulative_risk": path_cumulative_risk(terrain, path, path_samples),
            }
        )
    if detours:
        best = min(detours, key=lambda item: item["cumulative_risk"])
    else:
        best = {"offset": 0.0, "path": straight_path.tolist(), "cumulative_risk": straight_risk}
    return {
        "straight_path": straight_path.tolist(),
        "straight_path_cumulative_risk": float(straight_risk),
        "best_detour_path": best["path"],
        "best_detour_offset": float(best["offset"]),
        "best_detour_cumulative_risk": float(best["cumulative_risk"]),
        "detour_path_cumulative_risks": detours,
    }


def detour_path(start: np.ndarray, goal: np.ndarray, offset: float) -> np.ndarray:
    direction = np.asarray(goal, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        perpendicular = np.array([0.0, 1.0], dtype=np.float32)
    else:
        unit = direction / norm
        perpendicular = np.array([-unit[1], unit[0]], dtype=np.float32)
    waypoint = 0.5 * (np.asarray(start, dtype=np.float32) + np.asarray(goal, dtype=np.float32))
    waypoint = waypoint + float(offset) * perpendicular
    return np.vstack([start, waypoint, goal]).astype(np.float32)


def path_cumulative_risk(terrain: TerrainField, path: np.ndarray, path_samples: int) -> float:
    points = sample_polyline(path, path_samples)
    if len(points) < 2:
        return 0.0
    total = 0.0
    for first, second in zip(points[:-1], points[1:]):
        midpoint = 0.5 * (first + second)
        segment_length = float(np.linalg.norm(second - first))
        total += terrain.risk_cost(float(midpoint[0]), float(midpoint[1])) * segment_length
    return float(total)


def sample_polyline(path: np.ndarray, path_samples: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32)
    if len(path) <= 1:
        return path
    segment_lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1e-6:
        return path[:1]
    distances = np.linspace(0.0, total_length, max(2, int(path_samples)))
    sampled = []
    segment_start_distance = 0.0
    segment_index = 0
    for distance in distances:
        while (
            segment_index < len(segment_lengths) - 1
            and distance > segment_start_distance + segment_lengths[segment_index]
        ):
            segment_start_distance += segment_lengths[segment_index]
            segment_index += 1
        length = max(1e-6, float(segment_lengths[segment_index]))
        alpha = float(np.clip((distance - segment_start_distance) / length, 0.0, 1.0))
        sampled.append((1.0 - alpha) * path[segment_index] + alpha * path[segment_index + 1])
    return np.asarray(sampled, dtype=np.float32)


def plot_risk_map(
    *,
    grid: dict,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    start: np.ndarray,
    goal: np.ndarray,
    best_detour: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(
        grid["risk"],
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap="plasma",
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="terrain risk")
    ax.plot([start[0], goal[0]], [start[1], goal[1]], color="white", linewidth=2.0, label="straight")
    ax.plot(best_detour[:, 0], best_detour[:, 1], color="#22c55e", linewidth=2.0, label="best detour")
    ax.scatter([start[0]], [start[1]], color="#10b981", edgecolor="#111827", zorder=5, label="start")
    ax.scatter([goal[0]], [goal[1]], color="#a855f7", edgecolor="#111827", zorder=5, label="goal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title} risk map")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_maps(
    *,
    grid: dict,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output_path: Path,
    title: str,
) -> None:
    names = ["slope_f", "slope_l", "roughness", "friction"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle(f"{title} terrain features", fontweight="bold")
    for index, (ax, name) in enumerate(zip(axes.flat, names)):
        image = ax.imshow(
            grid["features"][:, :, index],
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        fig.colorbar(image, ax=ax, shrink=0.85)
        ax.set_title(name)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="Terrain config YAML files to inspect.")
    parser.add_argument("--output", required=True, help="Output directory for map artifacts.")
    parser.add_argument("--grid-resolution", type=int, default=100)
    parser.add_argument("--path-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_configs(
        config_paths=[Path(path) for path in args.configs],
        output_dir=Path(args.output),
        grid_resolution=args.grid_resolution,
        path_samples=args.path_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
