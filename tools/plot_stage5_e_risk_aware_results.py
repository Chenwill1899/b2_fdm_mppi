#!/usr/bin/env python3
"""Create Stage 5-E paper tables and Nature-style risk-aware figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.core.terrain import TerrainField
from tools.analyze_stage5_e_risk_aware import metric_delta_stats


PANEL_LABELS = ("a", "b", "c", "d", "e", "f")
METHOD_COLORS = {
    "nominal": "#6B7280",
    "nominal_risk": "#2F6DB3",
    "learned": "#E68632",
    "learned_risk": "#B44CC2",
}
SCENARIO_COLORS = {
    "low_friction_patch": "#2F6DB3",
    "safe_corridor": "#009E73",
    "risk_band": "#D55E00",
    "two_obstacle_standard": "#B44CC2",
    "b2_omni_oracle": "#B44CC2",
    "risk_island": "#CC79A7",
}
PLOT_METRICS = (
    "final_distance",
    "cumulative_terrain_risk",
    "terrain_risk_excess",
    "control_jerk",
)
STATS_METRICS = (
    "final_distance",
    "steps",
    "cumulative_terrain_risk",
    "max_terrain_risk",
    "terrain_risk_excess",
    "terrain_risk_exposure_ratio",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
)


def plot_stage5_e_results(
    *,
    sweep_summary_paths: Sequence[str | Path],
    output_dir: str | Path = "figures/stage5_e",
    tables_dir: str | Path = "tables/stage5_e",
    visual_summary_paths: Sequence[str | Path] | None = None,
    bootstrap_samples: int = 2000,
    random_seed: int = 123,
) -> dict:
    output_dir = Path(output_dir)
    tables_dir = Path(tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _apply_nature_style()
    cases = _load_cases(sweep_summary_paths)
    if not cases:
        raise ValueError("No Stage 5-E cases were found")

    table_rows = _main_table_rows(cases)
    stats_rows = _paired_stats_rows(cases, bootstrap_samples=bootstrap_samples, random_seed=random_seed)
    _write_csv(tables_dir / "table_s5e_main_results.csv", table_rows)
    _write_csv(tables_dir / "table_s5e_paired_stats.csv", stats_rows)

    _figure_main_delta_summary(cases, output_dir / "fig_s5e_main_risk_aware_summary")
    _figure_paired_delta_boxplots(cases, output_dir / "fig_s5e_paired_delta_boxplots")
    _figure_risk_pareto(cases, output_dir / "fig_s5e_risk_pareto")
    _figure_trajectory_over_risk_map(cases, output_dir / "fig_s5e_trajectory_over_risk_map")
    _figure_two_obstacle_trajectory(cases, output_dir / "fig_s5e_two_obstacle_trajectory_over_risk")
    _figure_risk_timeseries(cases, output_dir / "fig_s5e_risk_timeseries")
    _figure_runtime(cases, output_dir / "fig_s5e_runtime")
    copied_gifs = _copy_visual_artifacts(visual_summary_paths or [], output_dir)

    report = {
        "case_count": len(cases),
        "figure_dir": str(output_dir),
        "tables_dir": str(tables_dir),
        "figures": sorted(str(path) for path in output_dir.iterdir() if path.is_file()),
        "tables": sorted(str(path) for path in tables_dir.iterdir() if path.is_file()),
        "copied_gifs": copied_gifs,
    }
    (output_dir / "stage5_e_figure_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_cases(sweep_summary_paths: Sequence[str | Path]) -> list[dict]:
    cases = []
    for sweep_summary_path in sweep_summary_paths:
        sweep_path = Path(sweep_summary_path)
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        for case in sweep.get("cases", []):
            case_dir = Path(case["output_dir"])
            benchmark_path = case_dir / "stage5_benchmark_summary.json"
            if not benchmark_path.is_file():
                continue
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            cases.append(
                {
                    "case_name": str(case.get("case_name", case_dir.name)),
                    "scenario_name": _scenario_display_name(case.get("scenario_name", "scenario")),
                    "terrain_risk_weight": float(case.get("terrain_risk_weight", 0.0)),
                    "config_path": str(case.get("config_path") or benchmark.get("metadata", {}).get("config", "")),
                    "output_dir": str(case_dir),
                    "sweep_summary_path": str(sweep_path),
                    "benchmark_path": str(benchmark_path),
                    "benchmark": benchmark,
                }
            )
    return cases


def _main_table_rows(cases: Sequence[dict]) -> list[dict]:
    rows = []
    metrics = (
        "success_rate",
        "final_distance_mean",
        "steps_mean",
        "min_obstacle_clearance_mean",
        "mean_terrain_risk_mean",
        "max_terrain_risk_mean",
        "cumulative_terrain_risk_mean",
        "terrain_risk_excess_mean",
        "terrain_risk_excess_integral_mean",
        "terrain_risk_exposure_ratio_mean",
        "control_smoothness_mean",
        "control_jerk_mean",
        "mean_mppi_time_ms_mean",
    )
    for case in cases:
        aggregates = case["benchmark"].get("aggregates", {})
        row = {
            "scenario_name": case["scenario_name"],
            "case_name": case["case_name"],
            "terrain_risk_weight": case["terrain_risk_weight"],
            "backend": case["benchmark"].get("metadata", {}).get("backend"),
            "episodes": case["benchmark"].get("metadata", {}).get("episodes"),
        }
        for metric in metrics:
            nominal = aggregates.get("nominal", {}).get(metric)
            learned = aggregates.get("learned", {}).get(metric)
            row[f"nominal_{metric}"] = _csv_scalar(nominal)
            row[f"learned_{metric}"] = _csv_scalar(learned)
            row[metric.replace("_mean", "_delta")] = _csv_scalar(_delta(learned, nominal))
        rows.append(row)
    return rows


def _paired_stats_rows(cases: Sequence[dict], *, bootstrap_samples: int, random_seed: int) -> list[dict]:
    rows = []
    for case in cases:
        pairs = case["benchmark"].get("paired_deltas", {}).get("pairs", [])
        for metric in STATS_METRICS:
            values = [pair.get(f"{metric}_delta") for pair in pairs]
            stats = metric_delta_stats(
                values,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
            )
            low, high = stats.get("bootstrap_ci95", [None, None])
            rows.append(
                {
                    "scenario_name": case["scenario_name"],
                    "case_name": case["case_name"],
                    "terrain_risk_weight": case["terrain_risk_weight"],
                    "metric": metric,
                    "count": stats.get("count"),
                    "mean_delta": _csv_scalar(stats.get("mean_delta")),
                    "median_delta": _csv_scalar(stats.get("median_delta")),
                    "std_delta": _csv_scalar(stats.get("std_delta")),
                    "bootstrap_ci95_low": _csv_scalar(low),
                    "bootstrap_ci95_high": _csv_scalar(high),
                    "pct_improved": _csv_scalar(stats.get("pct_improved")),
                    "wilcoxon_p": _csv_scalar(stats.get("wilcoxon_p")),
                    "wilcoxon_method": stats.get("wilcoxon_method"),
                }
            )
    return rows


def _figure_main_delta_summary(cases: Sequence[dict], stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for idx, (ax, metric) in enumerate(zip(axes.ravel(), PLOT_METRICS)):
        labels, means, lows, highs, colors = [], [], [], [], []
        for case in cases:
            deltas = _case_metric_deltas(case, metric)
            if deltas.size == 0:
                continue
            mean = float(np.mean(deltas))
            low, high = _bootstrap_ci(deltas)
            labels.append(_short_case_label(case))
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
            colors.append(_scenario_color(case["scenario_name"]))
        x = np.arange(len(means))
        if len(means):
            ax.bar(x, means, color=colors, edgecolor="#333333", linewidth=0.6)
            ax.errorbar(x, means, yerr=np.vstack([lows, highs]), fmt="none", color="#222222", linewidth=1.0, capsize=2)
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_ylabel(_metric_label(metric, delta=True))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        _panel_label(ax, PANEL_LABELS[idx])
    _save_figure(fig, stem)


def _figure_paired_delta_boxplots(cases: Sequence[dict], stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for idx, (ax, metric) in enumerate(zip(axes.ravel(), PLOT_METRICS)):
        labels, values = [], []
        for case in cases:
            deltas = _case_metric_deltas(case, metric)
            if deltas.size:
                labels.append(_short_case_label(case))
                values.append(deltas)
        if values:
            box = ax.boxplot(values, patch_artist=True, showfliers=False)
            for patch, case in zip(box["boxes"], cases):
                patch.set_facecolor(_scenario_color(case["scenario_name"]))
                patch.set_alpha(0.72)
                patch.set_linewidth(0.7)
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_ylabel(_metric_label(metric, delta=True))
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        _panel_label(ax, PANEL_LABELS[idx])
    _save_figure(fig, stem)


def _figure_risk_pareto(cases: Sequence[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for case in cases:
        aggregates = case["benchmark"].get("aggregates", {})
        color = _scenario_color(case["scenario_name"])
        for controller, marker, edge in (("nominal", "o", "#333333"), ("learned", "s", "#111111")):
            stats = aggregates.get(controller, {})
            x = _finite_float(stats.get("final_distance_mean"))
            y = _finite_float(stats.get("cumulative_terrain_risk_mean"))
            if x is None or y is None:
                continue
            ax.scatter(x, y, marker=marker, s=44, color=color, edgecolor=edge, linewidth=0.5, alpha=0.85)
            ax.annotate(
                f"{_short_case_label(case)} {controller[0].upper()}",
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.8,
            )
    ax.set_xlabel("Final distance [m]")
    ax.set_ylabel("Cumulative terrain risk")
    ax.scatter([], [], marker="o", color="#888888", label="Nominal")
    ax.scatter([], [], marker="s", color="#888888", label="Learned")
    ax.legend(loc="best")
    _save_figure(fig, stem)


def _figure_trajectory_over_risk_map(cases: Sequence[dict], stem: Path) -> None:
    selected = _select_trajectory_cases(cases)
    cols = min(2, len(selected))
    rows = int(math.ceil(len(selected) / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 3.2 * rows), squeeze=False, constrained_layout=True)
    for idx, (ax, case) in enumerate(zip(axes.ravel(), selected)):
        _draw_case_trajectory_panel(ax, case)
        _panel_label(ax, PANEL_LABELS[idx])
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    _save_figure(fig, stem)


def _figure_two_obstacle_trajectory(cases: Sequence[dict], stem: Path) -> None:
    case = next(
        (item for item in cases if item["scenario_name"] in {"two_obstacle_standard", "b2_omni_oracle"}),
        None,
    )
    if case is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    _draw_case_trajectory_panel(ax, case)
    _panel_label(ax, "a")
    _save_figure(fig, stem)


def _figure_risk_timeseries(cases: Sequence[dict], stem: Path) -> None:
    selected = _select_trajectory_cases(cases)[:4]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    for case in selected:
        color = _scenario_color(case["scenario_name"])
        for controller, linestyle in (("nominal", "-"), ("learned", "--")):
            mean_risk = _mean_controller_risk(case, controller, cumulative=False)
            cumulative = _mean_controller_risk(case, controller, cumulative=True)
            if mean_risk.size:
                label = f"{_short_case_label(case)} {controller}"
                axes[0].plot(mean_risk, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9, label=label)
            if cumulative.size:
                axes[1].plot(cumulative, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9, label=label)
    axes[0].set_ylabel("Risk(t)")
    axes[1].set_ylabel("Cumulative risk(t)")
    axes[1].set_xlabel("Closed-loop step")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    if axes[0].lines:
        axes[0].legend(ncol=2, fontsize=6.5)
    _save_figure(fig, stem)


def _figure_runtime(cases: Sequence[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    width = 0.34
    x = np.arange(len(cases))
    nominal, learned, labels = [], [], []
    for case in cases:
        aggregates = case["benchmark"].get("aggregates", {})
        labels.append(_short_case_label(case))
        nominal.append(_finite_float(aggregates.get("nominal", {}).get("mean_mppi_time_ms_mean")) or 0.0)
        learned.append(_finite_float(aggregates.get("learned", {}).get("mean_mppi_time_ms_mean")) or 0.0)
    ax.bar(x - width / 2.0, nominal, width=width, color=METHOD_COLORS["nominal"], label="Nominal torch")
    ax.bar(x + width / 2.0, learned, width=width, color=METHOD_COLORS["learned_risk"], label="Learned torch")
    ax.set_ylabel("Mean MPPI time [ms]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    _save_figure(fig, stem)


def _draw_case_trajectory_panel(ax, case: dict) -> None:
    config = _read_config(case)
    xlim, ylim = _map_limits(config, case)
    risk_grid, extent = _risk_grid(config, xlim, ylim)
    im = ax.imshow(risk_grid, extent=extent, origin="lower", cmap="YlOrRd", alpha=0.36, aspect="auto")
    pairs = case["benchmark"].get("paired_deltas", {}).get("pairs", [])
    pair = pairs[0] if pairs else {}
    for controller, path_key, color, label in (
        ("nominal", "nominal_results_path", METHOD_COLORS["nominal_risk"], "Risk-aware Nominal-MPPI execution in oracle world"),
        ("learned", "learned_results_path", METHOD_COLORS["learned_risk"], "Risk-aware Learned-FDM-MPPI execution in oracle world"),
    ):
        run_path = Path(pair.get(path_key, ""))
        traj_path = run_path / "trajectory.csv"
        if traj_path.is_file():
            traj = pd.read_csv(traj_path)
            ax.plot(traj["x"], traj["y"], color=color, linewidth=1.8, label=label)
    _draw_start_goal_obstacles(ax, config)
    ax.set_title(_short_case_label(case))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=5.8, loc="best")
    return im


def _draw_start_goal_obstacles(ax, config: dict) -> None:
    from matplotlib.patches import Circle

    sim = config.get("simulation", {})
    robot = config.get("robot", {})
    start = np.asarray(sim.get("initial_state", [0.0, 0.0])[:2], dtype=float)
    goal = np.asarray(sim.get("goal", [0.0, 0.0])[:2], dtype=float)
    ax.scatter([start[0]], [start[1]], color="#009E73", s=26, marker="o", label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], color="#7A3DB8", s=44, marker="*", label="goal", zorder=5)
    robot_radius = float(robot.get("radius", 0.0))
    safety_dist = float(robot.get("safety_dist", 0.0))
    for obstacle in config.get("obstacles", {}).get("virtual", []):
        if len(obstacle) < 3:
            continue
        x, y, radius = float(obstacle[0]), float(obstacle[1]), float(obstacle[2])
        ax.add_patch(Circle((x, y), radius, facecolor="#B7B7B7", edgecolor="#555555", alpha=0.75, label="obstacle"))
        ax.add_patch(
            Circle(
                (x, y),
                radius + robot_radius + safety_dist,
                fill=False,
                edgecolor="#D55E00",
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
            )
        )


def _risk_grid(config: dict, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    terrain = TerrainField.from_config(config.get("terrain"))
    resolution = int(config.get("visualization", {}).get("terrain_grid_resolution", 100))
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    grid = np.zeros((len(ys), len(xs)), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            grid[iy, ix] = terrain.risk_cost(float(x), float(y))
    return grid, (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))


def _map_limits(config: dict, case: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    sim = config.get("simulation", {})
    if "map_size" in sim and "map_origin" in sim:
        origin = sim["map_origin"]
        size = sim["map_size"]
        return (float(origin[0]), float(origin[0] + size[0])), (float(origin[1]), float(origin[1] + size[1]))
    points_x, points_y = [], []
    for pair in case["benchmark"].get("paired_deltas", {}).get("pairs", [])[:1]:
        for key in ("nominal_results_path", "learned_results_path"):
            traj_path = Path(pair.get(key, "")) / "trajectory.csv"
            if traj_path.is_file():
                traj = pd.read_csv(traj_path)
                points_x.extend([float(v) for v in traj["x"].to_numpy()])
                points_y.extend([float(v) for v in traj["y"].to_numpy()])
    for field in ("initial_state", "goal"):
        value = sim.get(field, [0.0, 0.0])
        points_x.append(float(value[0]))
        points_y.append(float(value[1]))
    x_margin = max(0.8, 0.08 * (max(points_x) - min(points_x) if points_x else 1.0))
    y_margin = max(1.2, 0.25 * (max(points_y) - min(points_y) if points_y else 1.0))
    return (min(points_x) - x_margin, max(points_x) + x_margin), (min(points_y) - y_margin, max(points_y) + y_margin)


def _read_config(case: dict) -> dict:
    config_path = Path(case.get("config_path") or case["benchmark"].get("metadata", {}).get("config", ""))
    if not config_path.is_file():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _select_trajectory_cases(cases: Sequence[dict]) -> list[dict]:
    preferred = ("low_friction_patch", "safe_corridor", "risk_band", "two_obstacle_standard", "b2_omni_oracle")
    selected = []
    for name in preferred:
        match = next((case for case in cases if case["scenario_name"] == name), None)
        if match is not None:
            selected.append(match)
    if not selected:
        selected = list(cases[:4])
    return selected[:4]


def _case_metric_deltas(case: dict, metric: str) -> np.ndarray:
    values = []
    for pair in case["benchmark"].get("paired_deltas", {}).get("pairs", []):
        value = _finite_float(pair.get(f"{metric}_delta"))
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=float)


def _mean_controller_risk(case: dict, controller: str, *, cumulative: bool) -> np.ndarray:
    curves = []
    for run in case["benchmark"].get("runs", []):
        if run.get("controller") != controller:
            continue
        terrain_path = Path(run.get("results_path", "")) / "terrain.csv"
        if terrain_path.is_file():
            values = _read_risk_series(terrain_path)
            if values.size:
                curves.append(np.cumsum(values) if cumulative else values)
    if not curves:
        return np.asarray([], dtype=float)
    min_len = min(len(curve) for curve in curves)
    return np.vstack([curve[:min_len] for curve in curves]).mean(axis=0)


def _read_risk_series(path: Path) -> np.ndarray:
    values = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = _finite_float(row.get("risk_cost"))
            if value is not None:
                values.append(value)
    return np.asarray(values, dtype=float)


def _copy_visual_artifacts(visual_summary_paths: Sequence[str | Path], output_dir: Path) -> list[str]:
    copied = []
    for summary_path in visual_summary_paths:
        path = Path(summary_path)
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        run = summary.get("runs", {}).get("learned_risk_on") or summary.get("runs", {}).get("learned")
        if not run:
            continue
        source = Path(run.get("animation_gif", ""))
        if not source.is_file():
            continue
        target = output_dir / "fig_s5e_two_obstacle_animation.gif"
        shutil.copyfile(source, target)
        copied.append(str(target))
    return copied


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(123)
    idx = rng.integers(0, values.size, size=(1000, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _scenario_display_name(value) -> str:
    name = str(value)
    if name in {"b2_omni_oracle", "standard", "oracle"}:
        return "two_obstacle_standard"
    return name


def _short_case_label(case: dict) -> str:
    name = str(case["scenario_name"]).replace("_", "\n")
    return f"{name}\nw={case['terrain_risk_weight']:g}"


def _metric_label(metric: str, *, delta: bool = False) -> str:
    labels = {
        "final_distance": "Final distance [m]",
        "cumulative_terrain_risk": "Cumulative terrain risk",
        "terrain_risk_excess": "Terrain risk excess",
        "control_jerk": "Control jerk",
    }
    label = labels.get(metric, metric.replace("_", " "))
    return f"Learned - nominal\n{label}" if delta else label


def _scenario_color(scenario_name: str) -> str:
    return SCENARIO_COLORS.get(str(scenario_name), "#4C78A8")


def _panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom", ha="left")


def _delta(left, right):
    left = _finite_float(left)
    right = _finite_float(right)
    if left is None or right is None:
        return None
    return float(left - right)


def _finite_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _csv_scalar(value):
    value = _finite_float(value)
    return "" if value is None else value


def parse_paths(value: str) -> list[str]:
    paths = [item.strip() for item in str(value).split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one path is required")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", required=True, help="Comma-separated Stage 5-E sweep summary paths.")
    parser.add_argument("--visual-summary", default="", help="Optional comma-separated visual summary paths.")
    parser.add_argument("--output", default="figures/stage5_e")
    parser.add_argument("--tables-output", default="tables/stage5_e")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args()

    report = plot_stage5_e_results(
        sweep_summary_paths=parse_paths(args.sweep_summary),
        output_dir=args.output,
        tables_dir=args.tables_output,
        visual_summary_paths=parse_paths(args.visual_summary) if args.visual_summary else [],
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
