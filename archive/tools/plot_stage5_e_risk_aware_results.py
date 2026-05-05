#!/usr/bin/env python3
"""Generate Stage 5-E risk-aware paper tables and Nature-style figures."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from b2_fdm_mppi.core.terrain import TerrainField
from tools.analyze_stage5_e_risk_aware import metric_delta_stats


PLOT_METRICS = ("final_distance", "cumulative_terrain_risk", "terrain_risk_excess", "control_jerk")
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
SCENARIO_COLORS = {
    "low_friction_patch": "#2F6DB3",
    "safe_corridor": "#009E73",
    "risk_band": "#D55E00",
    "two_obstacle_standard": "#B44CC2",
    "b2_omni_oracle": "#B44CC2",
}
METHOD_COLORS = {
    "nominal": "#6B7280",
    "nominal_risk": "#2F6DB3",
    "learned": "#E68632",
    "learned_risk": "#B44CC2",
}


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
    _apply_style()
    cases = _load_cases(sweep_summary_paths)
    if not cases:
        raise ValueError("No Stage 5-E cases found")

    _write_csv(tables_dir / "table_s5e_main_results.csv", _main_rows(cases))
    _write_csv(
        tables_dir / "table_s5e_paired_stats.csv",
        _stats_rows(cases, bootstrap_samples=bootstrap_samples, random_seed=random_seed),
    )
    _plot_delta_summary(cases, output_dir / "fig_s5e_main_risk_aware_summary")
    _plot_delta_boxplots(cases, output_dir / "fig_s5e_paired_delta_boxplots")
    _plot_pareto(cases, output_dir / "fig_s5e_risk_pareto")
    _plot_trajectories(cases, output_dir / "fig_s5e_trajectory_over_risk_map")
    _plot_two_obstacle(cases, output_dir / "fig_s5e_two_obstacle_trajectory_over_risk")
    _plot_timeseries(cases, output_dir / "fig_s5e_risk_timeseries")
    _plot_runtime(cases, output_dir / "fig_s5e_runtime")
    copied_gifs = _copy_visual_gifs(visual_summary_paths or [], output_dir)

    report = {
        "case_count": len(cases),
        "figure_dir": str(output_dir),
        "tables_dir": str(tables_dir),
        "copied_gifs": copied_gifs,
    }
    (output_dir / "stage5_e_figure_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_cases(paths: Sequence[str | Path]) -> list[dict]:
    cases = []
    for path in paths:
        sweep_path = Path(path)
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        for case in sweep.get("cases", []):
            case_dir = Path(case["output_dir"])
            benchmark_path = case_dir / "stage5_benchmark_summary.json"
            if not benchmark_path.is_file():
                continue
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            scenario = str(case.get("scenario_name", "scenario"))
            if scenario in {"standard", "oracle", "b2_omni_oracle"}:
                scenario = "two_obstacle_standard"
            cases.append(
                {
                    "case_name": str(case.get("case_name", case_dir.name)),
                    "scenario_name": scenario,
                    "terrain_risk_weight": float(case.get("terrain_risk_weight", 0.0)),
                    "config_path": str(case.get("config_path") or benchmark.get("metadata", {}).get("config", "")),
                    "output_dir": str(case_dir),
                    "benchmark": benchmark,
                }
            )
    return cases


def _main_rows(cases: Sequence[dict]) -> list[dict]:
    metric_keys = (
        "success_rate",
        "final_distance_mean",
        "steps_mean",
        "min_obstacle_clearance_mean",
        "mean_terrain_risk_mean",
        "max_terrain_risk_mean",
        "cumulative_terrain_risk_mean",
        "terrain_risk_excess_mean",
        "terrain_risk_exposure_ratio_mean",
        "control_smoothness_mean",
        "control_jerk_mean",
        "mean_mppi_time_ms_mean",
    )
    rows = []
    for case in cases:
        ag = case["benchmark"].get("aggregates", {})
        row = {
            "scenario_name": case["scenario_name"],
            "case_name": case["case_name"],
            "terrain_risk_weight": case["terrain_risk_weight"],
            "backend": case["benchmark"].get("metadata", {}).get("backend"),
            "episodes": case["benchmark"].get("metadata", {}).get("episodes"),
        }
        for key in metric_keys:
            nominal = ag.get("nominal", {}).get(key)
            learned = ag.get("learned", {}).get(key)
            row[f"nominal_{key}"] = _csv(nominal)
            row[f"learned_{key}"] = _csv(learned)
            row[key.replace("_mean", "_delta")] = _csv(_delta(learned, nominal))
        rows.append(row)
    return rows


def _stats_rows(cases: Sequence[dict], *, bootstrap_samples: int, random_seed: int) -> list[dict]:
    rows = []
    for case in cases:
        pairs = case["benchmark"].get("paired_deltas", {}).get("pairs", [])
        for metric in STATS_METRICS:
            stats = metric_delta_stats(
                [pair.get(f"{metric}_delta") for pair in pairs],
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
                    "mean_delta": _csv(stats.get("mean_delta")),
                    "median_delta": _csv(stats.get("median_delta")),
                    "std_delta": _csv(stats.get("std_delta")),
                    "bootstrap_ci95_low": _csv(low),
                    "bootstrap_ci95_high": _csv(high),
                    "pct_improved": _csv(stats.get("pct_improved")),
                    "wilcoxon_p": _csv(stats.get("wilcoxon_p")),
                    "wilcoxon_method": stats.get("wilcoxon_method"),
                }
            )
    return rows


def _plot_delta_summary(cases: Sequence[dict], stem: Path) -> None:
    selected = _select_cases(cases)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    for idx, (ax, metric) in enumerate(zip(axes.ravel(), PLOT_METRICS)):
        means, labels, colors = [], [], []
        err_low, err_high = [], []
        for case in selected:
            deltas = _deltas(case, metric)
            if deltas.size == 0:
                continue
            mean = float(np.mean(deltas))
            low, high = _bootstrap_ci(deltas)
            means.append(mean)
            err_low.append(mean - low)
            err_high.append(high - mean)
            labels.append(_short_label(case, include_weight=True))
            colors.append(_color(case))
        x = np.arange(len(means))
        ax.bar(x, means, color=colors, edgecolor="#333333", linewidth=0.5)
        if means:
            ax.errorbar(x, means, yerr=np.vstack([err_low, err_high]), fmt="none", color="#222222", capsize=2)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"Learned - nominal\n{metric.replace('_', ' ')}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=7)
        _panel(ax, idx)
    _save(fig, stem)


def _plot_delta_boxplots(cases: Sequence[dict], stem: Path) -> None:
    selected = _select_cases(cases)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    for idx, (ax, metric) in enumerate(zip(axes.ravel(), PLOT_METRICS)):
        plotted_cases = [case for case in selected if _deltas(case, metric).size]
        values = [_deltas(case, metric) for case in plotted_cases]
        labels = [_short_label(case, include_weight=True) for case in plotted_cases]
        if values:
            box = ax.boxplot(values, patch_artist=True, showfliers=False)
            for patch, case in zip(box["boxes"], plotted_cases):
                patch.set_facecolor(_color(case))
                patch.set_alpha(0.72)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"Learned - nominal\n{metric.replace('_', ' ')}")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=7)
        _panel(ax, idx)
    _save(fig, stem)


def _plot_pareto(cases: Sequence[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for case in cases:
        for controller, marker in (("nominal", "o"), ("learned", "s")):
            stats = case["benchmark"].get("aggregates", {}).get(controller, {})
            x = _finite(stats.get("final_distance_mean"))
            y = _finite(stats.get("cumulative_terrain_risk_mean"))
            if x is None or y is None:
                continue
            ax.scatter(x, y, marker=marker, s=42, color=_color(case), edgecolor="#222222", linewidth=0.4)
            ax.annotate(f"{_label(case)} {controller[0]}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Final distance [m]")
    ax.set_ylabel("Cumulative terrain risk")
    ax.scatter([], [], marker="o", color="#777777", label="Nominal")
    ax.scatter([], [], marker="s", color="#777777", label="Learned")
    ax.legend()
    _save(fig, stem)


def _plot_trajectories(cases: Sequence[dict], stem: Path) -> None:
    selected = _select_cases(cases)
    cols = min(2, len(selected))
    rows = int(math.ceil(len(selected) / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 3.2 * rows), squeeze=False, constrained_layout=True)
    for idx, (ax, case) in enumerate(zip(axes.ravel(), selected)):
        _draw_trajectory_panel(ax, case)
        _panel(ax, idx)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    _save(fig, stem)


def _plot_two_obstacle(cases: Sequence[dict], stem: Path) -> None:
    case = next((case for case in cases if case["scenario_name"] == "two_obstacle_standard"), None)
    if case is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    _draw_trajectory_panel(ax, case)
    _panel(ax, 0)
    _save(fig, stem)


def _plot_timeseries(cases: Sequence[dict], stem: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    for case in _select_cases(cases):
        for controller, ls in (("nominal", "-"), ("learned", "--")):
            risks = _mean_risk(case, controller)
            if risks.size == 0:
                continue
            axes[0].plot(risks, color=_color(case), linestyle=ls, linewidth=1.2, label=f"{_label(case)} {controller}")
            axes[1].plot(np.cumsum(risks), color=_color(case), linestyle=ls, linewidth=1.2)
    axes[0].set_ylabel("Risk(t)")
    axes[1].set_ylabel("Cumulative risk(t)")
    axes[1].set_xlabel("Closed-loop step")
    _panel(axes[0], 0)
    _panel(axes[1], 1)
    if axes[0].lines:
        axes[0].legend(fontsize=6, ncol=2)
    _save(fig, stem)


def _plot_runtime(cases: Sequence[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    labels = [_short_label(case, include_weight=True) for case in cases]
    x = np.arange(len(cases))
    nominal = [_finite(case["benchmark"].get("aggregates", {}).get("nominal", {}).get("mean_mppi_time_ms_mean")) or 0 for case in cases]
    learned = [_finite(case["benchmark"].get("aggregates", {}).get("learned", {}).get("mean_mppi_time_ms_mean")) or 0 for case in cases]
    width = 0.34
    ax.bar(x - width / 2, nominal, width, color=METHOD_COLORS["nominal"], label="Nominal torch")
    ax.bar(x + width / 2, learned, width, color=METHOD_COLORS["learned_risk"], label="Learned torch")
    ax.set_ylabel("Mean MPPI time [ms]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=6)
    ax.legend()
    _save(fig, stem)


def _draw_trajectory_panel(ax, case: dict) -> None:
    config = _config(case)
    xlim, ylim = _limits(config, case)
    risk, extent = _risk_grid(config, xlim, ylim)
    ax.imshow(risk, extent=extent, origin="lower", cmap="YlOrRd", alpha=0.36, aspect="auto")
    pair = (case["benchmark"].get("paired_deltas", {}).get("pairs") or [{}])[0]
    for key, color, label in (
        ("nominal_results_path", METHOD_COLORS["nominal_risk"], "Risk-aware Nominal-MPPI execution in oracle world"),
        ("learned_results_path", METHOD_COLORS["learned_risk"], "Risk-aware Learned-FDM-MPPI execution in oracle world"),
    ):
        path = Path(pair.get(key, "")) / "trajectory.csv"
        if path.is_file():
            traj = pd.read_csv(path)
            ax.plot(traj["x"], traj["y"], color=color, linewidth=1.8, label=label)
    _draw_scene(ax, config)
    ax.set_title(_label(case))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=5.5, loc="best")


def _draw_scene(ax, config: dict) -> None:
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
        x, y, r = float(obstacle[0]), float(obstacle[1]), float(obstacle[2])
        ax.add_patch(Circle((x, y), r, facecolor="#B7B7B7", edgecolor="#555555", alpha=0.75))
        ax.add_patch(Circle((x, y), r + robot_radius + safety_dist, fill=False, edgecolor="#D55E00", linestyle="--", linewidth=0.9))


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


def _limits(config: dict, case: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    sim = config.get("simulation", {})
    if "map_origin" in sim and "map_size" in sim:
        origin, size = sim["map_origin"], sim["map_size"]
        return (float(origin[0]), float(origin[0] + size[0])), (float(origin[1]), float(origin[1] + size[1]))
    xs, ys = [], []
    for pair in case["benchmark"].get("paired_deltas", {}).get("pairs", [])[:1]:
        for key in ("nominal_results_path", "learned_results_path"):
            path = Path(pair.get(key, "")) / "trajectory.csv"
            if path.is_file():
                traj = pd.read_csv(path)
                xs.extend(traj["x"].astype(float).tolist())
                ys.extend(traj["y"].astype(float).tolist())
    for field in ("initial_state", "goal"):
        value = sim.get(field, [0.0, 0.0])
        xs.append(float(value[0]))
        ys.append(float(value[1]))
    return (min(xs) - 0.8, max(xs) + 0.8), (min(ys) - 1.2, max(ys) + 1.2)


def _config(case: dict) -> dict:
    path = Path(case["config_path"])
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _select_cases(cases: Sequence[dict]) -> list[dict]:
    selected = []
    for name in ("low_friction_patch", "safe_corridor", "risk_band", "two_obstacle_standard"):
        match = next((case for case in cases if case["scenario_name"] == name and case["terrain_risk_weight"] > 0), None)
        if match:
            selected.append(match)
    return selected[:4] or list(cases[:4])


def _mean_risk(case: dict, controller: str) -> np.ndarray:
    curves = []
    for run in case["benchmark"].get("runs", []):
        if run.get("controller") != controller:
            continue
        path = Path(run.get("results_path", "")) / "terrain.csv"
        if not path.is_file():
            continue
        terrain = pd.read_csv(path)
        if "risk_cost" in terrain:
            curves.append(terrain["risk_cost"].astype(float).to_numpy())
    if not curves:
        return np.asarray([], dtype=float)
    n = min(len(curve) for curve in curves)
    return np.vstack([curve[:n] for curve in curves]).mean(axis=0)


def _copy_visual_gifs(paths: Sequence[str | Path], output_dir: Path) -> list[str]:
    copied = []
    for path in paths:
        summary_path = Path(path)
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run = summary.get("runs", {}).get("learned_risk_on") or summary.get("runs", {}).get("learned")
        if not run:
            continue
        source = Path(run.get("animation_gif", ""))
        if source.is_file():
            target = output_dir / "fig_s5e_two_obstacle_animation.gif"
            shutil.copyfile(source, target)
            copied.append(str(target))
    return copied


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _deltas(case: dict, metric: str) -> np.ndarray:
    values = []
    for pair in case["benchmark"].get("paired_deltas", {}).get("pairs", []):
        value = _finite(pair.get(f"{metric}_delta"))
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=float)


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size <= 1:
        value = float(values[0]) if values.size else 0.0
        return value, value
    rng = np.random.default_rng(123)
    means = values[rng.integers(0, values.size, size=(1000, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _delta(left, right):
    left = _finite(left)
    right = _finite(right)
    return None if left is None or right is None else left - right


def _finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _csv(value):
    value = _finite(value)
    return "" if value is None else value


def _label(case: dict) -> str:
    return f"{case['scenario_name'].replace('_', ' ')}\nw={case['terrain_risk_weight']:g}"


def _short_label(case: dict, *, include_weight: bool = False) -> str:
    names = {
        "low_friction_patch": "low friction",
        "safe_corridor": "safe corridor",
        "risk_band": "risk band",
        "two_obstacle_standard": "two obstacle",
        "b2_omni_oracle": "two obstacle",
    }
    name = names.get(case["scenario_name"], case["scenario_name"].replace("_", " "))
    if not include_weight:
        return name
    weight = float(case["terrain_risk_weight"])
    suffix = "risk off" if weight == 0.0 else f"w={weight:g}"
    return f"{name}\n{suffix}"


def _color(case: dict) -> str:
    return SCENARIO_COLORS.get(case["scenario_name"], "#4C78A8")


def _panel(ax, idx: int) -> None:
    ax.text(-0.12, 1.04, "abcdef"[idx], transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def parse_paths(value: str) -> list[str]:
    paths = [item.strip() for item in str(value).split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one path is required")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--visual-summary", default="")
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
