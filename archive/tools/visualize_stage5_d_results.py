#!/usr/bin/env python3
"""Build static visualizations for Stage 5-D benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


METRICS = [
    ("final_distance", "Final dist", "m", "lower"),
    ("steps", "Steps", "steps", "lower"),
    ("min_obstacle_clearance", "Clearance", "m", "higher"),
    ("mean_terrain_risk", "Terrain risk", "", "lower"),
    ("control_smoothness", "Smoothness", "", "lower"),
    ("control_jerk", "Jerk", "", "lower"),
]

CONTROLLER_ORDER = [
    "nominal",
    "default g1.0",
    "current 0.5/3.5/1.0",
    "balanced 0.5/3.0/0.75",
]

CONTROLLER_LABELS = {
    "nominal": "Nominal CUDA",
    "default g1.0": "Default learned\n(g=1.0)",
    "current 0.5/3.5/1.0": "Efficiency\n0.5/3.5/1.0",
    "balanced 0.5/3.0/0.75": "Balanced\n0.5/3.0/0.75",
}

SCENARIO_LABELS = {
    "id_random": "ID random",
    "ood_obstacle": "OOD obstacle",
    "ood_terrain": "OOD terrain",
}

COLORS = {
    "nominal": "#4b5563",
    "default g1.0": "#2563eb",
    "current 0.5/3.5/1.0": "#dc2626",
    "balanced 0.5/3.0/0.75": "#059669",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "scenario",
        "controller",
        "run_count",
        "failed_count",
        "success_rate",
        "final_distance_mean",
        "final_distance_delta",
        "steps_mean",
        "steps_delta",
        "min_obstacle_clearance_mean",
        "min_obstacle_clearance_delta",
        "mean_terrain_risk_mean",
        "mean_terrain_risk_delta",
        "control_smoothness_mean",
        "control_smoothness_delta",
        "control_jerk_mean",
        "control_jerk_delta",
        "mean_mppi_time_ms_mean",
        "mean_mppi_time_ms_delta",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rows_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    for scenario in grouped:
        grouped[scenario].sort(key=lambda r: CONTROLLER_ORDER.index(r["controller"]))
    return dict(sorted(grouped.items()))


def _mean_value(row: dict[str, Any], metric: str) -> float:
    return float(row[f"{metric}_mean"])


def _delta_value(row: dict[str, Any], metric: str) -> float:
    return float(row[f"{metric}_delta"])


def plot_metric_means(rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped = _rows_by_scenario(rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), constrained_layout=True)
    fig.suptitle("S5-010 50-episode benchmark: metric means", fontsize=16, fontweight="bold")

    for ax, (metric, title, unit, direction) in zip(axes.flat, METRICS):
        width = 0.18
        scenario_names = list(grouped.keys())
        x = np.arange(len(scenario_names))
        for i, controller in enumerate(CONTROLLER_ORDER):
            vals = [
                _mean_value(next(r for r in grouped[s] if r["controller"] == controller), metric)
                for s in scenario_names
            ]
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width,
                label=CONTROLLER_LABELS[controller].replace("\n", " "),
                color=COLORS[controller],
            )
        subtitle = "lower is better" if direction == "lower" else "higher is better"
        ax.set_title(f"{title} ({subtitle})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenario_names], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if unit:
            ax.set_ylabel(unit)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_deltas(rows: list[dict[str, Any]], output_path: Path) -> None:
    learned_rows = [r for r in rows if r["controller"] != "nominal"]
    grouped = _rows_by_scenario(learned_rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), constrained_layout=True)
    fig.suptitle("S5-010 learned-minus-nominal deltas", fontsize=16, fontweight="bold")

    for ax, (metric, title, unit, direction) in zip(axes.flat, METRICS):
        scenario_names = list(grouped.keys())
        x = np.arange(len(scenario_names))
        width = 0.24
        for i, controller in enumerate(CONTROLLER_ORDER[1:]):
            vals = [
                _delta_value(next(r for r in grouped[s] if r["controller"] == controller), metric)
                for s in scenario_names
            ]
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=CONTROLLER_LABELS[controller].replace("\n", " "),
                color=COLORS[controller],
            )
        ax.axhline(0.0, color="#111827", linewidth=0.9)
        subtitle = "negative is better" if direction == "lower" else "positive is better"
        ax.set_title(f"{title} delta ({subtitle})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenario_names], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if unit:
            ax.set_ylabel(unit)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped = _rows_by_scenario(rows)
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    width = 0.18
    scenario_names = list(grouped.keys())
    x = np.arange(len(scenario_names))
    for i, controller in enumerate(CONTROLLER_ORDER):
        vals = [
            float(next(r for r in grouped[s] if r["controller"] == controller)["mean_mppi_time_ms_mean"])
            for s in scenario_names
        ]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=CONTROLLER_LABELS[controller].replace("\n", " "),
            color=COLORS[controller],
        )
    ax.set_title("Mean MPPI runtime per step", fontsize=14, fontweight="bold")
    ax.set_ylabel("ms")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenario_names])
    ax.axhline(50.0, color="#f59e0b", linestyle="--", linewidth=1.2, label="50 ms reference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, Any]], output_path: Path) -> None:
    learned_rows = [r for r in rows if r["controller"] != "nominal"]
    fig, ax = plt.subplots(figsize=(11.2, 6.5), constrained_layout=True)
    markers = {"id_random": "o", "ood_obstacle": "s", "ood_terrain": "^"}
    for row in learned_rows:
        controller = row["controller"]
        ax.scatter(
            row["steps_delta"],
            row["mean_terrain_risk_delta"],
            s=130,
            marker=markers.get(row["scenario"], "o"),
            color=COLORS[controller],
            edgecolors="#111827",
            linewidths=0.8,
            alpha=0.9,
        )
    ax.axvline(0.0, color="#111827", linewidth=0.9)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_title("Efficiency vs terrain-risk tradeoff", fontsize=14, fontweight="bold")
    ax.set_xlabel("Steps delta vs nominal (negative is faster)")
    ax.set_ylabel("Terrain-risk delta vs nominal (negative is safer)")
    ax.grid(alpha=0.25)
    controller_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[c], markeredgecolor="#111827", label=CONTROLLER_LABELS[c].replace("\n", " "), markersize=9)
        for c in CONTROLLER_ORDER[1:]
    ]
    scenario_handles = [
        plt.Line2D([0], [0], marker=marker, color="#111827", linestyle="None", label=SCENARIO_LABELS.get(scenario, scenario), markersize=8)
        for scenario, marker in markers.items()
    ]
    first_legend = ax.legend(handles=controller_handles, fontsize=8, loc="center", bbox_to_anchor=(0.64, 0.58), title="Controller")
    ax.add_artist(first_legend)
    ax.legend(handles=scenario_handles, fontsize=8, loc="center", bbox_to_anchor=(0.64, 0.34), title="Scenario")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_episode_metric_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        source = Path(row["source_dir"]) / "stage5_benchmark_summary.json"
        if not source.exists():
            continue
        data = _load_json(source)
        out[(row["scenario"], row["controller"])] = list(data.get("runs", []))
    return out


def plot_episode_distributions(rows: list[dict[str, Any]], output_path: Path) -> None:
    episode_rows = _load_episode_metric_rows(rows)
    grouped = _rows_by_scenario(rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), constrained_layout=True)
    fig.suptitle("S5-010 episode distributions", fontsize=16, fontweight="bold")
    metrics = [("final_distance", "Final distance", "m"), ("steps", "Steps", "steps")]
    scenarios = list(grouped.keys())
    for row_idx, (metric, metric_title, ylabel) in enumerate(metrics):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            values = []
            colors = []
            labels = []
            for controller in CONTROLLER_ORDER:
                runs = episode_rows.get((scenario, controller), [])
                values.append([float(run[metric]) for run in runs if run.get(metric) is not None])
                colors.append(COLORS[controller])
                labels.append(CONTROLLER_LABELS[controller])
            bp = ax.boxplot(values, widths=0.6, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)
            for median in bp["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.4)
            ax.set_title(f"{SCENARIO_LABELS.get(scenario, scenario)} - {metric_title}", fontsize=11)
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_episode_runs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for row in rows:
        source = Path(row["source_dir"]) / "stage5_benchmark_summary.json"
        if not source.exists():
            continue
        data = _load_json(source)
        run_df = pd.DataFrame(data.get("runs", []))
        run_df["scenario_key"] = row["scenario"]
        run_df["controller_key"] = row["controller"]
        run_df["source_dir"] = row["source_dir"]
        frames.append(run_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _choose_typical_episodes(run_df: pd.DataFrame) -> dict[str, int]:
    choices = {}
    current = run_df[run_df["controller_key"] == "current 0.5/3.5/1.0"]
    nominal = run_df[run_df["controller_key"] == "nominal"][
        ["scenario_key", "episode_id", "steps"]
    ].rename(columns={"steps": "nominal_steps"})
    paired = current.merge(nominal, on=["scenario_key", "episode_id"], how="inner")
    paired["delta_steps"] = paired["steps"] - paired["nominal_steps"]
    for scenario, sub in paired.groupby("scenario_key"):
        median_delta = sub["delta_steps"].median()
        idx = (sub["delta_steps"] - median_delta).abs().idxmin()
        choices[scenario] = int(sub.loc[idx, "episode_id"])
    return choices


def _run_path(run_df: pd.DataFrame, scenario: str, controller: str, episode_id: int) -> Path:
    row = run_df[
        (run_df["scenario_key"] == scenario)
        & (run_df["controller_key"] == controller)
        & (run_df["episode_id"] == episode_id)
    ].iloc[0]
    return Path(row["results_path"])


def _load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _load_obstacles(config_path: Path) -> list[tuple[float, float, float]]:
    config = _load_config(config_path)
    obstacles = []
    for item in config.get("obstacles", {}).get("virtual", []):
        if len(item) >= 3:
            obstacles.append((float(item[0]), float(item[1]), float(item[2])))
    return obstacles


def _load_goal_tolerance(config_path: Path) -> float:
    config = _load_config(config_path)
    return float(config.get("simulation", {}).get("minimum_distance", 0.0))


def plot_trajectory_gallery(rows: list[dict[str, Any]], output_path: Path) -> None:
    run_df = _load_episode_runs(rows)
    choices = _choose_typical_episodes(run_df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("S5-010 typical trajectory gallery", fontsize=16, fontweight="bold")

    for ax, scenario in zip(axes, SCENARIO_LABELS):
        episode_id = choices.get(scenario, 0)
        trajectories = {}
        for controller in CONTROLLER_ORDER:
            path = _run_path(run_df, scenario, controller, episode_id)
            trajectories[controller] = pd.read_csv(path / "trajectory.csv")

        ref = next(iter(trajectories.values()))
        goal_x = float(ref["x_des"].iloc[0])
        goal_y = float(ref["y_des"].iloc[0])
        start_x = float(ref["x"].iloc[0])
        start_y = float(ref["y"].iloc[0])
        all_x = np.concatenate([df["x"].to_numpy() for df in trajectories.values()])
        all_y = np.concatenate([df["y"].to_numpy() for df in trajectories.values()])
        config_path = _run_path(run_df, scenario, "nominal", episode_id) / "config.yaml"
        goal_tolerance = _load_goal_tolerance(config_path)
        xmin = min(float(all_x.min()), goal_x - goal_tolerance, start_x) - 3.0
        xmax = max(float(all_x.max()), goal_x + goal_tolerance, start_x) + 3.0
        ymin = min(float(all_y.min()), goal_y - goal_tolerance, start_y) - 3.0
        ymax = max(float(all_y.max()), goal_y + goal_tolerance, start_y) + 3.0

        for ox, oy, radius in _load_obstacles(config_path):
            if xmin - radius <= ox <= xmax + radius and ymin - radius <= oy <= ymax + radius:
                ax.add_patch(plt.Circle((ox, oy), radius, color="#9ca3af", alpha=0.28, linewidth=0))

        for controller, df in trajectories.items():
            ax.plot(
                df["x"],
                df["y"],
                color=COLORS[controller],
                linewidth=2.0,
                label=CONTROLLER_LABELS[controller].replace("\n", " "),
            )
        ax.scatter([start_x], [start_y], marker="o", s=70, color="#111827", label="Start")
        ax.scatter(
            [goal_x],
            [goal_y],
            marker="*",
            s=130,
            color="#f59e0b",
            edgecolor="#111827",
            linewidth=0.5,
            label="Goal",
        )
        if goal_tolerance > 0.0:
            ax.add_patch(
                plt.Circle(
                    (goal_x, goal_y),
                    goal_tolerance,
                    fill=False,
                    linestyle="--",
                    linewidth=1.4,
                    edgecolor="#f59e0b",
                    alpha=0.95,
                    label="Goal tolerance",
                )
            )
        ax.set_title(f"{SCENARIO_LABELS.get(scenario, scenario)} episode {episode_id:04d}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes[0].legend(fontsize=7, loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_summary_gif(rows: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 1200, 720
    bg = "#f8fafc"
    fg = "#111827"
    accent = "#2563eb"
    frames: list[Image.Image] = []

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
        h_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 25)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError:
        title_font = h_font = body_font = small_font = ImageFont.load_default()

    def frame(title: str, lines: list[str], footer: str = "") -> Image.Image:
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width, 86], fill="#e0f2fe")
        draw.text((48, 24), title, fill=fg, font=title_font)
        y = 140
        for line in lines:
            if line.startswith("## "):
                draw.text((58, y), line[3:], fill=accent, font=h_font)
                y += 48
            else:
                draw.text((82, y), line, fill=fg, font=body_font)
                y += 42
        if footer:
            draw.text((48, height - 50), footer, fill="#475569", font=small_font)
        return image

    frames.append(
        frame(
            "S5-010 Stage 5-D benchmark",
            [
                "## Official 50-episode matrix",
                "3 scenarios: ID random, OOD obstacle, OOD terrain",
                "4 controllers: nominal, default, efficiency, balanced",
                "12 official groups, 0 aggregate errors",
                "All official groups reached success_rate = 1.0",
            ],
            "Output: results/stage5_d/s5_010_parallel/visualization",
        )
    )
    frames.append(
        frame(
            "Default learned g=1.0",
            [
                "## Conservative / smooth mode",
                "Average final-distance delta: +0.00798 m",
                "Average steps delta: +24.43 steps",
                "Average terrain-risk delta: -0.02179",
                "Smoothness and jerk are both lower than nominal",
            ],
            "Good for risk/smoothness, not for arrival efficiency.",
        )
    )
    frames.append(
        frame(
            "Current efficiency 0.5/3.5/1.0",
            [
                "## Efficiency mode",
                "Average final-distance delta: -0.00322 m",
                "Average steps delta: -7.15 steps",
                "Best official final-distance and steps gains",
                "Small risk/smoothness/jerk penalties",
            ],
            "Use when the claim focuses on navigation efficiency.",
        )
    )
    frames.append(
        frame(
            "Balanced 0.5/3.0/0.75",
            [
                "## Balanced operating-point candidate",
                "Average final-distance delta: +0.00005 m",
                "Average steps delta: -3.20 steps",
                "Average terrain-risk delta: -0.00002",
                "Small smoothness and jerk penalties",
            ],
            "Balanced is a candidate, not an all-metric winner.",
        )
    )
    frames.append(
        frame(
            "Runtime boundary",
            [
                "## Offline benchmark is practical",
                "Nominal CUDA: about 5.8-6.4 ms per MPPI step",
                "Learned Torch/CUDA: about 52-61 ms per MPPI step",
                "Runtime profiling remains required before real-time claims",
            ],
            "Next: paper-result framing and focused runtime profiling.",
        )
    )

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1800,
        loop=0,
        optimize=False,
    )


def write_html(output_dir: Path) -> None:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>S5-010 Stage 5-D visualization</title>
  <style>
    :root { color-scheme: light; --ink:#111827; --muted:#475569; --line:#d8dee9; --band:#f8fafc; }
    body { font-family: Arial, sans-serif; margin: 0; color: var(--ink); background: white; }
    header { padding: 28px 32px 18px; background: #e0f2fe; border-bottom: 1px solid var(--line); }
    main { max-width: 1180px; margin: 0 auto; padding: 24px 24px 48px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin: 34px 0 12px; font-size: 22px; }
    p { line-height: 1.48; color: var(--muted); }
    img { width: 100%; border: 1px solid var(--line); margin: 8px 0 24px; background: var(--band); }
    code { background: #eef2f7; padding: 2px 5px; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 18px; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 7px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th:nth-child(2), td:nth-child(2) { text-align: left; }
    .note { background: var(--band); border: 1px solid var(--line); padding: 12px 14px; color: var(--muted); }
  </style>
</head>
<body>
<header>
  <h1>S5-010 Stage 5-D 50-episode visualization</h1>
  <p>Generated from <code>results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json</code>. Deltas are learned controller minus nominal CUDA. Lower is better for final distance, steps, risk, smoothness, jerk, and runtime; higher is better for clearance.</p>
</header>
<main>
  <section class="note">
    <strong>Result boundary:</strong> S5-010 supports three learned-FDM operating modes, not a single all-metric winner. Default <code>g=1.0</code> is conservative/smooth, current <code>0.5/3.5/1.0</code> is efficiency mode, and balanced <code>0.5/3.0/0.75</code> is a balanced operating-point candidate.
  </section>
  <h2>Summary GIF</h2>
  <img src="gifs/s5_010_summary_slides.gif" alt="S5-010 summary slides GIF">
  <h2>Metric Means</h2>
  <img src="s5_010_metric_means.png" alt="Metric mean bar charts">
  <h2>Learned-Minus-Nominal Deltas</h2>
  <img src="s5_010_metric_deltas.png" alt="Metric delta bar charts">
  <h2>Efficiency vs Terrain-Risk Tradeoff</h2>
  <img src="s5_010_tradeoff_scatter.png" alt="Tradeoff scatter plot">
  <h2>Runtime</h2>
  <img src="s5_010_runtime_bars.png" alt="Runtime bar chart">
  <h2>Typical Trajectory Gallery</h2>
  <p>The dashed circle around each goal is the configured arrival tolerance from <code>simulation.minimum_distance</code>. A trajectory is counted as successful once it enters this circle, so it does not need to end exactly on the star marker.</p>
  <img src="s5_010_trajectory_gallery.png" alt="Typical trajectory gallery with goal tolerance circles">
  <h2>Episode Distributions</h2>
  <img src="s5_010_episode_distributions.png" alt="Episode distribution boxplots">
  <p>Metrics CSV: <a href="s5_010_visual_metrics.csv">s5_010_visual_metrics.csv</a></p>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json"),
        help="Official S5-010 aggregate summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage5_d/s5_010_parallel/visualization"),
        help="Visualization output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = output_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    data = _load_json(summary_path)
    errors = data.get("errors", [])
    if errors:
        raise RuntimeError(f"Summary has aggregate errors: {errors}")
    rows = data.get("rows", [])
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 official rows, got {len(rows)}")

    _write_csv(rows, output_dir / "s5_010_visual_metrics.csv")
    plot_metric_means(rows, output_dir / "s5_010_metric_means.png")
    plot_metric_deltas(rows, output_dir / "s5_010_metric_deltas.png")
    plot_tradeoff(rows, output_dir / "s5_010_tradeoff_scatter.png")
    plot_runtime(rows, output_dir / "s5_010_runtime_bars.png")
    plot_trajectory_gallery(rows, output_dir / "s5_010_trajectory_gallery.png")
    plot_episode_distributions(rows, output_dir / "s5_010_episode_distributions.png")
    create_summary_gif(rows, gif_dir / "s5_010_summary_slides.gif")
    write_html(output_dir)

    manifest = {
        "summary": str(summary_path),
        "output_dir": str(output_dir),
        "official_rows": len(rows),
        "errors": errors,
        "artifacts": [
            "index.html",
            "s5_010_visual_metrics.csv",
            "s5_010_metric_means.png",
            "s5_010_metric_deltas.png",
            "s5_010_tradeoff_scatter.png",
            "s5_010_runtime_bars.png",
            "s5_010_trajectory_gallery.png",
            "s5_010_episode_distributions.png",
            "gifs/s5_010_summary_slides.gif",
        ],
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
