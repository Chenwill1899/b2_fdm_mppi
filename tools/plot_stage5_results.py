#!/usr/bin/env python3
"""Create a paper-ready Stage 5 result package from S5-010 outputs."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


CONTROLLER_ORDER = [
    "nominal",
    "default g1.0",
    "current 0.5/3.5/1.0",
    "balanced 0.5/3.0/0.75",
]

CONTROLLER_LABELS = {
    "nominal": "Nominal CUDA",
    "default g1.0": "Default learned",
    "current 0.5/3.5/1.0": "Efficiency",
    "balanced 0.5/3.0/0.75": "Balanced",
}

CONTROLLER_PARAMS = {
    "nominal": "no FDM",
    "default g1.0": "residual_gain=1.0",
    "current 0.5/3.5/1.0": "residual_gain=0.5, goal_xy_weight=3.5, smooth_weight=1.0",
    "balanced 0.5/3.0/0.75": "residual_gain=0.5, goal_xy_weight=3.0, smooth_weight=0.75",
}

COLORS = {
    "nominal": "#1f77b4",
    "default g1.0": "#ff7f0e",
    "current 0.5/3.5/1.0": "#2ca02c",
    "balanced 0.5/3.0/0.75": "#9467bd",
}

SCENARIO_LABELS = {
    "id_random": "ID random",
    "ood_obstacle": "OOD obstacle",
    "ood_terrain": "OOD terrain",
}

METRICS = [
    ("final_distance", "Final distance", "m", "lower"),
    ("steps", "Steps", "steps", "lower"),
    ("min_obstacle_clearance", "Clearance", "m", "higher"),
    ("mean_terrain_risk", "Terrain risk", "", "lower"),
    ("control_smoothness", "Smoothness", "", "lower"),
    ("control_jerk", "Jerk", "", "lower"),
    ("mean_mppi_time_ms", "Runtime", "ms", "lower"),
]

TABLE_METRICS = [
    ("success_rate", "success"),
    ("final_distance_mean", "final_distance"),
    ("steps_mean", "steps"),
    ("min_obstacle_clearance_mean", "clearance"),
    ("mean_terrain_risk_mean", "terrain_risk"),
    ("control_smoothness_mean", "smoothness"),
    ("control_jerk_mean", "jerk"),
    ("mean_mppi_time_ms_mean", "runtime_ms"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def load_summary(summary_path: Path) -> pd.DataFrame:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = data.get("errors", [])
    if errors:
        raise RuntimeError(f"Official summary has errors: {errors}")
    rows = data.get("rows", [])
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 official rows, got {len(rows)}")
    df = pd.DataFrame(rows)
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    df["controller_label"] = df["controller"].map(CONTROLLER_LABELS)
    df["controller_order"] = df["controller"].map({c: i for i, c in enumerate(CONTROLLER_ORDER)})
    return df.sort_values(["scenario", "controller_order"]).reset_index(drop=True)


def load_episode_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in summary_df.to_dict("records"):
        run_summary = Path(row["source_dir"]) / "stage5_benchmark_summary.json"
        data = json.loads(run_summary.read_text(encoding="utf-8"))
        run_df = pd.DataFrame(data["runs"])
        run_df["scenario_key"] = row["scenario"]
        run_df["scenario_label"] = SCENARIO_LABELS[row["scenario"]]
        run_df["controller_key"] = row["controller"]
        run_df["controller_label"] = CONTROLLER_LABELS[row["controller"]]
        run_df["source_dir"] = row["source_dir"]
        frames.append(run_df)
    return pd.concat(frames, ignore_index=True)


def compute_paired_deltas(run_df: pd.DataFrame) -> pd.DataFrame:
    nominal = run_df[run_df["controller_key"] == "nominal"].copy()
    learned = run_df[run_df["controller_key"] != "nominal"].copy()
    keys = ["scenario_key", "episode_id"]
    merged = learned.merge(
        nominal[keys + [m[0] for m in METRICS]],
        on=keys,
        suffixes=("", "_nominal"),
        validate="many_to_one",
    )
    for metric, _, _, _ in METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_nominal"]
    return merged


def write_table_files(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    path_base.with_suffix(".md").write_text(dataframe_to_markdown(df), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_tables(summary_df: pd.DataFrame, paired_df: pd.DataFrame, table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)

    main = summary_df[["scenario", "controller", *[c for c, _ in TABLE_METRICS]]].copy()
    main = main.rename(columns=dict(TABLE_METRICS))
    for col in ["final_distance", "clearance", "terrain_risk", "smoothness", "jerk", "runtime_ms"]:
        main[col] = main[col].map(lambda x: f"{float(x):.6f}")
    main["steps"] = main["steps"].map(lambda x: f"{float(x):.1f}")
    main["success"] = main["success"].map(lambda x: f"{float(x):.2f}")
    write_table_files(main, table_dir / "stage5_main_results")

    delta_rows = []
    for controller in CONTROLLER_ORDER[1:]:
        sub = paired_df[paired_df["controller_key"] == controller]
        row = {
            "controller": CONTROLLER_LABELS[controller],
            "params": CONTROLLER_PARAMS[controller],
            "episodes": len(sub),
        }
        for metric, _, _, _ in METRICS:
            row[f"mean_delta_{metric}"] = sub[f"delta_{metric}"].mean()
            row[f"median_delta_{metric}"] = sub[f"delta_{metric}"].median()
            row[f"pct_improved_{metric}"] = improvement_rate(sub[f"delta_{metric}"], metric)
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    write_table_files(deltas, table_dir / "stage5_paired_delta_summary")

    modes = pd.DataFrame(
        [
            {
                "mode": "Conservative / smooth",
                "controller": "Default learned",
                "params": CONTROLLER_PARAMS["default g1.0"],
                "main_gain": "lower terrain risk, smoother control, lower jerk",
                "tradeoff": "worse final distance and more steps than nominal",
                "paper_use": "risk/smoothness operating point",
            },
            {
                "mode": "Efficiency",
                "controller": "Efficiency",
                "params": CONTROLLER_PARAMS["current 0.5/3.5/1.0"],
                "main_gain": "best official final-distance and steps gains",
                "tradeoff": "small terrain-risk, smoothness, and jerk penalties",
                "paper_use": "navigation efficiency operating point",
            },
            {
                "mode": "Balanced",
                "controller": "Balanced",
                "params": CONTROLLER_PARAMS["balanced 0.5/3.0/0.75"],
                "main_gain": "steps improve while final distance and risk stay near nominal",
                "tradeoff": "small smoothness and jerk penalties",
                "paper_use": "balanced operating-point candidate",
            },
        ]
    )
    write_table_files(modes, table_dir / "stage5_operating_modes")

    runtime = summary_df[
        ["scenario", "controller", "mean_mppi_time_ms_mean", "mean_mppi_time_ms_delta"]
    ].rename(
        columns={
            "mean_mppi_time_ms_mean": "runtime_ms",
            "mean_mppi_time_ms_delta": "delta_runtime_ms",
        }
    )
    runtime["runtime_ms"] = runtime["runtime_ms"].map(lambda x: f"{float(x):.3f}")
    runtime["delta_runtime_ms"] = runtime["delta_runtime_ms"].map(lambda x: f"{float(x):+.3f}")
    write_table_files(runtime, table_dir / "stage5_runtime")

    paired_df.to_csv(table_dir / "stage5_paired_episode_deltas.csv", index=False)
    return {
        "main": table_dir / "stage5_main_results.csv",
        "deltas": table_dir / "stage5_paired_delta_summary.csv",
        "modes": table_dir / "stage5_operating_modes.csv",
        "runtime": table_dir / "stage5_runtime.csv",
        "paired": table_dir / "stage5_paired_episode_deltas.csv",
    }


def improvement_rate(delta: pd.Series, metric: str) -> float:
    if metric == "min_obstacle_clearance":
        return float((delta >= 0).mean())
    return float((delta <= 0).mean())


def save_main_result_bars(summary_df: pd.DataFrame, figure_dir: Path) -> Path:
    metrics = METRICS[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Stage 5-D main results, 50 episodes per scenario", fontweight="bold")
    scenarios = list(SCENARIO_LABELS)
    x = np.arange(len(scenarios))
    width = 0.18
    for ax, (metric, title, unit, direction) in zip(axes.flat, metrics):
        for idx, controller in enumerate(CONTROLLER_ORDER):
            vals = []
            for scenario in scenarios:
                row = summary_df[(summary_df["scenario"] == scenario) & (summary_df["controller"] == controller)].iloc[0]
                vals.append(row[f"{metric}_mean"])
            ax.bar(x + (idx - 1.5) * width, vals, width, color=COLORS[controller], label=CONTROLLER_LABELS[controller])
        subtitle = "lower is better" if direction == "lower" else "higher is better"
        ax.set_title(f"{title} ({subtitle})")
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=15, ha="right")
        if unit:
            ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    out = figure_dir / "stage5_main_result_bars.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_paired_delta_boxplots(paired_df: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    fig.suptitle("Paired learned-minus-nominal episode deltas", fontweight="bold")
    for ax, (metric, title, unit, direction) in zip(axes.flat, METRICS):
        values = []
        labels = []
        colors = []
        for controller in CONTROLLER_ORDER[1:]:
            sub = paired_df[paired_df["controller_key"] == controller]
            values.append(sub[f"delta_{metric}"].to_numpy())
            labels.append(CONTROLLER_LABELS[controller])
            colors.append(COLORS[controller])
        bp = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.3)
        ax.axhline(0.0, color="#111111", linewidth=0.9)
        subtitle = "negative is better" if direction == "lower" else "positive is better"
        ax.set_title(f"Delta {title}\n({subtitle})")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        if unit:
            ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    out = figure_dir / "stage5_paired_delta_boxplots.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_pareto_scatter(summary_df: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.7), constrained_layout=True)
    markers = {"id_random": "o", "ood_obstacle": "s", "ood_terrain": "^"}
    learned = summary_df[summary_df["controller"] != "nominal"]
    for _, row in learned.iterrows():
        ax.scatter(
            row["steps_delta"],
            row["mean_terrain_risk_delta"],
            s=130,
            marker=markers[row["scenario"]],
            color=COLORS[row["controller"]],
            edgecolor="#111111",
            linewidth=0.8,
        )
    ax.axvline(0.0, color="#111111", linewidth=0.9)
    ax.axhline(0.0, color="#111111", linewidth=0.9)
    ax.set_xlabel("Delta steps vs nominal (negative is faster)")
    ax.set_ylabel("Delta terrain risk vs nominal (negative is lower risk)")
    ax.set_title("Stage 5-D Pareto trade-off")
    ax.grid(alpha=0.25)
    controller_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=COLORS[c], markeredgecolor="#111111", label=CONTROLLER_LABELS[c], markersize=9)
        for c in CONTROLLER_ORDER[1:]
    ]
    scenario_handles = [
        plt.Line2D([0], [0], marker=m, linestyle="None", color="#111111", label=SCENARIO_LABELS[s], markersize=8)
        for s, m in markers.items()
    ]
    first = ax.legend(handles=controller_handles, title="Controller", loc="center", bbox_to_anchor=(0.63, 0.58))
    ax.add_artist(first)
    ax.legend(handles=scenario_handles, title="Scenario", loc="center", bbox_to_anchor=(0.63, 0.32))
    out = figure_dir / "stage5_pareto_scatter.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_runtime_bars(summary_df: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    scenarios = list(SCENARIO_LABELS)
    x = np.arange(len(scenarios))
    width = 0.18
    for idx, controller in enumerate(CONTROLLER_ORDER):
        vals = []
        for scenario in scenarios:
            row = summary_df[(summary_df["scenario"] == scenario) & (summary_df["controller"] == controller)].iloc[0]
            vals.append(row["mean_mppi_time_ms_mean"])
        ax.bar(x + (idx - 1.5) * width, vals, width, color=COLORS[controller], label=CONTROLLER_LABELS[controller])
    ax.axhline(50.0, color="#7f7f7f", linestyle="--", linewidth=1.1, label="50 ms reference")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("ms")
    ax.set_title("Mean MPPI runtime per step")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    out = figure_dir / "stage5_runtime_bars.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def read_trajectory(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def choose_typical_episodes(paired_df: pd.DataFrame) -> dict[str, int]:
    choices = {}
    current = paired_df[paired_df["controller_key"] == "current 0.5/3.5/1.0"]
    for scenario, sub in current.groupby("scenario_key"):
        median_delta = sub["delta_steps"].median()
        idx = (sub["delta_steps"] - median_delta).abs().idxmin()
        choices[scenario] = int(sub.loc[idx, "episode_id"])
    return choices


def run_path(run_df: pd.DataFrame, scenario: str, controller: str, episode_id: int) -> Path:
    row = run_df[
        (run_df["scenario_key"] == scenario)
        & (run_df["controller_key"] == controller)
        & (run_df["episode_id"] == episode_id)
    ].iloc[0]
    return Path(row["results_path"])


def load_obstacles(config_path: Path) -> list[tuple[float, float, float]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    obstacles = []
    for item in config.get("obstacles", {}).get("virtual", []):
        if len(item) >= 3:
            obstacles.append((float(item[0]), float(item[1]), float(item[2])))
    return obstacles


def save_trajectory_gallery(run_df: pd.DataFrame, paired_df: pd.DataFrame, figure_dir: Path) -> Path:
    choices = choose_typical_episodes(paired_df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("Typical trajectory gallery", fontweight="bold")
    for ax, scenario in zip(axes, SCENARIO_LABELS):
        episode = choices[scenario]
        trajectories = {}
        for controller in CONTROLLER_ORDER:
            path = run_path(run_df, scenario, controller, episode)
            trajectories[controller] = read_trajectory(path / "trajectory.csv")
        all_x = np.concatenate([df["x"].to_numpy() for df in trajectories.values()])
        all_y = np.concatenate([df["y"].to_numpy() for df in trajectories.values()])
        goal_x = float(next(iter(trajectories.values()))["x_des"].iloc[0])
        goal_y = float(next(iter(trajectories.values()))["y_des"].iloc[0])
        start_x = float(next(iter(trajectories.values()))["x"].iloc[0])
        start_y = float(next(iter(trajectories.values()))["y"].iloc[0])
        xmin = min(float(all_x.min()), goal_x, start_x) - 3.0
        xmax = max(float(all_x.max()), goal_x, start_x) + 3.0
        ymin = min(float(all_y.min()), goal_y, start_y) - 3.0
        ymax = max(float(all_y.max()), goal_y, start_y) + 3.0

        config_path = run_path(run_df, scenario, "nominal", episode) / "config.yaml"
        for ox, oy, radius in load_obstacles(config_path):
            if xmin - radius <= ox <= xmax + radius and ymin - radius <= oy <= ymax + radius:
                circle = plt.Circle((ox, oy), radius, color="#9ca3af", alpha=0.28, linewidth=0)
                ax.add_patch(circle)

        for controller, df in trajectories.items():
            ax.plot(df["x"], df["y"], color=COLORS[controller], linewidth=2.0, label=CONTROLLER_LABELS[controller])
        ax.scatter([start_x], [start_y], marker="o", s=70, color="#111111", label="Start")
        ax.scatter([goal_x], [goal_y], marker="*", s=130, color="#f59e0b", edgecolor="#111111", linewidth=0.5, label="Goal")
        ax.set_title(f"{SCENARIO_LABELS[scenario]} episode {episode:04d}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes[0].legend(fontsize=7, loc="best")
    out = figure_dir / "stage5_trajectory_gallery.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_tradeoff_summary(summary_df: pd.DataFrame, paired_df: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.axis("off")
    lines = [
        ("Default learned", "Conservative/smooth", "Risk, smoothness, jerk improve; final distance and steps regress."),
        ("Efficiency", "Fastest operating point", "Best official final-distance and steps gains; small risk/smoothness/jerk cost."),
        ("Balanced", "Balanced candidate", "Steps improve; final distance and terrain risk stay close to nominal."),
    ]
    ax.set_title("Failure and trade-off analysis", fontweight="bold", loc="left")
    y = 0.86
    for idx, (name, role, text) in enumerate(lines):
        controller = CONTROLLER_ORDER[idx + 1]
        ax.text(0.02, y, name, color=COLORS[controller], fontweight="bold", fontsize=12, transform=ax.transAxes)
        ax.text(0.26, y, role, color="#111111", fontweight="bold", fontsize=11, transform=ax.transAxes)
        ax.text(0.02, y - 0.09, text, color="#374151", fontsize=10, transform=ax.transAxes)
        y -= 0.25
    ax.text(
        0.02,
        0.08,
        "Boundary: S5-010 supports multiple calibrated operating modes, not a learned controller that dominates all metrics.",
        color="#111111",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    out = figure_dir / "stage5_failure_tradeoff_analysis.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def build_figures(summary_df: pd.DataFrame, run_df: pd.DataFrame, paired_df: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "main": save_main_result_bars(summary_df, figure_dir),
        "paired": save_paired_delta_boxplots(paired_df, figure_dir),
        "pareto": save_pareto_scatter(summary_df, figure_dir),
        "runtime": save_runtime_bars(summary_df, figure_dir),
        "gallery": save_trajectory_gallery(run_df, paired_df, figure_dir),
        "tradeoff": save_tradeoff_summary(summary_df, paired_df, figure_dir),
    }


def write_stage6_doc(summary_df: pd.DataFrame, paired_df: pd.DataFrame, doc_path: Path, figure_dir: Path, table_dir: Path, zip_path: Path) -> None:
    mode_stats = []
    for controller in CONTROLLER_ORDER[1:]:
        sub = paired_df[paired_df["controller_key"] == controller]
        mode_stats.append(
            {
                "controller": CONTROLLER_LABELS[controller],
                "delta_final": sub["delta_final_distance"].mean(),
                "delta_steps": sub["delta_steps"].mean(),
                "delta_risk": sub["delta_mean_terrain_risk"].mean(),
                "delta_smooth": sub["delta_control_smoothness"].mean(),
                "delta_jerk": sub["delta_control_jerk"].mean(),
                "delta_runtime": sub["delta_mean_mppi_time_ms"].mean(),
            }
        )
    stats_df = pd.DataFrame(mode_stats)
    for col in [c for c in stats_df.columns if c.startswith("delta_")]:
        stats_df[col] = stats_df[col].map(lambda x: f"{float(x):.6f}")
    stats_md = dataframe_to_markdown(stats_df)
    content = f"""# Stage 6 Result Package: Stage 5-D Paper-Ready Results

## Scope

S6-001 packages the S5-010 50-episode ID/OOD benchmark into paper-ready figures, tables, and a compressed archive. It does not add a new controller or tune new parameters.

Source result:

```text
results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json
```

Official matrix:

- ID random, OOD obstacle, OOD terrain
- nominal CUDA
- default learned `residual_gain=1.0`
- efficiency learned `residual_gain=0.5`, `goal_xy_weight=3.5`, `smooth_weight=1.0`
- balanced learned `residual_gain=0.5`, `goal_xy_weight=3.0`, `smooth_weight=0.75`
- `50` episodes per controller/scenario

## Generated Outputs

Figures:

```text
{figure_dir}
```

Tables:

```text
{table_dir}
```

Zip package:

```text
{zip_path}
```

## Main Result Framing

S5-010 supports multiple calibrated learned-FDM-MPPI operating modes, not a single learned controller that dominates every metric.

{stats_md}

## Figure Inventory

- `stage5_main_result_bars.png`: mean metric comparison across scenarios and controllers.
- `stage5_paired_delta_boxplots.png`: per-episode learned-minus-nominal deltas for final distance, steps, clearance, terrain risk, smoothness, jerk, and runtime.
- `stage5_pareto_scatter.png`: steps-vs-terrain-risk Pareto view.
- `stage5_trajectory_gallery.png`: representative paired trajectories for the three scenarios.
- `stage5_runtime_bars.png`: runtime comparison.
- `stage5_failure_tradeoff_analysis.png`: compact trade-off summary.

## Table Inventory

- `stage5_main_results.csv/md`
- `stage5_operating_modes.csv/md`
- `stage5_paired_delta_summary.csv/md`
- `stage5_paired_episode_deltas.csv`
- `stage5_runtime.csv/md`

## Interpretation

- Default learned `g=1.0` is the conservative/smooth mode. It reduces terrain risk, smoothness, and jerk, but regresses final distance and steps.
- Efficiency `0.5/3.5/1.0` is the strongest official efficiency operating point. It improves final distance and steps most, with small risk/smoothness/jerk penalties.
- Balanced `0.5/3.0/0.75` is a balanced operating-point candidate. It modestly improves steps while keeping final distance and terrain risk close to nominal.
- Learned runtime remains much slower than nominal CUDA, so runtime profiling remains a separate Stage 6/Stage 7 requirement before real-time claims.

## Reproduction

```bash
python3 tools/plot_stage5_results.py
```
"""
    doc_path.write_text(content, encoding="utf-8")


def build_zip(zip_path: Path, files: list[Path], summary_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, path.as_posix())
        if summary_path.exists():
            zf.write(summary_path, f"source/{summary_path.name}")
            csv_path = summary_path.with_suffix(".csv")
            if csv_path.exists():
                zf.write(csv_path, f"source/{csv_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/stage5"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables/stage5"))
    parser.add_argument("--doc", type=Path, default=Path("docs/agent_memory/STAGE6_RESULT_PACKAGE.md"))
    parser.add_argument("--zip", type=Path, default=Path("results/stage6_result_package/stage5_result_package.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    summary_df = load_summary(args.summary)
    run_df = load_episode_runs(summary_df)
    paired_df = compute_paired_deltas(run_df)
    table_paths = build_tables(summary_df, paired_df, args.table_dir)
    figure_paths = build_figures(summary_df, run_df, paired_df, args.figure_dir)
    write_stage6_doc(summary_df, paired_df, args.doc, args.figure_dir, args.table_dir, args.zip)

    manifest_path = args.table_dir / "stage5_result_package_manifest.json"
    manifest = {
        "summary": str(args.summary),
        "figures": {key: str(path) for key, path in figure_paths.items()},
        "tables": {key: str(path) for key, path in table_paths.items()},
        "doc": str(args.doc),
        "zip": str(args.zip),
        "official_rows": int(len(summary_df)),
        "paired_delta_rows": int(len(paired_df)),
        "controllers": CONTROLLER_ORDER,
        "scenarios": list(SCENARIO_LABELS),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    package_files = list(dict.fromkeys([
        *sorted(args.figure_dir.glob("*")),
        *sorted(args.table_dir.glob("*")),
        args.doc,
        manifest_path,
    ]))
    build_zip(args.zip, package_files, args.summary)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
