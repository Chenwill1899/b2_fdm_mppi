#!/usr/bin/env python3
"""Build Nature-style paper figures for the FDM-MPPI result package."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


STAGE4_SEED_SUMMARY = Path("results/fdm_baselines/stage4_seed_benchmark_summary.json")
STAGE4_ROLLOUT = Path("results/fdm_rollout_eval/stage4_mlp_seed123_hardened_b2_omni_oracle_seed123")
OOD_OBSTACLE = Path("results/fdm_ood_eval/ood_obstacle_seed123")
OOD_TERRAIN = Path("results/fdm_ood_eval/ood_terrain_seed123")
STAGE5_SUMMARY = Path("results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json")
STAGE5_DELTAS = Path("archive/tables/stage5/stage5_paired_episode_deltas.csv")
STAGE5E_STATS = Path("archive/tables/stage5_e/table_s5e_paired_stats.csv")
STAGE5E_MAIN = Path("archive/tables/stage5_e/table_s5e_main_results.csv")
STAGE5E_VISUAL = Path("results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123/stage5_e_visual_eval_summary.json")

PALETTE = {
    "ink": "#263238",
    "muted": "#6E7B8B",
    "nominal": "#8792A2",
    "oracle": "#1F2933",
    "default": "#8FA7D6",
    "efficiency": "#E6A15F",
    "balanced": "#4F7DB8",
    "risk": "#69A889",
    "risk2": "#8B73B8",
    "accent": "#CC5A57",
    "light": "#EEF2F5",
    "grid": "#DDE3EA",
}

CONTROLLER_ORDER = [
    "nominal",
    "default g1.0",
    "current 0.5/3.5/1.0",
    "balanced 0.5/3.0/0.75",
]
CONTROLLER_LABELS = {
    "nominal": "Nominal",
    "default g1.0": "Default",
    "current 0.5/3.5/1.0": "Efficiency",
    "balanced 0.5/3.0/0.75": "Balanced",
}
CONTROLLER_COLORS = {
    "nominal": PALETTE["nominal"],
    "default g1.0": PALETTE["default"],
    "current 0.5/3.5/1.0": PALETTE["efficiency"],
    "balanced 0.5/3.0/0.75": PALETTE["balanced"],
}
SCENARIO_LABELS = {
    "id_random": "ID\nrandom",
    "ood_obstacle": "OOD\nobstacle",
    "ood_terrain": "OOD\nterrain",
    "low_friction_patch": "Low-friction\npatch",
    "safe_corridor": "Safe\ncorridor",
    "risk_band": "Risk band",
    "two_obstacle_standard": "Two-obstacle\nstandard",
}
RISK_SCENARIO_LABELS = {
    "low_friction_patch": "Low\nfriction",
    "safe_corridor": "Safe\ncorridor",
    "risk_band": "Risk\nband",
    "two_obstacle_standard": "Two\nobstacle",
}


def plot_paper_figures(
    *,
    output_dir: str | Path = "figures/paper",
    tables_dir: str | Path = "tables/paper",
    package_dir: str | Path = "results/final_result_package",
    package_name: str = "b2_fdm_mppi_paper_figures_20260503.zip",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tables_dir = Path(tables_dir)
    package_dir = Path(package_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)
    _apply_nature_style()

    manifest: dict[str, Any] = {
        "figure_dir": str(output_dir),
        "tables_dir": str(tables_dir),
        "sources": {},
        "figures": [],
        "tables": [],
        "package": str(package_dir / package_name),
        "style_reference": "https://github.com/Yuan1z0825/nature-skills/tree/main/nature-figure",
        "runtime_claim": "Runtime is intentionally excluded from main figures because learned runtime is not improved.",
    }

    stage4 = _load_stage4_bundle()
    stage5_summary = _load_stage5_summary()
    stage5_deltas = pd.read_csv(STAGE5_DELTAS)
    stage5e_stats = pd.read_csv(STAGE5E_STATS)
    stage5e_main = pd.read_csv(STAGE5E_MAIN)
    visual = _read_json(STAGE5E_VISUAL)

    manifest["sources"] = {
        "stage4_seed_summary": str(STAGE4_SEED_SUMMARY),
        "stage4_rollout_metrics": str(STAGE4_ROLLOUT / "rollout_metrics.json"),
        "stage4_rollout_replay": str(STAGE4_ROLLOUT / "rollout_replay.npz"),
        "ood_obstacle_metrics": str(OOD_OBSTACLE / "ood_residual_metrics.json"),
        "ood_terrain_metrics": str(OOD_TERRAIN / "ood_residual_metrics.json"),
        "stage5_summary": str(STAGE5_SUMMARY),
        "stage5_paired_deltas": str(STAGE5_DELTAS),
        "stage5e_paired_stats": str(STAGE5E_STATS),
        "stage5e_main_results": str(STAGE5E_MAIN),
        "stage5e_visual_summary": str(STAGE5E_VISUAL),
    }

    manifest["tables"].extend(_write_paper_tables(stage4, stage5_summary, stage5_deltas, stage5e_stats, stage5e_main, tables_dir))
    manifest["figures"].extend(_plot_overview(output_dir))
    manifest["figures"].extend(_plot_model_fidelity(stage4, output_dir))
    manifest["figures"].extend(_plot_closed_loop_modes(stage5_summary, stage5_deltas, output_dir))
    manifest["figures"].extend(_plot_risk_aware(stage5e_stats, visual, output_dir))

    manifest_path = output_dir / "paper_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _package_outputs(output_dir, tables_dir, package_dir / package_name)
    return manifest


def _apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_stage4_bundle() -> dict[str, Any]:
    return {
        "seed_summary": _read_json(STAGE4_SEED_SUMMARY),
        "rollout_metrics": _read_json(STAGE4_ROLLOUT / "rollout_metrics.json"),
        "rollout_replay": np.load(STAGE4_ROLLOUT / "rollout_replay.npz"),
        "ood_obstacle_metrics": _read_json(OOD_OBSTACLE / "ood_residual_metrics.json"),
        "ood_obstacle_rollout": _read_json(OOD_OBSTACLE / "ood_rollout_metrics.json"),
        "ood_terrain_metrics": _read_json(OOD_TERRAIN / "ood_residual_metrics.json"),
        "ood_terrain_rollout": _read_json(OOD_TERRAIN / "ood_rollout_metrics.json"),
    }


def _load_stage5_summary() -> pd.DataFrame:
    data = _read_json(STAGE5_SUMMARY)
    errors = data.get("errors") or []
    if errors:
        raise RuntimeError(f"Stage 5 summary contains errors: {errors}")
    df = pd.DataFrame(data["rows"])
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    df["controller_label"] = df["controller"].map(CONTROLLER_LABELS)
    df["controller_order"] = df["controller"].map({name: idx for idx, name in enumerate(CONTROLLER_ORDER)})
    return df.sort_values(["scenario", "controller_order"]).reset_index(drop=True)


def _write_paper_tables(
    stage4: dict[str, Any],
    stage5_summary: pd.DataFrame,
    stage5_deltas: pd.DataFrame,
    stage5e_stats: pd.DataFrame,
    stage5e_main: pd.DataFrame,
    tables_dir: Path,
) -> list[str]:
    written: list[str] = []
    rows = []
    seed_mean = stage4["seed_summary"]["mean"]
    seed_std = stage4["seed_summary"]["std"]
    rollout = stage4["rollout_metrics"]
    rows.append(
        {
            "setting": "ID mean seeds 123/456/789",
            "test_mse": seed_mean["test_mse"],
            "test_mse_std": seed_std["test_mse"],
            "zero_residual_test_mse": seed_mean["zero_residual_test_mse"],
            "test_mse_reduction_pct": seed_mean["overall_test_mse_reduction_pct"],
            "test_improvement_x": seed_mean["overall_test_improvement_x"],
            "learned_ade": rollout["learned_ade_xy"],
            "nominal_ade": rollout["nominal_ade_xy"],
            "learned_fde": rollout["learned_fde_xy"],
            "nominal_fde": rollout["nominal_fde_xy"],
        }
    )
    for label, metrics, rollout_metrics in (
        ("OOD obstacle", stage4["ood_obstacle_metrics"], stage4["ood_obstacle_rollout"]),
        ("OOD terrain", stage4["ood_terrain_metrics"], stage4["ood_terrain_rollout"]),
    ):
        rows.append(
            {
                "setting": label,
                "test_mse": metrics["test_mse"],
                "test_mse_std": "",
                "zero_residual_test_mse": metrics["zero_residual_test_mse"],
                "test_mse_reduction_pct": metrics["overall_test_mse_reduction_pct"],
                "test_improvement_x": metrics["overall_test_improvement_x"],
                "learned_ade": rollout_metrics["learned_ade_xy"],
                "nominal_ade": rollout_metrics["nominal_ade_xy"],
                "learned_fde": rollout_metrics["learned_fde_xy"],
                "nominal_fde": rollout_metrics["nominal_fde_xy"],
            }
        )
    written.append(_write_csv(tables_dir / "paper_stage4_model_metrics.csv", rows))

    written.append(_write_csv(tables_dir / "paper_stage5_closed_loop_summary.csv", stage5_summary.drop(columns=["controller_order"]).to_dict("records")))
    delta_cols = [
        "scenario_key",
        "controller_label",
        "episode_id",
        "delta_final_distance",
        "delta_steps",
        "delta_mean_terrain_risk",
        "delta_control_smoothness",
        "delta_control_jerk",
        "delta_min_obstacle_clearance",
    ]
    written.append(_write_csv(tables_dir / "paper_stage5_paired_deltas.csv", stage5_deltas[delta_cols].to_dict("records")))
    written.append(_write_csv(tables_dir / "paper_stage5e_main_results.csv", stage5e_main.to_dict("records")))
    written.append(_write_csv(tables_dir / "paper_stage5e_paired_stats.csv", stage5e_stats.to_dict("records")))
    return written


def _plot_overview(output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.set_axis_off()
    stages = [
        ("Oracle residual\nworld", "executed-control\ndeviation"),
        ("Episode dataset", "ID + OOD\nsplits"),
        ("Residual FDM", "one-step\nMLP"),
        ("Learned-FDM-MPPI", "closed-loop\nplanner"),
        ("Risk-aware MPPI", "terrain-risk\nobjective"),
    ]
    x_positions = np.linspace(0.08, 0.84, len(stages))
    for idx, ((title, subtitle), x) in enumerate(zip(stages, x_positions)):
        box = FancyBboxPatch(
            (x, 0.48),
            0.13,
            0.26,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=0.8,
            edgecolor=PALETTE["ink"],
            facecolor="#F8FAFC",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x + 0.065, 0.64, title, ha="center", va="center", fontsize=7.3, weight="bold", transform=ax.transAxes)
        ax.text(x + 0.065, 0.54, subtitle, ha="center", va="center", fontsize=6.2, color="#4B5563", transform=ax.transAxes)
        if idx < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.135, 0.61),
                    (x_positions[idx + 1] - 0.005, 0.61),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color=PALETTE["ink"],
                    transform=ax.transAxes,
                )
            )
    scene_y = 0.19
    for x, label, color in (
        (0.12, "ID random", "#C7D2FE"),
        (0.30, "OOD obstacle", "#FED7AA"),
        (0.50, "OOD terrain", "#BBF7D0"),
        (0.71, "Risk fields", "#E9D5FF"),
    ):
        ax.add_patch(plt.Circle((x, scene_y), 0.045, color=color, ec="#333333", lw=0.7, transform=ax.transAxes))
        ax.text(x, scene_y - 0.085, label, ha="center", va="center", fontsize=7, transform=ax.transAxes)
    ax.text(0.03, 0.93, "a", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.88, "FDM-MPPI evaluation pipeline", fontsize=10, weight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.10, "Main claims: residual fidelity, open-loop prediction, closed-loop operating points, risk exposure reduction.", fontsize=7.5, color="#374151", transform=ax.transAxes)
    return _save_figure(fig, output_dir / "fig1_method_and_experiment_suite")


def _plot_model_fidelity(stage4: dict[str, Any], output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0), constrained_layout=True)
    seed_summary = stage4["seed_summary"]
    settings = [
        ("ID mean", seed_summary["mean"]["test_mse"], seed_summary["mean"]["zero_residual_test_mse"]),
        ("OOD obs.", stage4["ood_obstacle_metrics"]["test_mse"], stage4["ood_obstacle_metrics"]["zero_residual_test_mse"]),
        ("OOD terr.", stage4["ood_terrain_metrics"]["test_mse"], stage4["ood_terrain_metrics"]["zero_residual_test_mse"]),
    ]
    ax = axes[0, 0]
    x = np.arange(len(settings))
    width = 0.34
    ax.bar(x - width / 2, [s[2] for s in settings], width, color="#CBD3DD", label="Zero residual")
    ax.bar(x + width / 2, [s[1] for s in settings], width, color=PALETTE["balanced"], label="Residual FDM")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in settings])
    ax.set_ylabel("Test MSE")
    ax.legend(loc="upper right", handlelength=1.1)
    _style_axis(ax)
    _panel(ax, "a", "Residual model error")

    ax = axes[0, 1]
    axes_names = ["vx", "vy", "wz"]
    ax.bar(axes_names, [seed_summary["mean"][f"test_rmse_{name}"] for name in axes_names], color=[PALETTE["balanced"], "#80A9C4", PALETTE["risk"]])
    ax.set_ylabel("Test RMSE")
    _style_axis(ax)
    _panel(ax, "b", "Per-axis residual RMSE")

    replay = stage4["rollout_replay"]
    ax = axes[1, 0]
    ax.plot(replay["oracle_states"][:, 0], replay["oracle_states"][:, 1], color=PALETTE["oracle"], lw=1.7, label="Oracle execution")
    ax.plot(replay["nominal_states"][:, 0], replay["nominal_states"][:, 1], color=PALETTE["nominal"], lw=1.35, label="Nominal replay")
    ax.plot(replay["learned_states"][:, 0], replay["learned_states"][:, 1], color=PALETTE["balanced"], lw=1.55, label="Learned replay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    y_values = np.concatenate([replay["oracle_states"][:, 1], replay["nominal_states"][:, 1], replay["learned_states"][:, 1]])
    y_pad = max(0.25, 0.08 * float(np.ptp(y_values)))
    ax.set_ylim(float(np.min(y_values) - y_pad), float(np.max(y_values) + y_pad))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=3, handlelength=1.4, columnspacing=1.0)
    _style_axis(ax, grid=False)
    _panel(ax, "c", "Open-loop rollout replay")

    ax = axes[1, 1]
    rollout = stage4["rollout_metrics"]
    horizons = ["1s", "2s", "4s", "Full"]
    nominal = [
        rollout["nominal_ade_xy_at_1s"],
        rollout["nominal_ade_xy_at_2s"],
        rollout["nominal_ade_xy_at_4s"],
        rollout["nominal_ade_xy"],
    ]
    learned = [
        rollout["learned_ade_xy_at_1s"],
        rollout["learned_ade_xy_at_2s"],
        rollout["learned_ade_xy_at_4s"],
        rollout["learned_ade_xy"],
    ]
    x = np.arange(len(horizons))
    ax.bar(x - width / 2, nominal, width, color=PALETTE["nominal"], label="Nominal")
    ax.bar(x + width / 2, learned, width, color=PALETTE["balanced"], label="Learned")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("ADE (m)")
    ax.legend(loc="upper left", handlelength=1.1)
    _style_axis(ax)
    _panel(ax, "d", "Prediction error over horizon")
    return _save_figure(fig, output_dir / "fig2_residual_fidelity_open_loop")


def _plot_closed_loop_modes(summary: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> list[str]:
    fig = plt.figure(figsize=(7.8, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.18, 1.0, 1.0])
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.set_axis_off()
    metrics = [
        ("final_distance_mean", "Final distance (m)", "a"),
        ("steps_mean", "Steps", "b"),
        ("mean_terrain_risk_mean", "Terrain risk", "c"),
    ]
    top_handles = None
    top_labels = None
    for idx, (metric, ylabel, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, idx])
        _grouped_summary_bars(ax, summary, metric, ylabel, legend=False)
        if idx == 0:
            top_handles, top_labels = ax.get_legend_handles_labels()
        _panel(ax, label, ylabel)
    if top_handles and top_labels:
        legend_ax.legend(top_handles, top_labels, loc="center", ncol=4, handlelength=1.3, columnspacing=1.5)

    ax = fig.add_subplot(gs[2, 0])
    _paired_box(ax, deltas, "delta_final_distance", "Final distance\ndelta (m)")
    _panel(ax, "d", "Paired efficiency")

    ax = fig.add_subplot(gs[2, 1])
    _paired_box(ax, deltas, "delta_mean_terrain_risk", "Terrain risk\ndelta")
    _panel(ax, "e", "Risk trade-off")

    ax = fig.add_subplot(gs[2, 2])
    learned = summary[summary["controller"] != "nominal"].copy()
    markers = {"id_random": "o", "ood_obstacle": "s", "ood_terrain": "^"}
    for controller, sub in learned.groupby("controller", sort=False):
        for _, row in sub.iterrows():
            ax.scatter(
                row["steps_delta"],
                row["mean_terrain_risk_delta"],
                s=58,
                marker=markers.get(row["scenario"], "o"),
                color=CONTROLLER_COLORS[controller],
                edgecolor="#333333",
                linewidth=0.5,
                label=CONTROLLER_LABELS[controller] if row["scenario"] == "id_random" else None,
            )
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xlabel("Steps delta")
    ax.set_ylabel("Risk delta")
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor="#FFFFFF", markeredgecolor=PALETTE["ink"], label=SCENARIO_LABELS[key].replace("\n", " "))
        for key, marker in markers.items()
    ]
    ax.legend(handles=marker_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.4, borderaxespad=0.0)
    _style_axis(ax)
    _panel(ax, "f", "Operating-point Pareto")
    return _save_figure(fig, output_dir / "fig3_closed_loop_operating_modes")


def _plot_risk_aware(stats: pd.DataFrame, visual: dict[str, Any], output_dir: Path) -> list[str]:
    risk_on = stats[stats["terrain_risk_weight"] > 0].copy()
    selected = risk_on[risk_on["metric"].isin(["cumulative_terrain_risk", "terrain_risk_excess", "terrain_risk_exposure_ratio", "final_distance"])]
    fig = plt.figure(figsize=(7.8, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    _risk_delta_bars(ax, selected, "cumulative_terrain_risk", "Cumulative risk delta")
    _panel(ax, "a", "Cumulative risk")

    ax = fig.add_subplot(gs[0, 1])
    _risk_delta_bars(ax, selected, "terrain_risk_excess", "Excess risk delta")
    _panel(ax, "b", "Excess risk")

    ax = fig.add_subplot(gs[0, 2])
    _risk_delta_bars(ax, selected, "terrain_risk_exposure_ratio", "Exposure-ratio delta")
    _panel(ax, "c", "Exposure")

    ax = fig.add_subplot(gs[1, :])
    _plot_visual_trajectories(ax, visual)
    _panel(ax, "d", "Fixed two-obstacle trajectories")
    return _save_figure(fig, output_dir / "fig4_risk_aware_closed_loop")


def _grouped_summary_bars(ax: plt.Axes, summary: pd.DataFrame, metric: str, ylabel: str, *, legend: bool = False) -> None:
    scenarios = ["id_random", "ood_obstacle", "ood_terrain"]
    x = np.arange(len(scenarios))
    width = 0.18
    for idx, controller in enumerate(CONTROLLER_ORDER):
        sub = summary[summary["controller"] == controller].set_index("scenario")
        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            [sub.loc[scenario, metric] for scenario in scenarios],
            width,
            color=CONTROLLER_COLORS[controller],
            label=CONTROLLER_LABELS[controller],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    _style_axis(ax)
    if legend:
        ax.legend(fontsize=6, ncols=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))


def _paired_box(ax: plt.Axes, deltas: pd.DataFrame, metric: str, ylabel: str) -> None:
    controllers = [c for c in ["Default", "Efficiency", "Balanced"] if c in set(deltas["controller_label"])]
    values = [deltas[deltas["controller_label"] == c][metric].dropna().to_numpy() for c in controllers]
    box = ax.boxplot(values, labels=controllers, patch_artist=True, showfliers=False, widths=0.48)
    for patch, controller in zip(box["boxes"], controllers):
        key = next(k for k, v in CONTROLLER_LABELS.items() if v == controller)
        patch.set_facecolor(CONTROLLER_COLORS[key])
        patch.set_alpha(0.78)
    ax.axhline(0, color=PALETTE["ink"], lw=0.75, ls="--")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=0)
    _style_axis(ax)


def _risk_delta_bars(ax: plt.Axes, stats: pd.DataFrame, metric: str, ylabel: str) -> None:
    sub = stats[stats["metric"] == metric].copy()
    scenario_order = ["low_friction_patch", "safe_corridor", "risk_band", "two_obstacle_standard"]
    sub["order"] = sub["scenario_name"].map({name: idx for idx, name in enumerate(scenario_order)})
    sub = sub.sort_values("order")
    x = np.arange(len(sub))
    y = sub["mean_delta"].to_numpy(dtype=float)
    low = sub["bootstrap_ci95_low"].to_numpy(dtype=float)
    high = sub["bootstrap_ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - low, high - y])
    colors = [PALETTE["risk"], "#86BBA9", PALETTE["efficiency"], PALETTE["risk2"]][: len(sub)]
    ax.bar(x, y, color=colors, edgecolor=PALETTE["ink"], linewidth=0.45)
    ax.errorbar(x, y, yerr=yerr, fmt="none", color=PALETTE["ink"], capsize=1.8, lw=0.75)
    ax.axhline(0, color=PALETTE["ink"], lw=0.75, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([RISK_SCENARIO_LABELS.get(s, s) for s in sub["scenario_name"]], rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    _style_axis(ax)


def _plot_visual_trajectories(ax: plt.Axes, visual: dict[str, Any]) -> None:
    colors = {
        "nominal_risk_off": PALETTE["nominal"],
        "nominal_risk_on": PALETTE["risk"],
        "learned_risk_off": PALETTE["efficiency"],
        "learned_risk_on": PALETTE["balanced"],
    }
    labels = {
        "nominal_risk_off": "Nominal",
        "nominal_risk_on": "Nominal + risk",
        "learned_risk_off": "Learned",
        "learned_risk_on": "Learned + risk",
    }
    for key, run in visual.get("runs", {}).items():
        csv_path = Path(run["trajectory_csv"])
        if not csv_path.is_file():
            continue
        traj = pd.read_csv(csv_path)
        ax.plot(traj["x"], traj["y"], color=colors.get(key, PALETTE["ink"]), lw=1.35, label=labels.get(key, key))
    metadata = visual.get("metadata", {})
    ax.scatter([0.01], [0.01], s=22, color="#111827", marker="o", zorder=4)
    ax.scatter([18.0], [0.0], s=38, color=PALETTE["accent"], marker="*", zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, handlelength=1.5, columnspacing=1.2)
    ax.text(0.01, 0.03, f"risk weight={metadata.get('risk_weight', 3)}", transform=ax.transAxes, fontsize=6.2, color="#374151")
    _style_axis(ax, grid=False)


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    outputs = []
    for suffix, kwargs in ((".svg", {}), (".png", {"dpi": 300}), (".pdf", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontsize=8.0, pad=3)


def _style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.tick_params(axis="both", length=2.4, width=0.7, pad=2)
    if grid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _package_outputs(output_dir: Path, tables_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base in (output_dir, tables_dir):
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(base.parent))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-ready FDM-MPPI figures and tables")
    parser.add_argument("--output", default="figures/paper")
    parser.add_argument("--tables-output", default="tables/paper")
    parser.add_argument("--package-dir", default="results/final_result_package")
    parser.add_argument("--package-name", default="b2_fdm_mppi_paper_figures_20260503.zip")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = plot_paper_figures(
        output_dir=args.output,
        tables_dir=args.tables_output,
        package_dir=args.package_dir,
        package_name=args.package_name,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
