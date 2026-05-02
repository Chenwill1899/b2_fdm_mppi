#!/usr/bin/env python3
"""Run and visualize paired Stage 5 closed-loop nominal vs learned-FDM MPPI."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller
from tools.benchmark_learned_fdm_mppi import current_git_metadata, prepare_run_config, shell_join


COMPARISON_METRICS = (
    "reached_goal",
    "failed",
    "steps",
    "run_time",
    "final_distance",
    "path_length",
    "min_obstacle_clearance",
    "mean_terrain_risk",
    "max_terrain_risk",
    "cumulative_terrain_risk",
    "terrain_risk_excess",
    "terrain_risk_excess_integral",
    "terrain_risk_exposure_ratio",
    "mean_cmd_real_error",
    "mean_residual_norm",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
    "max_mppi_time_ms",
)


def run_visual_eval(
    *,
    config_path: str | Path,
    scenario_name: str,
    output_dir: str | Path,
    seed: int,
    backend: str = "cuda",
    fdm_model_dir: str | Path = "results/fdm_baselines/stage4_mlp_seed123_hardened",
    fdm_checkpoint: str | Path = "best_model.pt",
    fdm_normalization: str | Path = "normalization.npz",
    fdm_device: str | None = None,
    fdm_residual_gain: float = 1.0,
    command: str | None = None,
    argv: Sequence[str] | None = None,
    runner_cls=OmniMppiSimulationRunner,
) -> dict:
    backend = str(backend).lower()
    if backend not in {"numpy", "cuda", "torch"}:
        raise ValueError("Stage 5 visual eval supports only numpy, cuda, or torch backends")
    fdm_device = fdm_device or ("cuda" if backend == "cuda" else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    base_config = load_config(config_path)

    run_paths: dict[str, Path] = {}
    episode_id = 0
    for controller in ("nominal", "learned"):
        run_config = prepare_run_config(
            base_config,
            output_dir=output_dir,
            scenario_name=scenario_name,
            controller=controller,
            episode_id=episode_id,
            seed=seed,
            backend=backend,
            fdm_model_dir=fdm_model_dir,
            fdm_checkpoint=fdm_checkpoint,
            fdm_normalization=fdm_normalization,
            fdm_device=fdm_device,
            fdm_residual_gain=fdm_residual_gain,
        )
        run_config = _enable_visual_outputs(run_config)
        runner = runner_cls(
            run_config,
            controller_factory=lambda *, config, runner, seed=seed: create_omni_controller(
                config,
                seed=seed,
            ),
        )
        summary = runner.run()
        run_paths[controller] = Path(summary.results_path)

    comparison = write_closed_loop_comparison(
        nominal_results_path=run_paths["nominal"],
        learned_results_path=run_paths["learned"],
        config_path=config_path,
        output_dir=output_dir,
        scenario_name=scenario_name,
        seed=seed,
    )
    summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config": str(config_path),
            "scenario_name": str(scenario_name),
            "output_dir": str(output_dir),
            "seed": int(seed),
            "backend": backend,
            "fdm_model_dir": str(fdm_model_dir),
            "fdm_checkpoint": str(fdm_checkpoint),
            "fdm_normalization": str(fdm_normalization),
            "fdm_device": str(fdm_device),
            "fdm_residual_gain": float(fdm_residual_gain),
            **current_git_metadata(),
        },
        "runs": {
            "nominal": _run_output_record(run_paths["nominal"]),
            "learned": _run_output_record(run_paths["learned"]),
        },
        "comparison": comparison,
    }
    (output_dir / "stage5_visual_eval_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def run_risk_aware_visual_eval(
    *,
    config_path: str | Path,
    scenario_name: str,
    output_dir: str | Path,
    seed: int,
    backend: str = "torch",
    fdm_model_dir: str | Path = "results/fdm_baselines/stage4_mlp_seed123_hardened",
    fdm_checkpoint: str | Path = "best_model.pt",
    fdm_normalization: str | Path = "normalization.npz",
    fdm_device: str | None = None,
    fdm_residual_gain: float = 0.5,
    risk_weight: float = 3.0,
    risk_power: float = 2.0,
    risk_threshold: float = 0.3,
    risk_mode: str = "excess",
    learned_goal_xy_weight: float = 3.0,
    learned_smooth_weight: float = 0.75,
    command: str | None = None,
    argv: Sequence[str] | None = None,
    runner_cls=OmniMppiSimulationRunner,
) -> dict:
    backend = str(backend).lower()
    if backend not in {"numpy", "cuda", "torch"}:
        raise ValueError("Stage 5-E risk-aware visual eval supports only numpy, cuda, or torch backends")
    fdm_device = fdm_device or ("cuda" if backend == "cuda" else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    base_config = load_config(config_path)
    learned_overrides = {
        "goal_xy_weight": float(learned_goal_xy_weight),
        "smooth_weight": float(learned_smooth_weight),
    }
    cases = (
        ("nominal_risk_off", "nominal", 0.0, "Nominal-MPPI execution in oracle world"),
        ("nominal_risk_on", "nominal", float(risk_weight), "Risk-aware Nominal-MPPI execution in oracle world"),
        ("learned_risk_off", "learned", 0.0, "Learned-FDM-MPPI execution in oracle world"),
        (
            "learned_risk_on",
            "learned",
            float(risk_weight),
            "Risk-aware Learned-FDM-MPPI execution in oracle world",
        ),
    )

    runs = {}
    for case_name, controller, case_risk_weight, label in cases:
        run_config = prepare_run_config(
            base_config,
            output_dir=output_dir,
            scenario_name=f"{scenario_name}_{case_name}",
            controller=controller,
            episode_id=0,
            seed=seed,
            backend=backend,
            fdm_model_dir=fdm_model_dir,
            fdm_checkpoint=fdm_checkpoint,
            fdm_normalization=fdm_normalization,
            fdm_device=fdm_device,
            fdm_residual_gain=fdm_residual_gain,
            mppi_overrides={
                "terrain_risk_weight": float(case_risk_weight),
                "terrain_risk_power": float(risk_power),
                "terrain_risk_threshold": float(risk_threshold),
                "terrain_risk_mode": str(risk_mode).lower(),
            },
            learned_mppi_overrides=learned_overrides,
        )
        run_config = _enable_visual_outputs(run_config)
        runner = runner_cls(
            run_config,
            controller_factory=lambda *, config, runner, seed=seed: create_omni_controller(
                config,
                seed=seed,
            ),
        )
        summary = runner.run()
        results_path = Path(summary.results_path)
        runs[case_name] = {
            **_run_output_record(results_path),
            "controller": controller,
            "terrain_risk_weight": float(case_risk_weight),
            "label": label,
            "metrics": _read_json(results_path / "summary.json"),
        }

    summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config": str(config_path),
            "scenario_name": str(scenario_name),
            "output_dir": str(output_dir),
            "seed": int(seed),
            "backend": backend,
            "fdm_model_dir": str(fdm_model_dir),
            "fdm_checkpoint": str(fdm_checkpoint),
            "fdm_normalization": str(fdm_normalization),
            "fdm_device": str(fdm_device),
            "fdm_residual_gain": float(fdm_residual_gain),
            "risk_weight": float(risk_weight),
            "risk_power": float(risk_power),
            "risk_threshold": float(risk_threshold),
            "risk_mode": str(risk_mode).lower(),
            "learned_mppi_overrides": learned_overrides,
            **current_git_metadata(),
        },
        "runs": runs,
    }
    (output_dir / "stage5_e_visual_eval_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def write_closed_loop_comparison(
    *,
    nominal_results_path: str | Path,
    learned_results_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    scenario_name: str,
    seed: int,
) -> dict:
    nominal_results_path = Path(nominal_results_path)
    learned_results_path = Path(learned_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)

    nominal_summary = _read_json(nominal_results_path / "summary.json")
    learned_summary = _read_json(learned_results_path / "summary.json")
    metrics = _comparison_metrics(nominal_summary, learned_summary)
    overlay_png = output_dir / "closed_loop_nominal_vs_learned.png"
    metrics_csv = output_dir / "closed_loop_compare_metrics.csv"
    metrics_json = output_dir / "closed_loop_compare_metrics.json"

    _plot_closed_loop_overlay(
        nominal_results_path=nominal_results_path,
        learned_results_path=learned_results_path,
        config_path=config_path,
        output_path=overlay_png,
        nominal_summary=nominal_summary,
        learned_summary=learned_summary,
    )
    _write_metrics_csv(metrics_csv, metrics)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "scenario_name": str(scenario_name),
        "seed": int(seed),
        "nominal_results_path": str(nominal_results_path),
        "learned_results_path": str(learned_results_path),
        "overlay_png": str(overlay_png),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "metrics": metrics,
    }


def _enable_visual_outputs(config: dict) -> dict:
    updated = copy.deepcopy(config)
    updated.setdefault("results", {})["enable_plots"] = True
    updated.setdefault("results", {})["enable_animation"] = True
    return updated


def _run_output_record(results_path: Path) -> dict:
    return {
        "results_path": str(results_path),
        "summary_json": str(results_path / "summary.json"),
        "trajectory_csv": str(results_path / "trajectory.csv"),
        "trajectory_png": str(results_path / "trajectory.png"),
        "animation_gif": str(results_path / "animation.gif"),
    }


def _comparison_metrics(nominal_summary: dict, learned_summary: dict) -> dict:
    metrics = {}
    for metric in COMPARISON_METRICS:
        nominal = nominal_summary.get(metric)
        learned = learned_summary.get(metric)
        delta = _delta(learned, nominal)
        metrics[metric] = {
            "nominal": _json_scalar(nominal),
            "learned": _json_scalar(learned),
            "delta": _json_scalar(delta),
        }
    return metrics


def _plot_closed_loop_overlay(
    *,
    nominal_results_path: Path,
    learned_results_path: Path,
    config_path: Path,
    output_path: Path,
    nominal_summary: dict,
    learned_summary: dict,
) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    nominal = pd.read_csv(nominal_results_path / "trajectory.csv")
    learned = pd.read_csv(learned_results_path / "trajectory.csv")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    goal = np.asarray(config.get("simulation", {}).get("goal", [0.0, 0.0])[:2], dtype=float)
    start = np.asarray(config.get("simulation", {}).get("initial_state", [0.0, 0.0])[:2], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        nominal["x"],
        nominal["y"],
        color="tab:blue",
        linewidth=2.2,
        label=_trajectory_label("Nominal closed-loop", nominal_summary),
    )
    ax.plot(
        learned["x"],
        learned["y"],
        color="tab:orange",
        linewidth=2.2,
        label=_trajectory_label("Learned-FDM closed-loop", learned_summary),
    )
    ax.scatter([start[0]], [start[1]], color="green", s=70, marker="o", label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], color="purple", s=90, marker="*", label="goal", zorder=5)
    _draw_obstacles(ax, config)
    ax.set_title("Closed-loop oracle execution: Nominal vs Learned-FDM MPPI")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    _set_axis_limits(ax, nominal, learned, goal, start)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_obstacles(ax, config: dict) -> None:
    import matplotlib.patches as patches

    robot = config.get("robot", {})
    robot_radius = float(robot.get("radius", 0.0))
    safety_dist = float(robot.get("safety_dist", 0.0))
    for obstacle in config.get("obstacles", {}).get("virtual", []):
        if len(obstacle) < 3:
            continue
        x, y, radius = float(obstacle[0]), float(obstacle[1]), float(obstacle[2])
        ax.add_patch(
            patches.Circle(
                (x, y),
                radius,
                facecolor="0.75",
                edgecolor="0.45",
                alpha=0.65,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x, y),
                radius + robot_radius + safety_dist,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
        )


def _set_axis_limits(ax, nominal: pd.DataFrame, learned: pd.DataFrame, goal: np.ndarray, start: np.ndarray) -> None:
    xs = np.concatenate([nominal["x"].to_numpy(), learned["x"].to_numpy(), [goal[0], start[0]]])
    ys = np.concatenate([nominal["y"].to_numpy(), learned["y"].to_numpy(), [goal[1], start[1]]])
    x_margin = max(0.5, 0.05 * float(np.ptp(xs) if len(xs) else 1.0))
    y_margin = max(0.5, 0.2 * float(np.ptp(ys) if len(ys) else 1.0))
    ax.set_xlim(float(np.min(xs) - x_margin), float(np.max(xs) + x_margin))
    ax.set_ylim(float(np.min(ys) - y_margin), float(np.max(ys) + y_margin))


def _trajectory_label(prefix: str, summary: dict) -> str:
    final_distance = summary.get("final_distance")
    steps = summary.get("steps")
    if isinstance(final_distance, (int, float)) and isinstance(steps, (int, float)):
        return f"{prefix}: d={float(final_distance):.3f}m, steps={int(steps)}"
    return prefix


def _write_metrics_csv(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "nominal", "learned", "learned_minus_nominal"))
        for metric, values in metrics.items():
            writer.writerow((metric, values["nominal"], values["learned"], values["delta"]))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(learned, nominal):
    if isinstance(learned, bool) or isinstance(nominal, bool):
        return None
    if not isinstance(learned, (int, float)) or not isinstance(nominal, (int, float)):
        return None
    if not math.isfinite(float(learned)) or not math.isfinite(float(nominal)):
        return None
    return float(learned) - float(nominal)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/b2_omni_oracle.yaml")
    parser.add_argument("--scenario-name", default="standard")
    parser.add_argument("--output", default="results/stage5_visual_eval/standard_seed123_cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", choices=["numpy", "cuda", "torch"], default="cuda")
    parser.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-device", default=None)
    parser.add_argument("--fdm-residual-gain", type=float, default=1.0)
    parser.add_argument("--risk-aware-2x2", action="store_true")
    parser.add_argument("--risk-weight", type=float, default=3.0)
    parser.add_argument("--risk-power", type=float, default=2.0)
    parser.add_argument("--risk-threshold", type=float, default=0.3)
    parser.add_argument("--risk-mode", choices=["none", "cumulative", "excess"], default="excess")
    parser.add_argument("--learned-goal-xy-weight", type=float, default=3.0)
    parser.add_argument("--learned-smooth-weight", type=float, default=0.75)
    args = parser.parse_args()

    if args.risk_aware_2x2:
        summary = run_risk_aware_visual_eval(
            config_path=args.config,
            scenario_name=args.scenario_name,
            output_dir=args.output,
            seed=args.seed,
            backend=args.backend,
            fdm_model_dir=args.fdm_model_dir,
            fdm_checkpoint=args.fdm_checkpoint,
            fdm_normalization=args.fdm_normalization,
            fdm_device=args.fdm_device or ("cuda" if args.backend == "cuda" else "cpu"),
            fdm_residual_gain=args.fdm_residual_gain,
            risk_weight=args.risk_weight,
            risk_power=args.risk_power,
            risk_threshold=args.risk_threshold,
            risk_mode=args.risk_mode,
            learned_goal_xy_weight=args.learned_goal_xy_weight,
            learned_smooth_weight=args.learned_smooth_weight,
            command=shell_join([sys.executable, *sys.argv]),
            argv=[sys.executable, *sys.argv],
        )
    else:
        summary = run_visual_eval(
            config_path=args.config,
            scenario_name=args.scenario_name,
            output_dir=args.output,
            seed=args.seed,
            backend=args.backend,
            fdm_model_dir=args.fdm_model_dir,
            fdm_checkpoint=args.fdm_checkpoint,
            fdm_normalization=args.fdm_normalization,
            fdm_device=args.fdm_device or ("cuda" if args.backend == "cuda" else "cpu"),
            fdm_residual_gain=args.fdm_residual_gain,
            command=shell_join([sys.executable, *sys.argv]),
            argv=[sys.executable, *sys.argv],
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
