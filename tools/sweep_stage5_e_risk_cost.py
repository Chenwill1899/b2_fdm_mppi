#!/usr/bin/env python3
"""Run Stage 5-E terrain-risk cost sanity sweeps on complex terrain maps."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.config import load_config
from tools.benchmark_learned_fdm_mppi import (
    current_git_metadata,
    parse_controllers,
    run_benchmark,
    shell_join,
)


DEFAULT_CONFIGS = (
    "config/b2_omni_oracle_risk_band.yaml",
    "config/b2_omni_oracle_risk_island.yaml",
    "config/b2_omni_oracle_low_friction_patch.yaml",
    "config/b2_omni_oracle_safe_corridor.yaml",
)
BenchmarkFn = Callable[..., dict]


def run_risk_sweep(
    *,
    config_paths: Sequence[str | Path],
    output_dir: str | Path,
    episodes: int,
    base_seed: int,
    backend: str,
    controllers: Sequence[str],
    risk_weights: Sequence[float],
    risk_power: float,
    risk_threshold: float,
    risk_mode: str,
    fdm_model_dir: str | Path,
    fdm_checkpoint: str | Path,
    fdm_normalization: str | Path,
    fdm_device: str | None,
    fdm_residual_gain: float = 0.5,
    learned_goal_xy_weight: float | None = None,
    learned_smooth_weight: float | None = None,
    num_trajectories: int | None = None,
    time_horizon: float | None = None,
    max_steps: int | None = None,
    draw_num_traj: int | None = None,
    benchmark_fn: BenchmarkFn = run_benchmark,
    command: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = expand_risk_sweep_cases(config_paths=config_paths, risk_weights=risk_weights)
    case_records = []
    prepared_configs: dict[str, str] = {}
    learned_overrides = _learned_overrides(learned_goal_xy_weight, learned_smooth_weight)

    for case in cases:
        source_config = str(case["config_path"])
        if _needs_reduced_config(num_trajectories, time_horizon, max_steps, draw_num_traj):
            prepared_configs[source_config] = prepared_configs.get(source_config) or prepare_reduced_config(
                config_path=source_config,
                output_dir=output_dir / "reduced_configs",
                num_trajectories=num_trajectories,
                time_horizon=time_horizon,
                max_steps=max_steps,
                draw_num_traj=draw_num_traj,
            )
            config_path = prepared_configs[source_config]
        else:
            config_path = source_config
        mppi_overrides = {
            "terrain_risk_weight": float(case["terrain_risk_weight"]),
            "terrain_risk_power": float(risk_power),
            "terrain_risk_threshold": float(risk_threshold),
            "terrain_risk_mode": str(risk_mode).lower(),
        }
        case_output = output_dir / "cases" / case["case_name"]
        benchmark_summary = benchmark_fn(
            config_path=config_path,
            scenario_name=case["scenario_name"],
            output_dir=case_output,
            episodes=episodes,
            base_seed=base_seed,
            backend=backend,
            controllers=controllers,
            fdm_model_dir=fdm_model_dir,
            fdm_checkpoint=fdm_checkpoint,
            fdm_normalization=fdm_normalization,
            fdm_device=fdm_device,
            fdm_residual_gain=fdm_residual_gain,
            mppi_overrides=mppi_overrides,
            learned_mppi_overrides=learned_overrides,
            command=command,
            argv=argv,
        )
        case_records.append(_case_record(case, case_output, benchmark_summary))

    summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config_paths": [str(path) for path in config_paths],
            "output_dir": str(output_dir),
            "episodes": int(episodes),
            "base_seed": int(base_seed),
            "backend": str(backend),
            "controllers": [str(controller) for controller in controllers],
            "risk_weights": [float(value) for value in risk_weights],
            "risk_power": float(risk_power),
            "risk_threshold": float(risk_threshold),
            "risk_mode": str(risk_mode).lower(),
            "fdm_model_dir": str(fdm_model_dir),
            "fdm_checkpoint": str(fdm_checkpoint),
            "fdm_normalization": str(fdm_normalization),
            "fdm_device": str(fdm_device),
            "fdm_residual_gain": float(fdm_residual_gain),
            "learned_mppi_overrides": learned_overrides,
            "reduced_config_overrides": {
                "num_trajectories": num_trajectories,
                "time_horizon": time_horizon,
                "max_steps": max_steps,
                "draw_num_traj": draw_num_traj,
            },
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **current_git_metadata(),
        },
        "cases": case_records,
    }
    (output_dir / "stage5_e_risk_sweep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def expand_risk_sweep_cases(
    *,
    config_paths: Sequence[str | Path],
    risk_weights: Sequence[float],
) -> list[dict]:
    if not config_paths:
        raise ValueError("At least one config path is required")
    if not risk_weights:
        raise ValueError("At least one terrain risk weight is required")
    cases = []
    for config_path in config_paths:
        scenario_name = _scenario_name(config_path)
        for weight in risk_weights:
            cases.append(
                {
                    "case_name": f"{scenario_name}_risk_w_{_float_token(float(weight))}",
                    "scenario_name": scenario_name,
                    "config_path": str(config_path),
                    "terrain_risk_weight": float(weight),
                }
            )
    return cases


def prepare_reduced_config(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    num_trajectories: int | None = None,
    time_horizon: float | None = None,
    max_steps: int | None = None,
    draw_num_traj: int | None = None,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(load_config(config_path))
    if num_trajectories is not None:
        config.setdefault("mppi", {})["num_trajectories"] = int(num_trajectories)
    if draw_num_traj is not None:
        config.setdefault("mppi", {})["draw_num_traj"] = int(draw_num_traj)
    if time_horizon is not None:
        config.setdefault("simulation", {})["time_horizon"] = float(time_horizon)
    if max_steps is not None:
        config.setdefault("simulation", {})["max_steps"] = int(max_steps)
    config.setdefault("results", {})["enable_plots"] = False
    config.setdefault("results", {})["enable_animation"] = False
    path = output_dir / f"{Path(config_path).stem}_reduced.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def parse_config_paths(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_CONFIGS)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_risk_weights(value: str) -> list[float]:
    weights = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not weights:
        raise ValueError("At least one terrain risk weight is required")
    return weights


def _case_record(case: dict, output_dir: Path, benchmark_summary: dict) -> dict:
    aggregates = benchmark_summary.get("aggregates", {})
    paired = benchmark_summary.get("paired_deltas", {}).get("aggregate", {})
    record = {
        **case,
        "output_dir": str(output_dir),
        "paired_delta_aggregate": paired,
    }
    for controller in ("nominal", "learned"):
        stats = aggregates.get(controller, {})
        prefix = f"{controller}_"
        for metric in (
            "success_rate",
            "final_distance_mean",
            "steps_mean",
            "mean_terrain_risk_mean",
            "max_terrain_risk_mean",
            "cumulative_terrain_risk_mean",
            "terrain_risk_excess_mean",
            "terrain_risk_excess_integral_mean",
            "terrain_risk_exposure_ratio_mean",
            "control_smoothness_mean",
            "control_jerk_mean",
            "mean_mppi_time_ms_mean",
        ):
            record[f"{prefix}{metric}"] = stats.get(metric)
    return record


def _learned_overrides(goal_xy_weight: float | None, smooth_weight: float | None) -> dict[str, float]:
    overrides = {}
    if goal_xy_weight is not None:
        overrides["goal_xy_weight"] = float(goal_xy_weight)
    if smooth_weight is not None:
        overrides["smooth_weight"] = float(smooth_weight)
    return overrides


def _needs_reduced_config(*values) -> bool:
    return any(value is not None for value in values)


def _scenario_name(config_path: str | Path) -> str:
    stem = Path(config_path).stem
    prefix = "b2_omni_oracle_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _float_token(value: float) -> str:
    token = f"{value:g}".replace("-", "neg_").replace(".", "_")
    return token or "0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--output", default="results/stage5_e_risk_aware/s5_e1_sanity_sweep")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--backend", choices=["numpy", "cuda"], default="numpy")
    parser.add_argument("--controllers", default="nominal")
    parser.add_argument("--risk-weights", default="0,1,3,5,10")
    parser.add_argument("--risk-power", type=float, default=2.0)
    parser.add_argument("--risk-threshold", type=float, default=0.3)
    parser.add_argument("--risk-mode", choices=["none", "cumulative", "excess"], default="excess")
    parser.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-device", default=None)
    parser.add_argument("--fdm-residual-gain", type=float, default=0.5)
    parser.add_argument("--learned-goal-xy-weight", type=float, default=None)
    parser.add_argument("--learned-smooth-weight", type=float, default=None)
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--draw-num-traj", type=int, default=None)
    args = parser.parse_args()

    summary = run_risk_sweep(
        config_paths=parse_config_paths(args.configs),
        output_dir=args.output,
        episodes=args.episodes,
        base_seed=args.base_seed,
        backend=args.backend,
        controllers=parse_controllers(args.controllers),
        risk_weights=parse_risk_weights(args.risk_weights),
        risk_power=args.risk_power,
        risk_threshold=args.risk_threshold,
        risk_mode=args.risk_mode,
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_device=args.fdm_device or ("cuda" if args.backend == "cuda" else "cpu"),
        fdm_residual_gain=args.fdm_residual_gain,
        learned_goal_xy_weight=args.learned_goal_xy_weight,
        learned_smooth_weight=args.learned_smooth_weight,
        num_trajectories=args.num_trajectories,
        time_horizon=args.time_horizon,
        max_steps=args.max_steps,
        draw_num_traj=args.draw_num_traj,
        command=shell_join([sys.executable, *sys.argv]),
        argv=[sys.executable, *sys.argv],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
