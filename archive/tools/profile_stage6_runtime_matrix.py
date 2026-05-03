#!/usr/bin/env python3
"""Run fixed-seed Stage 6 Torch runtime profiling matrix."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller
from tools.benchmark_learned_fdm_mppi import current_git_metadata, shell_join


RUN_METRICS = (
    "success",
    "failed",
    "final_distance",
    "steps",
    "path_length",
    "min_obstacle_clearance",
    "mean_terrain_risk",
    "max_terrain_risk",
    "cumulative_terrain_risk",
    "terrain_risk_excess",
    "terrain_risk_excess_integral",
    "terrain_risk_exposure_ratio",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
    "max_mppi_time_ms",
)

DELTA_METRICS = (
    "final_distance",
    "steps",
    "cumulative_terrain_risk",
    "terrain_risk_excess",
    "terrain_risk_exposure_ratio",
    "mean_mppi_time_ms",
    "max_mppi_time_ms",
    "profile_mean_sample_candidates_ms",
    "profile_mean_update_distribution_ms",
    "profile_mean_rollout_total_ms",
    "profile_mean_terrain_features_ms",
    "profile_mean_fdm_inference_ms",
    "profile_mean_state_integrate_ms",
    "profile_mean_obstacle_cost_ms",
    "profile_mean_terrain_risk_cost_ms",
)

CASE_PAIRS = {
    "nominal_risk_on_vs_nominal_risk_off": ("nominal_risk_off", "nominal_risk_on"),
    "learned_risk_on_vs_learned_risk_off": ("learned_risk_off", "learned_risk_on"),
    "learned_risk_off_vs_nominal_risk_off": ("nominal_risk_off", "learned_risk_off"),
    "learned_risk_on_vs_nominal_risk_on": ("nominal_risk_on", "learned_risk_on"),
}


def run_runtime_matrix(
    *,
    config_path: str | Path,
    scenario_name: str,
    output_dir: str | Path,
    episodes: int,
    steps: int,
    base_seed: int,
    backend: str = "torch",
    device: str = "auto",
    risk_weight: float = 3.0,
    risk_power: float = 2.0,
    risk_threshold: float = 0.3,
    risk_mode: str = "excess",
    fdm_model_dir: str | Path = "results/fdm_baselines/stage4_mlp_seed123_hardened",
    fdm_checkpoint: str | Path = "best_model.pt",
    fdm_normalization: str | Path = "normalization.npz",
    fdm_residual_gain: float = 0.5,
    learned_goal_xy_weight: float = 3.0,
    learned_smooth_weight: float = 0.75,
    force_steps: bool = True,
    command: str | None = None,
    argv: Sequence[str] | None = None,
    runner_cls=OmniMppiSimulationRunner,
) -> dict:
    backend = str(backend).lower()
    if backend != "torch":
        raise ValueError("Stage 6 runtime matrix currently supports only backend='torch'")
    device = _resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    base_config = load_config(config_path)
    seeds = [int(base_seed) + episode_id for episode_id in range(int(episodes))]

    runs = []
    for episode_id, seed in enumerate(seeds):
        for case in _case_specs(risk_weight):
            run_config = prepare_runtime_run_config(
                base_config,
                output_dir=output_dir,
                scenario_name=scenario_name,
                case=case,
                episode_id=episode_id,
                seed=seed,
                steps=steps,
                backend=backend,
                device=device,
                risk_power=risk_power,
                risk_threshold=risk_threshold,
                risk_mode=risk_mode,
                fdm_model_dir=fdm_model_dir,
                fdm_checkpoint=fdm_checkpoint,
                fdm_normalization=fdm_normalization,
                fdm_residual_gain=fdm_residual_gain,
                learned_goal_xy_weight=learned_goal_xy_weight,
                learned_smooth_weight=learned_smooth_weight,
                force_steps=force_steps,
            )
            runner = runner_cls(
                run_config,
                controller_factory=lambda *, config, runner, seed=seed: create_omni_controller(
                    config,
                    seed=seed,
                ),
            )
            simulation_summary = runner.run()
            summary_path = Path(simulation_summary.results_path) / "summary.json"
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runs.append(
                _run_record(
                    summary=run_summary,
                    profile=_controller_profile(runner.controller),
                    scenario=scenario_name,
                    case=case,
                    episode_id=episode_id,
                    seed=seed,
                    results_path=Path(simulation_summary.results_path),
                    summary_path=summary_path,
                )
            )

    summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config": str(config_path),
            "scenario_name": str(scenario_name),
            "output_dir": str(output_dir),
            "backend": backend,
            "device": device,
            "episodes": int(episodes),
            "steps": int(steps),
            "steps_semantics": "forced_compute_control_calls" if force_steps else "maximum_simulation_steps",
            "force_steps": bool(force_steps),
            "base_seed": int(base_seed),
            "seeds": seeds,
            "risk_weight": float(risk_weight),
            "risk_power": float(risk_power),
            "risk_threshold": float(risk_threshold),
            "risk_mode": str(risk_mode),
            "fdm_model_dir": str(fdm_model_dir),
            "fdm_checkpoint": str(fdm_checkpoint),
            "fdm_normalization": str(fdm_normalization),
            "fdm_residual_gain": float(fdm_residual_gain),
            "learned_goal_xy_weight": float(learned_goal_xy_weight),
            "learned_smooth_weight": float(learned_smooth_weight),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **current_git_metadata(),
        },
        "runs": runs,
        "profile_call_consistency": profile_call_consistency(runs, expected_calls=int(steps) if force_steps else None),
        "aggregates": aggregate_runs_by_case(runs),
        "paired_deltas": compute_case_pair_deltas(runs),
    }
    (output_dir / "stage6_runtime_matrix_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def prepare_runtime_run_config(
    base_config: dict,
    *,
    output_dir: Path,
    scenario_name: str,
    case: dict,
    episode_id: int,
    seed: int,
    steps: int,
    backend: str,
    device: str,
    risk_power: float,
    risk_threshold: float,
    risk_mode: str,
    fdm_model_dir: str | Path,
    fdm_checkpoint: str | Path,
    fdm_normalization: str | Path,
    fdm_residual_gain: float,
    learned_goal_xy_weight: float,
    learned_smooth_weight: float,
    force_steps: bool = True,
) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault("simulation", {})["world_mode"] = "oracle"
    config["simulation"]["max_steps"] = int(steps)
    config["simulation"]["disable_goal_termination"] = bool(force_steps)
    _apply_episode_seed(config, seed)
    mppi = config.setdefault("mppi", {})
    mppi["backend"] = str(backend).lower()
    mppi["device"] = str(device)
    mppi["profile_enabled"] = True
    mppi["terrain_risk_weight"] = float(case["risk_weight"])
    mppi["terrain_risk_power"] = float(risk_power)
    mppi["terrain_risk_threshold"] = float(risk_threshold)
    mppi["terrain_risk_mode"] = str(risk_mode)
    safe_scenario = _safe_name(scenario_name)
    config["results"] = {
        **config.get("results", {}),
        "root": str(output_dir / "runs"),
        "run_name": f"{safe_scenario}_episode_{int(episode_id):04d}_{case['name']}",
        "timestamp_suffix": False,
        "overwrite": True,
        "enable_plots": False,
        "enable_animation": False,
    }
    if case["controller"] == "learned":
        weights = list(mppi.get("weights", [1.0, 0.0, 0.0]))
        if weights:
            weights[0] = float(learned_goal_xy_weight)
        mppi["weights"] = weights
        mppi["smooth_weight"] = float(learned_smooth_weight)
        config["fdm"] = {
            **config.get("fdm", {}),
            "enabled": True,
            "model_dir": str(fdm_model_dir),
            "checkpoint": str(fdm_checkpoint),
            "normalization": str(fdm_normalization),
            "device": str(device),
            "residual_gain": float(fdm_residual_gain),
            "profile_enabled": True,
        }
    else:
        config.pop("fdm", None)
    return config


def aggregate_runs_by_case(runs: Sequence[dict]) -> dict:
    aggregates = {}
    for case_name in sorted({run["case"] for run in runs}):
        subset = [run for run in runs if run["case"] == case_name]
        aggregate = {
            "count": len(subset),
            "success_rate": float(sum(1 for run in subset if bool(run.get("success", False))) / len(subset))
            if subset
            else 0.0,
        }
        numeric_keys = sorted({key for run in subset for key in run if _finite_float(run.get(key)) is not None})
        for key in numeric_keys:
            if key in {"episode_id", "seed", "risk_weight"}:
                continue
            values = [_finite_float(run.get(key)) for run in subset]
            values = [value for value in values if value is not None]
            if values:
                aggregate[f"{key}_mean"] = float(sum(values) / len(values))
        aggregates[case_name] = aggregate
    return aggregates


def profile_call_consistency(runs: Sequence[dict], expected_calls: int | None = None) -> dict:
    total_calls_by_case = {}
    mismatches = []
    for run in runs:
        case = str(run["case"])
        episode_id = int(run["episode_id"])
        total_calls = int(run.get("profile_total_calls", 0))
        total_calls_by_case.setdefault(case, []).append(total_calls)
        if expected_calls is not None and total_calls != int(expected_calls):
            mismatches.append(
                {
                    "case": case,
                    "episode_id": episode_id,
                    "seed": int(run["seed"]),
                    "total_calls": total_calls,
                    "expected_calls": int(expected_calls),
                    "failed": bool(run.get("failed", False)),
                }
            )
    unique_counts = sorted({count for counts in total_calls_by_case.values() for count in counts})
    return {
        "consistent": len(unique_counts) <= 1 and not mismatches,
        "expected_calls_per_run": int(expected_calls) if expected_calls is not None else None,
        "unique_total_calls": unique_counts,
        "total_calls_by_case": total_calls_by_case,
        "mismatches": mismatches,
    }


def compute_case_pair_deltas(runs: Sequence[dict]) -> dict:
    indexed = {(run["case"], run["scenario"], int(run["episode_id"]), int(run["seed"])): run for run in runs}
    paired = {}
    for pair_name, (base_case, compare_case) in CASE_PAIRS.items():
        pairs = []
        base_keys = {key[1:] for key in indexed if key[0] == base_case}
        compare_keys = {key[1:] for key in indexed if key[0] == compare_case}
        for scenario, episode_id, seed in sorted(base_keys & compare_keys):
            base_run = indexed[(base_case, scenario, episode_id, seed)]
            compare_run = indexed[(compare_case, scenario, episode_id, seed)]
            pair = {
                "scenario": scenario,
                "episode_id": int(episode_id),
                "seed": int(seed),
                "base_case": base_case,
                "compare_case": compare_case,
                "success_delta": float(bool(compare_run.get("success", False)))
                - float(bool(base_run.get("success", False))),
            }
            for metric in DELTA_METRICS:
                base_value = _finite_float(base_run.get(metric))
                compare_value = _finite_float(compare_run.get(metric))
                pair[f"{metric}_delta"] = (
                    float(compare_value - base_value)
                    if base_value is not None and compare_value is not None
                    else None
                )
            pairs.append(pair)
        paired[pair_name] = {
            "pairs": pairs,
            "aggregate": _aggregate_deltas(pairs),
        }
    return paired


def _aggregate_deltas(pairs: Sequence[dict]) -> dict:
    aggregate = {"count": len(pairs)}
    for metric in ("success", *DELTA_METRICS):
        key = f"{metric}_delta"
        values = [_finite_float(pair.get(key)) for pair in pairs]
        values = [value for value in values if value is not None]
        aggregate[f"{key}_mean"] = float(sum(values) / len(values)) if values else None
    return aggregate


def _run_record(
    *,
    summary: dict,
    profile: dict,
    scenario: str,
    case: dict,
    episode_id: int,
    seed: int,
    results_path: Path,
    summary_path: Path,
) -> dict:
    record = {
        "scenario": str(scenario),
        "case": str(case["name"]),
        "controller": str(case["controller"]),
        "risk_label": str(case["risk_label"]),
        "risk_weight": float(case["risk_weight"]),
        "episode_id": int(episode_id),
        "seed": int(seed),
        "results_path": str(results_path),
        "summary_json": str(summary_path),
        "profile": profile,
        "profile_total_calls": int(profile.get("total_calls", 0)),
    }
    for metric in RUN_METRICS:
        record[metric] = _json_scalar(summary.get(metric))
    for bucket, value in profile.get("means_ms", {}).items():
        record[f"profile_mean_{bucket}"] = float(value)
    for bucket, value in profile.get("totals_ms", {}).items():
        record[f"profile_total_{bucket}"] = float(value)
    return record


def _case_specs(risk_weight: float) -> list[dict]:
    return [
        {
            "name": "nominal_risk_off",
            "controller": "nominal",
            "risk_label": "risk_off",
            "risk_weight": 0.0,
        },
        {
            "name": "nominal_risk_on",
            "controller": "nominal",
            "risk_label": "risk_on",
            "risk_weight": float(risk_weight),
        },
        {
            "name": "learned_risk_off",
            "controller": "learned",
            "risk_label": "risk_off",
            "risk_weight": 0.0,
        },
        {
            "name": "learned_risk_on",
            "controller": "learned",
            "risk_label": "risk_on",
            "risk_weight": float(risk_weight),
        },
    ]


def _apply_episode_seed(config: dict, seed: int) -> None:
    if bool(config.get("scenario", {}).get("random_start_goal_enabled", False)):
        config.setdefault("scenario", {})["random_seed"] = int(seed)
    if "oracle_residual" in config:
        config.setdefault("oracle_residual", {})["seed"] = int(seed)


def _controller_profile(controller) -> dict:
    profile_summary = getattr(controller, "profile_summary", None)
    if callable(profile_summary):
        return profile_summary()
    return {
        "enabled": False,
        "total_calls": 0,
        "totals_ms": {},
        "means_ms": {},
        "counts": {},
    }


def _resolve_device(device: str) -> str:
    requested = str(device).lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_name(name: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(name)]
    safe = "".join(chars).strip("_")
    return safe or "scenario"


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


def _json_scalar(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/b2_omni_oracle_random100_dataset.yaml")
    parser.add_argument("--scenario-name", default="id_random")
    parser.add_argument("--output", default="results/stage6_runtime_profile/paired_runtime_matrix")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--backend", choices=["torch"], default="torch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--risk-weight", type=float, default=3.0)
    parser.add_argument("--risk-power", type=float, default=2.0)
    parser.add_argument("--risk-threshold", type=float, default=0.3)
    parser.add_argument("--risk-mode", default="excess")
    parser.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-residual-gain", type=float, default=0.5)
    parser.add_argument("--learned-goal-xy-weight", type=float, default=3.0)
    parser.add_argument("--learned-smooth-weight", type=float, default=0.75)
    parser.add_argument(
        "--allow-goal-termination",
        action="store_true",
        help="Keep normal runner semantics where --steps is only max_steps. By default, Stage 6 profiling forces exactly --steps control calls unless the run fails.",
    )
    args = parser.parse_args()

    summary = run_runtime_matrix(
        config_path=args.config,
        scenario_name=args.scenario_name,
        output_dir=args.output,
        episodes=args.episodes,
        steps=args.steps,
        base_seed=args.base_seed,
        backend=args.backend,
        device=args.device,
        risk_weight=args.risk_weight,
        risk_power=args.risk_power,
        risk_threshold=args.risk_threshold,
        risk_mode=args.risk_mode,
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_residual_gain=args.fdm_residual_gain,
        learned_goal_xy_weight=args.learned_goal_xy_weight,
        learned_smooth_weight=args.learned_smooth_weight,
        force_steps=not args.allow_goal_termination,
        command=shell_join([sys.executable, *sys.argv]),
        argv=[sys.executable, *sys.argv],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
