#!/usr/bin/env python3
"""Run Stage 5-C residual-gain and learned-cost calibration sweeps."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.benchmark_learned_fdm_mppi import (
    LEARNED_MPPI_OVERRIDE_KEYS,
    current_git_metadata,
    parse_controllers,
    run_benchmark,
    shell_join,
)


BenchmarkFn = Callable[..., dict]


def run_calibration_sweep(
    *,
    config_path: str | Path,
    scenario_name: str,
    output_dir: str | Path,
    episodes: int,
    base_seed: int,
    backend: str,
    controllers: Sequence[str],
    fdm_model_dir: str | Path,
    fdm_checkpoint: str | Path,
    fdm_normalization: str | Path,
    fdm_device: str | None,
    residual_gains: Sequence[float],
    cost_grid: dict[str, Sequence[float]] | None = None,
    benchmark_fn: BenchmarkFn = run_benchmark,
    command: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = expand_sweep_cases(residual_gains=residual_gains, cost_grid=cost_grid or {})
    case_records = []

    for case in cases:
        case_output = output_dir / "cases" / case["case_name"]
        benchmark_summary = benchmark_fn(
            config_path=config_path,
            scenario_name=scenario_name,
            output_dir=case_output,
            episodes=episodes,
            base_seed=base_seed,
            backend=backend,
            controllers=controllers,
            fdm_model_dir=fdm_model_dir,
            fdm_checkpoint=fdm_checkpoint,
            fdm_normalization=fdm_normalization,
            fdm_device=fdm_device,
            fdm_residual_gain=case["fdm_residual_gain"],
            learned_mppi_overrides=case["learned_mppi_overrides"],
            command=command,
            argv=argv,
        )
        case_records.append(_case_record(case, case_output, benchmark_summary))

    summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config": str(config_path),
            "scenario_name": str(scenario_name),
            "output_dir": str(output_dir),
            "episodes": int(episodes),
            "base_seed": int(base_seed),
            "backend": str(backend),
            "controllers": [str(controller) for controller in controllers],
            "residual_gains": [float(value) for value in residual_gains],
            "cost_grid": {key: [float(value) for value in values] for key, values in (cost_grid or {}).items()},
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **current_git_metadata(),
        },
        "cases": case_records,
    }
    (output_dir / "stage5_calibration_sweep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def expand_sweep_cases(
    *,
    residual_gains: Sequence[float],
    cost_grid: dict[str, Sequence[float]],
) -> list[dict]:
    if not residual_gains:
        raise ValueError("At least one residual gain is required")
    _validate_cost_grid(cost_grid)
    keys = list(cost_grid)
    value_product = itertools.product(*(cost_grid[key] for key in keys)) if keys else [()]
    cost_cases = [dict(zip(keys, values)) for values in value_product]
    cases = []
    for gain in residual_gains:
        for overrides in cost_cases:
            case_name = _case_name(float(gain), overrides)
            cases.append(
                {
                    "case_name": case_name,
                    "fdm_residual_gain": float(gain),
                    "learned_mppi_overrides": {key: float(value) for key, value in overrides.items()},
                }
            )
    return cases


def parse_residual_gains(value: str) -> list[float]:
    gains = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not gains:
        raise ValueError("At least one residual gain is required")
    return gains


def parse_cost_grid_entries(entries: Sequence[str] | None) -> dict[str, list[float]]:
    grid: dict[str, list[float]] = {}
    for entry in entries or []:
        if "=" not in str(entry):
            raise ValueError(f"Expected cost grid entry in KEY=V1,V2 form, got: {entry}")
        key, values = str(entry).split("=", 1)
        key = key.strip()
        if key not in LEARNED_MPPI_OVERRIDE_KEYS:
            raise ValueError(f"Unsupported learned MPPI override '{key}'. Expected one of {LEARNED_MPPI_OVERRIDE_KEYS}")
        parsed_values = [float(value.strip()) for value in values.split(",") if value.strip()]
        if not parsed_values:
            raise ValueError(f"Cost grid entry '{entry}' must include at least one numeric value")
        grid[key] = parsed_values
    return grid


def _case_record(case: dict, output_dir: Path, benchmark_summary: dict) -> dict:
    aggregates = benchmark_summary.get("aggregates", {})
    paired = benchmark_summary.get("paired_deltas", {}).get("aggregate", {})
    learned = aggregates.get("learned", {})
    nominal = aggregates.get("nominal", {})
    return {
        **case,
        "output_dir": str(output_dir),
        "learned_success_rate": learned.get("success_rate"),
        "nominal_success_rate": nominal.get("success_rate"),
        "learned_final_distance_mean": learned.get("final_distance_mean"),
        "nominal_final_distance_mean": nominal.get("final_distance_mean"),
        "learned_steps_mean": learned.get("steps_mean"),
        "nominal_steps_mean": nominal.get("steps_mean"),
        "learned_mean_mppi_time_ms": learned.get("mean_mppi_time_ms_mean"),
        "nominal_mean_mppi_time_ms": nominal.get("mean_mppi_time_ms_mean"),
        "paired_delta_aggregate": paired,
    }


def _validate_cost_grid(cost_grid: dict[str, Sequence[float]]) -> None:
    for key, values in cost_grid.items():
        if key not in LEARNED_MPPI_OVERRIDE_KEYS:
            raise ValueError(f"Unsupported learned MPPI override '{key}'. Expected one of {LEARNED_MPPI_OVERRIDE_KEYS}")
        if not values:
            raise ValueError(f"Cost grid key '{key}' must include at least one value")


def _case_name(residual_gain: float, overrides: dict[str, float]) -> str:
    parts = ["gain", _float_token(residual_gain)]
    for key, value in overrides.items():
        parts.extend([key, _float_token(float(value))])
    return "_".join(parts)


def _float_token(value: float) -> str:
    token = f"{value:g}".replace("-", "neg_").replace(".", "_")
    return token or "0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/b2_omni_oracle_random100_dataset.yaml")
    parser.add_argument("--scenario-name", default="id_random_tasks_quick")
    parser.add_argument("--output", default="results/stage5_calibration/id_random_tasks_quick_seed123_cuda")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--backend", choices=["numpy", "cuda"], default="cuda")
    parser.add_argument("--controllers", default="nominal,learned")
    parser.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-device", default=None)
    parser.add_argument("--residual-gains", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--cost-grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Learned-controller-only cost grid entry; repeat for multiple keys.",
    )
    args = parser.parse_args()

    summary = run_calibration_sweep(
        config_path=args.config,
        scenario_name=args.scenario_name,
        output_dir=args.output,
        episodes=args.episodes,
        base_seed=args.base_seed,
        backend=args.backend,
        controllers=parse_controllers(args.controllers),
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_device=args.fdm_device or ("cuda" if args.backend == "cuda" else "cpu"),
        residual_gains=parse_residual_gains(args.residual_gains),
        cost_grid=parse_cost_grid_entries(args.cost_grid),
        command=shell_join([sys.executable, *sys.argv]),
        argv=[sys.executable, *sys.argv],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
