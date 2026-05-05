import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_plot_module():
    module_path = Path("tools/plot_stage5_e_risk_aware_results.py")
    spec = importlib.util.spec_from_file_location("stage5_e_plot_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_stage5_e_results_writes_tables_and_nature_style_figures(tmp_path):
    module = load_plot_module()
    case_dir = tmp_path / "cases" / "two_obstacle_standard_risk_w_3"
    nominal_dir = case_dir / "runs" / "two_obstacle_standard_episode_0000_nominal"
    learned_dir = case_dir / "runs" / "two_obstacle_standard_episode_0000_learned"
    _write_run_dir(nominal_dir, risks=[0.4, 0.5, 0.45], points=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    _write_run_dir(learned_dir, risks=[0.2, 0.25, 0.2], points=[(0.0, 0.0), (1.0, 0.2), (2.0, 0.3)])
    benchmark = {
        "metadata": {
            "config": "config/b2_omni_oracle.yaml",
            "backend": "torch",
            "mppi_overrides": {"terrain_risk_weight": 3.0},
        },
        "runs": [
            _run_record("two_obstacle_standard", "nominal", nominal_dir, final_distance=0.8, risk=1.35, runtime=2.0),
            _run_record("two_obstacle_standard", "learned", learned_dir, final_distance=0.5, risk=0.65, runtime=6.0),
        ],
        "aggregates": {
            "nominal": {
                "success_rate": 1.0,
                "final_distance_mean": 0.8,
                "steps_mean": 3.0,
                "cumulative_terrain_risk_mean": 1.35,
                "terrain_risk_excess_mean": 0.45,
                "terrain_risk_exposure_ratio_mean": 1.0,
                "control_jerk_mean": 0.2,
                "mean_mppi_time_ms_mean": 2.0,
            },
            "learned": {
                "success_rate": 1.0,
                "final_distance_mean": 0.5,
                "steps_mean": 3.0,
                "cumulative_terrain_risk_mean": 0.65,
                "terrain_risk_excess_mean": 0.0,
                "terrain_risk_exposure_ratio_mean": 0.0,
                "control_jerk_mean": 0.3,
                "mean_mppi_time_ms_mean": 6.0,
            },
        },
        "paired_deltas": {
            "pairs": [
                {
                    "scenario": "two_obstacle_standard",
                    "episode_id": 0,
                    "seed": 123,
                    "nominal_results_path": str(nominal_dir),
                    "learned_results_path": str(learned_dir),
                    "final_distance_delta": -0.3,
                    "cumulative_terrain_risk_delta": -0.7,
                    "terrain_risk_excess_delta": -0.45,
                    "control_jerk_delta": 0.1,
                    "mean_mppi_time_ms_delta": 4.0,
                }
            ]
        },
    }
    (case_dir / "stage5_benchmark_summary.json").write_text(json.dumps(benchmark), encoding="utf-8")
    sweep = {
        "metadata": {"backend": "torch"},
        "cases": [
            {
                "case_name": "two_obstacle_standard_risk_w_3",
                "scenario_name": "two_obstacle_standard",
                "config_path": "config/b2_omni_oracle.yaml",
                "terrain_risk_weight": 3.0,
                "output_dir": str(case_dir),
            }
        ],
    }
    sweep_path = tmp_path / "stage5_e_risk_sweep_summary.json"
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    figure_dir = tmp_path / "figures"
    table_dir = tmp_path / "tables"

    report = module.plot_stage5_e_results(
        sweep_summary_paths=[sweep_path],
        output_dir=figure_dir,
        tables_dir=table_dir,
        bootstrap_samples=20,
        random_seed=5,
    )

    assert report["case_count"] == 1
    assert (table_dir / "table_s5e_main_results.csv").is_file()
    assert (table_dir / "table_s5e_paired_stats.csv").is_file()
    assert (figure_dir / "fig_s5e_main_risk_aware_summary.png").is_file()
    assert (figure_dir / "fig_s5e_main_risk_aware_summary.pdf").is_file()
    assert (figure_dir / "fig_s5e_risk_pareto.png").is_file()
    assert (figure_dir / "fig_s5e_trajectory_over_risk_map.png").is_file()
    assert (figure_dir / "fig_s5e_two_obstacle_trajectory_over_risk.png").is_file()
    assert (figure_dir / "fig_s5e_risk_timeseries.png").is_file()
    assert (figure_dir / "fig_s5e_runtime.png").is_file()

    rows = list(csv.DictReader((table_dir / "table_s5e_main_results.csv").open(encoding="utf-8")))
    assert rows[0]["scenario_name"] == "two_obstacle_standard"
    assert float(rows[0]["final_distance_delta"]) == pytest.approx(-0.3)


def _write_run_dir(path: Path, *, risks: list[float], points: list[tuple[float, float]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "terrain.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "risk_cost", "risk_excess"])
        writer.writeheader()
        for step, risk in enumerate(risks):
            writer.writerow({"step": step, "risk_cost": risk, "risk_excess": max(risk - 0.3, 0.0)})
    with (path / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "x", "y"])
        writer.writeheader()
        for step, (x, y) in enumerate(points):
            writer.writerow({"step": step, "x": x, "y": y})


def _run_record(scenario: str, controller: str, path: Path, *, final_distance: float, risk: float, runtime: float) -> dict:
    return {
        "scenario": scenario,
        "controller": controller,
        "episode_id": 0,
        "seed": 123,
        "results_path": str(path),
        "success": True,
        "final_distance": final_distance,
        "steps": 3,
        "cumulative_terrain_risk": risk,
        "terrain_risk_excess": max(risk - 0.9, 0.0),
        "terrain_risk_exposure_ratio": 0.5,
        "control_jerk": 0.2,
        "mean_mppi_time_ms": runtime,
    }
