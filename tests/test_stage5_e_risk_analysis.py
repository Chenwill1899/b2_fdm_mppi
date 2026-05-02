import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_analysis_module():
    module_path = Path("tools/analyze_stage5_e_risk_aware.py")
    spec = importlib.util.spec_from_file_location("stage5_e_risk_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metric_delta_stats_include_bootstrap_ci_and_wilcoxon():
    module = load_analysis_module()

    stats = module.metric_delta_stats(
        [-1.0, -2.0, -3.0, -4.0],
        metric="final_distance",
        bootstrap_samples=200,
        random_seed=7,
    )

    assert stats["count"] == 4
    assert stats["mean_delta"] == pytest.approx(-2.5)
    assert stats["median_delta"] == pytest.approx(-2.5)
    assert stats["pct_improved"] == pytest.approx(1.0)
    assert stats["bootstrap_ci95"][0] < stats["mean_delta"] < stats["bootstrap_ci95"][1]
    assert stats["wilcoxon_p"] == pytest.approx(0.125)
    assert stats["wilcoxon_method"] in {"scipy", "normal_approx"}


def test_analyze_sweep_writes_stats_tables_and_figures(tmp_path):
    module = load_analysis_module()
    case_dir = tmp_path / "cases" / "safe_corridor_risk_w_1"
    runs_dir = case_dir / "runs"
    nominal_dir = runs_dir / "safe_corridor_episode_0000_nominal"
    learned_dir = runs_dir / "safe_corridor_episode_0000_learned"
    nominal_dir.mkdir(parents=True)
    learned_dir.mkdir(parents=True)
    _write_terrain_csv(nominal_dir / "terrain.csv", [0.4, 0.5, 0.6])
    _write_terrain_csv(learned_dir / "terrain.csv", [0.3, 0.35, 0.4])
    _write_trajectory_csv(nominal_dir / "trajectory.csv", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    _write_trajectory_csv(learned_dir / "trajectory.csv", [(0.0, 0.0), (1.0, 0.2), (2.0, 0.3)])
    benchmark = {
        "runs": [
            {
                "scenario": "safe_corridor",
                "controller": "nominal",
                "episode_id": 0,
                "seed": 123,
                "results_path": str(nominal_dir),
                "final_distance": 1.0,
                "cumulative_terrain_risk": 1.5,
                "terrain_risk_excess": 0.5,
                "control_jerk": 0.2,
                "mean_mppi_time_ms": 2.0,
            },
            {
                "scenario": "safe_corridor",
                "controller": "learned",
                "episode_id": 0,
                "seed": 123,
                "results_path": str(learned_dir),
                "final_distance": 0.8,
                "cumulative_terrain_risk": 1.05,
                "terrain_risk_excess": 0.2,
                "control_jerk": 0.3,
                "mean_mppi_time_ms": 7.0,
            },
        ],
        "paired_deltas": {
            "pairs": [
                {
                    "scenario": "safe_corridor",
                    "episode_id": 0,
                    "seed": 123,
                    "nominal_results_path": str(nominal_dir),
                    "learned_results_path": str(learned_dir),
                    "final_distance_delta": -0.2,
                    "cumulative_terrain_risk_delta": -0.45,
                    "terrain_risk_excess_delta": -0.3,
                    "control_jerk_delta": 0.1,
                    "mean_mppi_time_ms_delta": 5.0,
                }
            ],
            "aggregate": {"count": 1},
        },
    }
    (case_dir / "stage5_benchmark_summary.json").write_text(json.dumps(benchmark), encoding="utf-8")
    sweep = {
        "metadata": {"backend": "torch"},
        "cases": [
            {
                "case_name": "safe_corridor_risk_w_1",
                "scenario_name": "safe_corridor",
                "terrain_risk_weight": 1.0,
                "output_dir": str(case_dir),
            }
        ],
    }
    sweep_path = tmp_path / "stage5_e_risk_sweep_summary.json"
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    output_dir = tmp_path / "analysis"

    report = module.analyze_sweep(
        sweep_summary_path=sweep_path,
        output_dir=output_dir,
        bootstrap_samples=50,
        random_seed=3,
    )

    assert report["cases"][0]["metrics"]["final_distance"]["mean_delta"] == pytest.approx(-0.2)
    assert (output_dir / "stage5_e_paired_stats.json").is_file()
    assert (output_dir / "stage5_e_paired_stats.csv").is_file()
    assert (output_dir / "stage5_e_paired_delta_boxplots.png").is_file()
    assert (output_dir / "stage5_e_risk_pareto.png").is_file()
    assert (output_dir / "stage5_e_mean_risk_timeseries.png").is_file()
    assert (output_dir / "stage5_e_mean_cumulative_risk.png").is_file()


def _write_terrain_csv(path: Path, risks: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "risk_cost", "risk_excess"])
        writer.writeheader()
        for step, risk in enumerate(risks):
            writer.writerow({"step": step, "risk_cost": risk, "risk_excess": max(risk - 0.3, 0.0)})


def _write_trajectory_csv(path: Path, points: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "x", "y"])
        writer.writeheader()
        for step, (x, y) in enumerate(points):
            writer.writerow({"step": step, "x": x, "y": y})
