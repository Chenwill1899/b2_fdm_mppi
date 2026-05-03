import csv
import importlib.util
import json
import sys
from pathlib import Path


def load_plot_module():
    module_path = Path("tools/plot_stage6_runtime_results.py")
    spec = importlib.util.spec_from_file_location("stage6_runtime_plot", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_stage6_runtime_results_writes_figures_and_tables(tmp_path):
    module = load_plot_module()
    summary_path = tmp_path / "stage6_runtime_matrix_summary.json"
    summary_path.write_text(json.dumps(_runtime_summary()), encoding="utf-8")
    figure_dir = tmp_path / "figures"
    table_dir = tmp_path / "tables"

    report = module.plot_stage6_runtime_results(
        summary_paths=[summary_path],
        output_dir=figure_dir,
        tables_dir=table_dir,
    )

    assert report["summary_count"] == 1
    for name in (
        "fig_stage6_runtime_summary",
        "fig_stage6_runtime_breakdown",
        "fig_stage6_runtime_delta_buckets",
    ):
        assert (figure_dir / f"{name}.png").is_file()
        assert (figure_dir / f"{name}.pdf").is_file()
    assert (figure_dir / "stage6_runtime_figure_manifest.json").is_file()
    assert (table_dir / "table_stage6_runtime_summary.csv").is_file()
    assert (table_dir / "table_stage6_runtime_paired_deltas.csv").is_file()

    with (table_dir / "table_stage6_runtime_summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["case"] for row in rows} == {
        "nominal_risk_off",
        "nominal_risk_on",
        "learned_risk_off",
        "learned_risk_on",
    }


def _runtime_summary() -> dict:
    aggregates = {}
    for case, mean_mppi, rollout, terrain, fdm, risk in (
        ("nominal_risk_off", 20.0, 9.0, None, None, 0.05),
        ("nominal_risk_on", 22.0, 9.5, None, None, 1.0),
        ("learned_risk_off", 42.0, 36.0, 0.8, 0.2, 0.05),
        ("learned_risk_on", 44.0, 37.0, 0.9, 0.3, 1.2),
    ):
        aggregates[case] = {
            "count": 2,
            "success_rate": 0.0,
            "mean_mppi_time_ms_mean": mean_mppi,
            "profile_mean_rollout_total_ms_mean": rollout,
            "profile_mean_sample_candidates_ms_mean": 0.2,
            "profile_mean_update_distribution_ms_mean": 0.3,
            "profile_mean_terrain_risk_cost_ms_mean": risk,
            "profile_mean_obstacle_cost_ms_mean": 2.0,
            "profile_mean_state_integrate_ms_mean": 0.2,
        }
        if terrain is not None:
            aggregates[case]["profile_mean_terrain_features_ms_mean"] = terrain
            aggregates[case]["profile_mean_fdm_inference_ms_mean"] = fdm
    return {
        "metadata": {
            "scenario_name": "unit_runtime",
            "output_dir": "results/stage6_runtime_profile/unit_runtime",
            "backend": "torch",
            "device": "cuda",
            "episodes": 2,
            "steps": 3,
            "base_seed": 123,
            "git_sha": "abc123",
        },
        "runs": [],
        "aggregates": aggregates,
        "paired_deltas": {
            "learned_risk_on_vs_nominal_risk_on": {
                "aggregate": {
                    "count": 2,
                    "mean_mppi_time_ms_delta_mean": 22.0,
                    "profile_mean_rollout_total_ms_delta_mean": 27.5,
                    "profile_mean_sample_candidates_ms_delta_mean": 0.0,
                    "profile_mean_update_distribution_ms_delta_mean": 0.0,
                    "profile_mean_terrain_risk_cost_ms_delta_mean": 0.2,
                }
            }
        },
    }
