import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_plot_module():
    module_path = Path("tools/plot_paper_figures.py")
    spec = importlib.util.spec_from_file_location("plot_paper_figures", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_paper_figures_writes_nature_style_package(tmp_path):
    module = load_plot_module()
    sources = _write_minimal_sources(tmp_path)
    module.STAGE4_SEED_SUMMARY = sources["stage4_seed_summary"]
    module.STAGE4_ROLLOUT = sources["stage4_rollout"]
    module.OOD_OBSTACLE = sources["ood_obstacle"]
    module.OOD_TERRAIN = sources["ood_terrain"]
    module.STAGE5_SUMMARY = sources["stage5_summary"]
    module.STAGE5_DELTAS = sources["stage5_deltas"]
    module.STAGE5E_STATS = sources["stage5e_stats"]
    module.STAGE5E_MAIN = sources["stage5e_main"]
    module.STAGE5E_VISUAL = sources["stage5e_visual"]

    report = module.plot_paper_figures(
        output_dir=tmp_path / "figures" / "paper",
        tables_dir=tmp_path / "tables" / "paper",
        package_dir=tmp_path / "package",
        package_name="paper_figures.zip",
    )

    assert (tmp_path / "figures" / "paper" / "fig1_method_and_experiment_suite.svg").is_file()
    assert (tmp_path / "figures" / "paper" / "fig2_residual_fidelity_open_loop.png").is_file()
    assert (tmp_path / "figures" / "paper" / "fig3_closed_loop_operating_modes.pdf").is_file()
    assert (tmp_path / "figures" / "paper" / "fig4_risk_aware_closed_loop.svg").is_file()
    assert (tmp_path / "tables" / "paper" / "paper_stage4_model_metrics.csv").is_file()
    assert (tmp_path / "figures" / "paper" / "paper_figure_manifest.json").is_file()
    assert (tmp_path / "package" / "paper_figures.zip").is_file()
    assert "Runtime is intentionally excluded" in report["runtime_claim"]


def _write_minimal_sources(tmp_path: Path) -> dict:
    stage4_seed = tmp_path / "stage4_seed_benchmark_summary.json"
    stage4_seed.write_text(
        json.dumps(
            {
                "mean": {
                    "test_mse": 1.0e-5,
                    "zero_residual_test_mse": 6.0e-4,
                    "overall_test_mse_reduction_pct": 98.0,
                    "overall_test_improvement_x": 60.0,
                    "test_rmse_vx": 0.003,
                    "test_rmse_vy": 0.003,
                    "test_rmse_wz": 0.003,
                },
                "std": {"test_mse": 1.0e-7},
            }
        ),
        encoding="utf-8",
    )
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "rollout_metrics.json").write_text(
        json.dumps(
            {
                "learned_ade_xy": 0.05,
                "nominal_ade_xy": 0.5,
                "learned_fde_xy": 0.08,
                "nominal_fde_xy": 0.9,
                "nominal_ade_xy_at_1s": 0.02,
                "learned_ade_xy_at_1s": 0.002,
                "nominal_ade_xy_at_2s": 0.04,
                "learned_ade_xy_at_2s": 0.004,
                "nominal_ade_xy_at_4s": 0.08,
                "learned_ade_xy_at_4s": 0.008,
            }
        ),
        encoding="utf-8",
    )
    states = np.column_stack([np.linspace(0, 2, 5), np.linspace(0, 1, 5), np.zeros((5, 4))]).astype("float32")
    np.savez(
        rollout / "rollout_replay.npz",
        oracle_states=states,
        nominal_states=states + np.array([0.0, 0.4, 0, 0, 0, 0], dtype="float32"),
        learned_states=states + np.array([0.0, 0.05, 0, 0, 0, 0], dtype="float32"),
    )
    ood_obstacle = _write_ood_dir(tmp_path / "ood_obstacle", test_mse=1.1e-5, zero=6.1e-4, ade=0.03, nominal=0.25)
    ood_terrain = _write_ood_dir(tmp_path / "ood_terrain", test_mse=1.2e-5, zero=6.2e-4, ade=0.04, nominal=0.30)

    stage5_summary = tmp_path / "stage5_summary.json"
    rows = []
    for scenario in ("id_random", "ood_obstacle", "ood_terrain"):
        for controller in module_controllers():
            delta = 0.0 if controller == "nominal" else -0.01
            rows.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "source_dir": "source",
                    "run_count": 2,
                    "failed_count": 0,
                    "success_rate": 1.0,
                    "final_distance_mean": 0.5 + delta,
                    "steps_mean": 10 + delta,
                    "min_obstacle_clearance_mean": 0.3,
                    "mean_terrain_risk_mean": 0.2 + delta,
                    "control_smoothness_mean": 0.01,
                    "control_jerk_mean": 0.02,
                    "mean_mppi_time_ms_mean": 4.0,
                    "success_rate_delta": 0.0,
                    "final_distance_delta": delta,
                    "steps_delta": delta,
                    "min_obstacle_clearance_delta": 0.0,
                    "mean_terrain_risk_delta": delta,
                    "control_smoothness_delta": 0.0,
                    "control_jerk_delta": 0.0,
                    "mean_mppi_time_ms_delta": 1.0,
                }
            )
    stage5_summary.write_text(json.dumps({"rows": rows, "errors": []}), encoding="utf-8")

    stage5_deltas = tmp_path / "stage5_deltas.csv"
    pd.DataFrame(
        [
            {
                "scenario_key": "id_random",
                "controller_label": label,
                "episode_id": idx,
                "delta_final_distance": -0.1,
                "delta_steps": -1,
                "delta_mean_terrain_risk": -0.01,
                "delta_control_smoothness": 0.001,
                "delta_control_jerk": 0.001,
                "delta_min_obstacle_clearance": 0.0,
            }
            for idx, label in enumerate(("Default", "Efficiency", "Balanced"))
        ]
    ).to_csv(stage5_deltas, index=False)

    stage5e_stats = tmp_path / "stage5e_stats.csv"
    stat_rows = []
    for scenario in ("low_friction_patch", "safe_corridor", "risk_band", "two_obstacle_standard"):
        for metric in ("cumulative_terrain_risk", "terrain_risk_excess", "terrain_risk_exposure_ratio", "final_distance"):
            stat_rows.append(
                {
                    "scenario_name": scenario,
                    "case_name": f"{scenario}_risk_w_1",
                    "terrain_risk_weight": 1.0,
                    "metric": metric,
                    "count": 2,
                    "mean_delta": -0.1,
                    "median_delta": -0.1,
                    "std_delta": 0.01,
                    "bootstrap_ci95_low": -0.2,
                    "bootstrap_ci95_high": -0.01,
                    "pct_improved": 1.0,
                    "wilcoxon_p": 0.1,
                    "wilcoxon_method": "unit",
                }
            )
    pd.DataFrame(stat_rows).to_csv(stage5e_stats, index=False)
    stage5e_main = tmp_path / "stage5e_main.csv"
    pd.DataFrame([{"scenario_name": "low_friction_patch", "case_name": "case", "terrain_risk_weight": 1.0}]).to_csv(stage5e_main, index=False)

    visual_dir = tmp_path / "visual"
    visual_dir.mkdir()
    runs = {}
    for key in ("nominal_risk_off", "nominal_risk_on", "learned_risk_off", "learned_risk_on"):
        traj = visual_dir / f"{key}.csv"
        pd.DataFrame({"x": [0, 1, 2], "y": [0, 0.1, 0.2]}).to_csv(traj, index=False)
        runs[key] = {"trajectory_csv": str(traj)}
    stage5e_visual = tmp_path / "stage5e_visual.json"
    stage5e_visual.write_text(json.dumps({"metadata": {"risk_weight": 1.0}, "runs": runs}), encoding="utf-8")

    return {
        "stage4_seed_summary": stage4_seed,
        "stage4_rollout": rollout,
        "ood_obstacle": ood_obstacle,
        "ood_terrain": ood_terrain,
        "stage5_summary": stage5_summary,
        "stage5_deltas": stage5_deltas,
        "stage5e_stats": stage5e_stats,
        "stage5e_main": stage5e_main,
        "stage5e_visual": stage5e_visual,
    }


def _write_ood_dir(path: Path, *, test_mse: float, zero: float, ade: float, nominal: float) -> Path:
    path.mkdir()
    (path / "ood_residual_metrics.json").write_text(
        json.dumps(
            {
                "test_mse": test_mse,
                "zero_residual_test_mse": zero,
                "overall_test_mse_reduction_pct": 98.0,
                "overall_test_improvement_x": zero / test_mse,
            }
        ),
        encoding="utf-8",
    )
    (path / "ood_rollout_metrics.json").write_text(
        json.dumps(
            {
                "learned_ade_xy": ade,
                "nominal_ade_xy": nominal,
                "learned_fde_xy": ade * 2,
                "nominal_fde_xy": nominal * 2,
            }
        ),
        encoding="utf-8",
    )
    return path


def module_controllers() -> tuple[str, ...]:
    return ("nominal", "default g1.0", "current 0.5/3.5/1.0", "balanced 0.5/3.0/0.75")
