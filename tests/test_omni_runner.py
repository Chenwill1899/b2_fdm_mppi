import json

import numpy as np
import pandas as pd
import pytest

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner


class ConstantOmniController:
    horizon_steps = 3
    draw_num_traj = 2

    def compute_control(self, state, cost_params):
        optimal_u = np.zeros((self.horizon_steps, 3), dtype=np.float32)
        sample_u = np.zeros((self.draw_num_traj, self.horizon_steps, 3), dtype=np.float32)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), optimal_u, sample_u, 1.0, 0.25


class PlotCounter:
    def __init__(self):
        self.plot_calls = 0

    def plot(self, *_args, **_kwargs):
        self.plot_calls += 1


def make_config(tmp_path, enable_plots=False, max_steps=5):
    config = load_config("config/b2_omni_nominal.yaml")
    config["simulation"]["goal"] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["simulation"]["max_steps"] = max_steps
    config["simulation"]["time_horizon"] = 0.3
    config["mppi"]["num_trajectories"] = 8
    config["mppi"]["draw_num_traj"] = 2
    config["results"]["root"] = str(tmp_path)
    config["results"]["enable_plots"] = enable_plots
    config["results"]["enable_animation"] = enable_plots
    return config


def test_omni_runner_saves_summary_csv_outputs(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    summary = runner.run()
    summary_json = json.loads((summary.results_path / "summary.json").read_text())

    assert summary.steps == 5
    assert not summary.failed
    assert summary_json["final_distance"] < 1.0
    assert (summary.results_path / "trajectory.csv").exists()
    assert (summary.results_path / "controls.csv").exists()
    assert (summary.results_path / "residuals.csv").exists()
    assert (summary.results_path / "terrain.csv").exists()
    assert (summary.results_path / "time_results.csv").exists()
    assert (summary.results_path / "test_summary.yaml").exists()
    assert "control_smoothness" in summary_json
    assert "control_jerk" in summary_json
    assert "acceleration_cost" in summary_json
    assert "lateral_usage" in summary_json
    assert "yaw_rate_usage" in summary_json
    assert "sample_terminal_y_std_mean" in summary_json
    assert "sample_terminal_y_range_mean" in summary_json
    assert "sample_terminal_spread_mean" in summary_json
    assert "sample_terminal_x_range_mean" in summary_json
    assert "vx_variance" in summary_json
    assert "vy_variance" in summary_json
    assert "wz_variance" in summary_json
    assert summary_json["controls_csv"] == "executed_controls"
    assert "world_mode" in summary_json
    assert "mean_residual_norm" in summary_json
    assert "max_residual_norm" in summary_json
    assert "mean_cmd_real_error" in summary_json
    assert "max_cmd_real_error" in summary_json
    assert "mean_terrain_risk" in summary_json
    assert "max_terrain_risk" in summary_json


def test_omni_runner_summary_reports_control_smoothness_metrics(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    runner.control_history = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    ]

    metrics = runner._control_metrics()

    assert metrics["control_smoothness"] == 1.0
    assert metrics["smooth_vx"] == 0.5
    assert metrics["smooth_vy"] == 0.5
    assert metrics["smooth_wz"] == 0.0
    assert metrics["control_jerk"] == 2.0
    assert metrics["acceleration_cost"] == 200.0
    assert metrics["lateral_usage"] == np.mean([0.0, 0.0, 1.0])
    assert metrics["yaw_rate_usage"] == 0.0
    assert metrics["jerk_vx"] == 1.0
    assert metrics["jerk_vy"] == 1.0
    assert metrics["jerk_wz"] == 0.0
    assert metrics["vx_variance"] == np.var([0.0, 1.0, 1.0])


def test_omni_runner_reports_sample_terminal_coverage_metrics(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    runner.state_history = [np.zeros(6, dtype=np.float32)]
    runner.sample_u_history = [
        np.array(
            [
                [[0.5, -0.2, 0.0], [0.5, -0.2, 0.0]],
                [[0.5, 0.3, 0.0], [0.5, 0.3, 0.0]],
                [[0.5, 0.1, 0.0], [0.5, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
    ]

    metrics = runner._sample_coverage_metrics()

    assert metrics["sample_terminal_y_std_mean"] > 0.0
    assert metrics["sample_terminal_y_range_mean"] > 0.0
    assert metrics["sample_terminal_spread_mean"] > 0.0
    assert metrics["sample_terminal_x_range_mean"] >= 0.0


def test_omni_runner_predict_trajectory_uses_kinodynamic_rollout(tmp_path):
    config = make_config(tmp_path)
    config["execution"] = {"filter_enabled": False, "filter_alpha": 0.0}
    config["robot"]["max_ax"] = 0.8
    config["robot"]["max_ay"] = 0.5
    config["robot"]["max_awz"] = 1.2
    config["robot"]["velocity_lag_beta"] = 0.35
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    controls = np.zeros((2, 3), dtype=np.float32)
    controls[:, 0] = 1.5

    predicted = runner._predict_trajectory(np.zeros(6, dtype=np.float32), controls)

    assert predicted[1, 3] == pytest.approx(0.08, abs=1e-7)
    assert predicted[1, 0] == pytest.approx(0.008, abs=1e-7)


def test_omni_runner_predict_trajectory_rotates_body_frame_velocity_by_heading(tmp_path):
    config = make_config(tmp_path)
    config["execution"] = {"filter_enabled": False, "filter_alpha": 0.0}
    config["robot"]["max_ax"] = 1000.0
    config["robot"]["max_ay"] = 1000.0
    config["robot"]["max_awz"] = 1000.0
    config["robot"]["velocity_lag_beta"] = 0.0
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    state = np.array([2.0, 3.0, np.pi / 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    controls = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)[0]

    predicted = runner._predict_trajectory(state, controls)

    assert predicted[1, 0] == pytest.approx(2.0, abs=1e-6)
    assert predicted[1, 1] == pytest.approx(3.1, abs=1e-6)


def test_omni_runner_applies_execution_low_pass_filter(tmp_path):
    config = make_config(tmp_path, max_steps=2)
    config["execution"] = {"filter_enabled": True, "filter_alpha": 0.5}
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    runner.step()
    runner.step()

    assert runner.raw_control_history[0].tolist() == [1.0, 0.0, 0.0]
    assert runner.control_history[0].tolist() == [0.5, 0.0, 0.0]
    assert runner.control_history[1].tolist() == [0.75, 0.0, 0.0]


def test_omni_runner_saves_png_and_gif_when_enabled(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path, enable_plots=True, max_steps=4),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    summary = runner.run()

    assert (summary.results_path / "trajectory.png").exists()
    assert (summary.results_path / "animation.gif").exists()


def test_omni_runner_draws_sampled_and_optimized_rollouts(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    runner.sample_u_history.append(np.zeros((2, 3, 3), dtype=np.float32))
    runner.optimal_u_history.append(np.zeros((3, 3), dtype=np.float32))
    ax = PlotCounter()

    runner._draw_predicted_rollouts(ax, np.zeros(6, dtype=np.float32), frame=0)

    assert ax.plot_calls == 3


def test_omni_runner_oracle_animation_legend_describes_nominal_rollouts(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    labels = [handle.get_label() for handle in runner._animation_legend_handles(is_oracle=True)]

    assert "terrain risk" in labels
    assert "nominal sampled rollouts" in labels
    assert "nominal optimal rollout" in labels
    assert "actual heading" in labels
    assert "u_cmd" not in labels
    assert "u_real" not in labels


def test_omni_runner_can_overwrite_named_results_directory(tmp_path):
    config = make_config(tmp_path)
    config["results"]["run_name"] = "latest"
    config["results"]["timestamp_suffix"] = False
    config["results"]["overwrite"] = True

    first = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    first_summary = first.run()
    stale_file = first_summary.results_path / "stale.txt"
    stale_file.write_text("old", encoding="utf-8")

    second = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    second_summary = second.run()

    assert first_summary.results_path == tmp_path / "latest"
    assert second_summary.results_path == tmp_path / "latest"
    assert not stale_file.exists()
    assert (second_summary.results_path / "summary.json").exists()


def test_omni_runner_oracle_world_records_residuals(tmp_path):
    config = make_config(tmp_path, max_steps=4)
    config["simulation"]["world_mode"] = "oracle"
    config["terrain"] = {"enabled": True}
    config["oracle_residual"] = {
        "enabled": True,
        "alpha": 0.5,
        "residual_scale": 0.6,
        "noise_std": 0.0,
        "max_residual_ratio": 0.4,
        "seed": 7,
    }
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    summary = runner.run()
    summary_json = json.loads((summary.results_path / "summary.json").read_text())

    assert summary_json["world_mode"] == "oracle"
    assert summary_json["mean_residual_norm"] > 0.0
    assert (summary.results_path / "residuals.csv").exists()
    assert (summary.results_path / "terrain.csv").exists()


def test_omni_runner_residuals_csv_separates_oracle_and_execution_residuals(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    runner.cmd_control_history = [np.array([0.9, 0.0, 0.2], dtype=np.float32)]
    runner.control_history = [np.array([0.5, -0.1, 0.3], dtype=np.float32)]
    runner.residual_history = [np.array([0.8, -0.2, 0.0], dtype=np.float32)]

    runner._save_residuals()

    row = pd.read_csv(runner.results_path / "residuals.csv").iloc[0]
    assert row["oracle_du_vx"] == pytest.approx(0.8)
    assert row["oracle_du_vy"] == pytest.approx(-0.2)
    assert row["oracle_du_wz"] == pytest.approx(0.0)
    assert row["exec_du_vx"] == pytest.approx(-0.4)
    assert row["exec_du_vy"] == pytest.approx(-0.1)
    assert row["exec_du_wz"] == pytest.approx(0.1)
    assert row["exec_du_norm"] == pytest.approx(float(np.linalg.norm([-0.4, -0.1, 0.1])))
    assert row["du_vx"] == pytest.approx(row["oracle_du_vx"])
    assert row["du_norm"] == pytest.approx(row["oracle_du_norm"])


def test_omni_runner_oracle_animation_writes_diagnostic_outputs(tmp_path):
    config = make_config(tmp_path, enable_plots=True, max_steps=4)
    config["simulation"]["world_mode"] = "oracle"
    config["terrain"] = {
        "enabled": True,
        "goal_relief": {
            "enabled": True,
            "center": [1.0, 0.0],
            "sigma": [1.0, 0.8],
            "strength": 0.5,
            "floor": 0.3,
        },
    }
    config["oracle_residual"] = {
        "enabled": True,
        "alpha": 0.5,
        "residual_scale": 0.6,
        "noise_std": 0.0,
        "max_residual_ratio": 0.4,
        "seed": 7,
    }
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    summary = runner.run()

    assert (summary.results_path / "oracle_diagnostics.png").exists()
    assert (summary.results_path / "animation.gif").exists()
    assert (summary.results_path / "animation.gif").stat().st_size > 0


def test_omni_runner_cmd_real_error_uses_executed_minus_commanded_norm(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    runner.cmd_control_history = [
        np.array([0.9, 0.0, 0.0], dtype=np.float32),
        np.array([0.2, 0.2, 0.0], dtype=np.float32),
    ]
    runner.control_history = [
        np.array([0.5, 0.0, 0.0], dtype=np.float32),
        np.array([0.2, -0.1, 0.4], dtype=np.float32),
    ]
    runner.residual_history = [
        np.array([0.9, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.2], dtype=np.float32),
    ]

    metrics = runner._summary_metrics()

    cmd_real_errors = np.array([0.4, 0.5], dtype=np.float32)
    residual_norms = np.array([0.9, 0.2], dtype=np.float32)
    assert metrics["mean_cmd_real_error"] == pytest.approx(float(np.mean(cmd_real_errors)))
    assert metrics["max_cmd_real_error"] == pytest.approx(float(np.max(cmd_real_errors)))
    assert metrics["mean_residual_norm"] == pytest.approx(float(np.mean(residual_norms)))
    assert metrics["max_residual_norm"] == pytest.approx(float(np.max(residual_norms)))


def test_omni_runner_uses_cuda_backend_when_configured(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config["mppi"]["backend"] = "cuda"
    created = []

    class FakeCudaController(ConstantOmniController):
        @classmethod
        def from_config(cls, config, seed=None):
            created.append((config, seed))
            return cls()

    import b2_fdm_mppi.simulation.omni_runner as omni_runner

    monkeypatch.setattr(omni_runner, "MppiOmniCuda", FakeCudaController)
    runner = OmniMppiSimulationRunner(config)

    assert isinstance(runner.controller, FakeCudaController)
    assert created == [(config, 123)]
