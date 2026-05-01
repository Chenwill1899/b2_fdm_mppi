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


def test_omni_runner_trajectory_includes_final_state_after_last_action(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path, max_steps=3),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    summary = runner.run()
    trajectory = pd.read_csv(summary.results_path / "trajectory.csv")

    assert len(trajectory) == summary.steps + 1
    assert trajectory.iloc[-1]["step"] == summary.steps
    assert trajectory.iloc[-1]["x"] == pytest.approx(float(runner.state[0]))
    assert trajectory.iloc[-1]["y"] == pytest.approx(float(runner.state[1]))


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


def test_omni_runner_uses_random_obstacles_and_records_summary(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    config["simulation"]["world_mode"] = "oracle"
    config["simulation"]["initial_state"] = [5.0, 50.0, 0.0, 0.0, 0.0, 0.0]
    config["simulation"]["goal"] = [95.0, 50.0, 0.0, 0.0, 0.0, 0.0]
    config["obstacles"] = {
        "random_enabled": True,
        "random_seed": 9,
        "num_random": 4,
        "radius_range": [0.5, 1.0],
        "x_range": [10.0, 90.0],
        "y_range": [10.0, 90.0],
        "min_obstacle_gap": 1.0,
        "min_start_goal_clearance": 5.0,
        "virtual": [],
    }
    config["oracle_residual"] = {"enabled": False}
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    assert runner.obstacles.shape == (4, 7)
    assert runner.obstacle_mode == "random"
    assert runner.obstacle_random_seed == 9

    summary = runner.run()
    summary_json = json.loads((summary.results_path / "summary.json").read_text())

    assert summary_json["obstacle_mode"] == "random"
    assert summary_json["num_obstacles"] == 4
    assert summary_json["obstacle_random_seed"] == 9
    assert (summary.results_path / "obs_results.csv").exists()


def test_omni_runner_random_start_goal_overrides_fixed_state_and_updates_goal_relief(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    config["simulation"]["world_mode"] = "oracle"
    config["simulation"]["initial_state"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["simulation"]["goal"] = [99.0, 99.0, 0.0, 0.0, 0.0, 0.0]
    config["scenario"] = {
        "random_start_goal_enabled": True,
        "random_seed": 11,
        "x_range": [5.0, 20.0],
        "y_range": [5.0, 20.0],
        "distance_range": [5.0, 10.0],
        "min_obstacle_clearance": 1.0,
        "max_attempts": 500,
        "start_yaw": 0.1,
        "goal_yaw": 0.2,
    }
    config["terrain"] = {
        "enabled": True,
        "goal_relief": {
            "enabled": True,
            "center": [99.0, 99.0],
            "sigma": [2.0, 1.2],
            "strength": 0.75,
            "floor": 0.25,
        },
    }
    config["oracle_residual"] = {"enabled": False}

    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    assert not np.allclose(runner.init_pose[:2], [0.0, 0.0])
    assert not np.allclose(runner.goal[:2], [99.0, 99.0])
    assert runner.init_pose[2] == pytest.approx(0.1)
    assert runner.goal[2] == pytest.approx(0.2)
    assert runner.config["terrain"]["goal_relief"]["center"] == pytest.approx(runner.goal[:2].tolist())
    assert runner.terrain.goal_relief["center"] == pytest.approx(runner.goal[:2].tolist())

    summary = runner.run()
    summary_json = json.loads((summary.results_path / "summary.json").read_text())
    saved_config = load_config(summary.results_path / "config.yaml")

    assert summary_json["scenario_mode"] == "random_start_goal"
    assert summary_json["scenario_random_seed"] == 11
    assert 5.0 <= summary_json["start_goal_distance"] <= 10.0
    assert saved_config["terrain"]["goal_relief"]["center"] == pytest.approx(runner.goal[:2].tolist())


def test_omni_runner_resolves_auto_scenario_seed_and_records_it(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    config["simulation"]["world_mode"] = "oracle"
    config["scenario"] = {
        "random_start_goal_enabled": True,
        "random_seed": "auto",
        "x_range": [5.0, 20.0],
        "y_range": [5.0, 20.0],
        "distance_range": [5.0, 10.0],
        "min_obstacle_clearance": 1.0,
        "max_attempts": 500,
        "start_yaw": 0.0,
        "goal_yaw": 0.0,
    }
    config["obstacles"] = {
        "random_enabled": True,
        "random_seed": 123,
        "num_random": 2,
        "radius_range": [0.5, 0.7],
        "x_range": [30.0, 40.0],
        "y_range": [30.0, 40.0],
        "min_obstacle_gap": 1.0,
        "min_start_goal_clearance": 2.0,
        "virtual": [],
    }
    config["oracle_residual"] = {"enabled": False}

    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )
    assert isinstance(runner.config["scenario"]["random_seed"], int)
    assert runner.scenario_random_seed == runner.config["scenario"]["random_seed"]
    assert runner.obstacle_random_seed == 123

    summary = runner.run()
    summary_json = json.loads((summary.results_path / "summary.json").read_text())
    saved_config = load_config(summary.results_path / "config.yaml")

    assert isinstance(summary_json["scenario_random_seed"], int)
    assert saved_config["scenario"]["random_seed"] == summary_json["scenario_random_seed"]
    assert saved_config["obstacles"]["random_seed"] == 123


def test_omni_runner_fixed_obstacles_record_summary(tmp_path):
    runner = OmniMppiSimulationRunner(
        make_config(tmp_path, max_steps=1),
        controller_factory=lambda *_args, **_kwargs: ConstantOmniController(),
    )

    summary = runner.run()
    summary_json = json.loads((summary.results_path / "summary.json").read_text())

    assert runner.obstacle_mode == "fixed"
    assert summary_json["obstacle_mode"] == "fixed"
    assert summary_json["num_obstacles"] == len(runner.obstacles)
    assert summary_json["obstacle_random_seed"] is None
    assert summary_json["scenario_mode"] == "fixed"
    assert summary_json["scenario_random_seed"] is None
    assert summary_json["start_goal_distance"] == pytest.approx(
        float(np.linalg.norm(runner.goal[:2] - runner.init_pose[:2]))
    )


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
