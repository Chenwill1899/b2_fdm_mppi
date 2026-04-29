import json

import numpy as np

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
    assert (summary.results_path / "time_results.csv").exists()
    assert (summary.results_path / "test_summary.yaml").exists()


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


def test_omni_runner_can_overwrite_named_results_directory(tmp_path):
    config = make_config(tmp_path)
    config["results"]["run_name"] = "latest"
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
