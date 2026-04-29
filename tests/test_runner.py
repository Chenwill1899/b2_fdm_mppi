import numpy as np
import pytest
from pathlib import Path

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.runner import MppiSimulationRunner


class FakeController:
    def __init__(self):
        self.calls = 0

    def compute_control(self, state, cost_params):
        self.calls += 1
        optimal_u = np.zeros((3, 2), dtype=np.float32)
        sample_u = np.zeros((2, 3, 2), dtype=np.float32)
        return np.array([0.5, 0.0], dtype=np.float32), optimal_u, sample_u, 1.0, 0.25


def test_runner_advances_internal_simulation_with_injected_controller(tmp_path):
    config = load_config("config/fdm_mppi.yaml")
    config["simulation"]["max_steps"] = 3
    config["simulation"]["time_horizon"] = 0.3
    config["mppi"]["draw_num_traj"] = 2
    config["mppi"]["std_normal"] = [0.1, 0.1]
    config["results"]["root"] = str(tmp_path)
    config["results"]["enable_plots"] = False
    controller = FakeController()

    runner = MppiSimulationRunner(config, controller_factory=lambda *_args, **_kwargs: controller)
    summary = runner.run()

    assert summary.steps == 3
    assert controller.calls == 3
    assert runner.state_history[-1][0] > runner.state_history[0][0]
    assert (summary.results_path / "results.csv").exists()
    assert (summary.results_path / "obs_results.csv").exists()


def test_runner_results_keep_legacy_velocity_column_names(tmp_path):
    config = load_config("config/fdm_mppi.yaml")
    config["simulation"]["max_steps"] = 1
    config["simulation"]["time_horizon"] = 0.3
    config["mppi"]["draw_num_traj"] = 2
    config["mppi"]["std_normal"] = [0.1, 0.1]
    config["results"]["root"] = str(tmp_path)
    config["results"]["enable_plots"] = False

    runner = MppiSimulationRunner(config, controller_factory=lambda *_args, **_kwargs: FakeController())
    summary = runner.run()

    header = (summary.results_path / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "dx" in header.split(",")
    assert "dy" in header.split(",")


def test_runner_continues_when_animation_fails(tmp_path, monkeypatch):
    config = load_config("config/fdm_mppi.yaml")
    config["simulation"]["max_steps"] = 1
    config["simulation"]["time_horizon"] = 0.3
    config["mppi"]["draw_num_traj"] = 2
    config["mppi"]["std_normal"] = [0.1, 0.1]
    config["results"]["root"] = str(tmp_path)
    config["results"]["enable_plots"] = True
    config["results"]["enable_animation"] = True

    from b2_fdm_mppi.visualization import utils

    monkeypatch.setattr(utils, "statePlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "controlPlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "costPlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "pathPlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "plot_cbf", lambda *_args, **_kwargs: None)

    def fail_animation(*_args, **_kwargs):
        raise RuntimeError("animation writer unavailable")

    monkeypatch.setattr(utils, "animate_simulation", fail_animation)

    runner = MppiSimulationRunner(config, controller_factory=lambda *_args, **_kwargs: FakeController())

    with pytest.warns(RuntimeWarning, match="Animation failed"):
        summary = runner.run()

    assert summary.steps == 1
    assert not summary.failed
    assert (summary.results_path / "results.csv").exists()


def test_runner_saves_gif_when_animation_is_enabled(tmp_path, monkeypatch):
    config = load_config("config/fdm_mppi.yaml")
    config["simulation"]["max_steps"] = 1
    config["simulation"]["time_horizon"] = 0.3
    config["mppi"]["draw_num_traj"] = 2
    config["mppi"]["std_normal"] = [0.1, 0.1]
    config["results"]["root"] = str(tmp_path)
    config["results"]["enable_plots"] = True
    config["results"]["enable_animation"] = True

    from b2_fdm_mppi.visualization import utils

    monkeypatch.setattr(utils, "statePlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "controlPlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "costPlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "pathPlotting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils, "plot_cbf", lambda *_args, **_kwargs: None)

    def save_fake_animation(*args, **_kwargs):
        results_path = args[6]
        (Path(results_path) / "animation.gif").write_bytes(b"GIF89a")

    monkeypatch.setattr(utils, "animate_simulation", save_fake_animation)

    runner = MppiSimulationRunner(config, controller_factory=lambda *_args, **_kwargs: FakeController())
    summary = runner.run()

    assert (summary.results_path / "animation.gif").exists()
