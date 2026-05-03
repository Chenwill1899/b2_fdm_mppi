from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from b2_fdm_mppi.experiment import build_experiment_config, run_experiment_profile


def write_profile(path: Path) -> Path:
    profile = {
        "experiment": {
            "name": "unit_profile",
            "base_config": "configs/smoke.yaml",
            "seed": 123,
            "output_root": str(path.parent / "runs"),
        },
        "scenario": {
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "goal": [2.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            "world_mode": "oracle",
            "max_steps": 30,
            "minimum_distance": 0.35,
            "obstacles": {
                "virtual": [[1.0, 0.2, 0.25, 0.0, 0.0, 0.0, 0.0]],
                "static_enabled": True,
            },
            "terrain": {"enabled": True, "friction_base": 0.65},
        },
        "controllers": [
            {
                "name": "nominal_cuda",
                "method": "nominal",
                "backend": "cuda",
                "mppi": {"num_trajectories": 64, "draw_num_traj": 8},
            },
            {
                "name": "learned_torch",
                "method": "learned_fdm",
                "backend": "torch",
                "mppi": {"num_trajectories": 32},
                "learned_fdm": {
                    "model_dir": "results/fdm_baselines/demo",
                    "checkpoint": "best_model.pt",
                    "normalization": "normalization.npz",
                    "device": "cuda",
                    "residual_gain": 0.75,
                },
            },
        ],
        "default_controller": "nominal_cuda",
        "visualization": {"plots": True, "animation": False, "terrain_grid_resolution": 24},
    }
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def test_build_experiment_config_maps_scene_controller_model_and_visuals(tmp_path):
    profile_path = write_profile(tmp_path / "profile.yaml")

    config, metadata = build_experiment_config(
        profile_path,
        controller_name="learned_torch",
        seed=456,
        output_root=tmp_path / "custom_runs",
        model_dir="results/fdm_baselines/override",
    )

    assert metadata["experiment_name"] == "unit_profile"
    assert metadata["controller_name"] == "learned_torch"
    assert metadata["method"] == "learned_fdm"
    assert metadata["seed"] == 456
    assert config["simulation"]["goal"] == [2.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    assert config["simulation"]["max_steps"] == 30
    assert config["simulation"]["minimum_distance"] == 0.35
    assert config["obstacles"]["virtual"] == [[1.0, 0.2, 0.25, 0.0, 0.0, 0.0, 0.0]]
    assert config["terrain"]["friction_base"] == 0.65
    assert config["visualization"]["terrain_grid_resolution"] == 24
    assert config["mppi"]["backend"] == "torch"
    assert config["mppi"]["num_trajectories"] == 32
    assert config["fdm"]["enabled"] is True
    assert config["fdm"]["model_dir"] == "results/fdm_baselines/override"
    assert config["fdm"]["checkpoint"] == "best_model.pt"
    assert config["fdm"]["normalization"] == "normalization.npz"
    assert config["fdm"]["device"] == "cuda"
    assert config["fdm"]["residual_gain"] == 0.75
    assert config["results"]["root"] == str(tmp_path / "custom_runs")
    assert config["results"]["run_name"] == "unit_profile_learned_torch_seed456"
    assert config["results"]["enable_plots"] is True
    assert config["results"]["enable_animation"] is False


def test_run_experiment_profile_writes_artifact_manifest(tmp_path):
    profile_path = write_profile(tmp_path / "profile.yaml")

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            self.config = config
            self.results_path = Path(config["results"]["root"]) / config["results"]["run_name"]

        def run(self):
            self.results_path.mkdir(parents=True, exist_ok=True)
            (self.results_path / "summary.json").write_text(
                json.dumps({"success": True, "controller_type": "FakeController"}),
                encoding="utf-8",
            )
            for name in ["trajectory.csv", "controls.csv", "residuals.csv", "terrain.csv"]:
                (self.results_path / name).write_text("step\n0\n", encoding="utf-8")
            (self.results_path / "trajectory.png").write_bytes(b"png")
            (self.results_path / "animation.gif").write_bytes(b"gif")
            return SimpleNamespace(
                steps=2,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.2,
            )

    result = run_experiment_profile(
        profile_path,
        controller_name="nominal_cuda",
        seed=321,
        backend="numpy",
        runner_cls=FakeRunner,
    )

    manifest_path = Path(result["artifacts"]["experiment_summary"])
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["controller_name"] == "nominal_cuda"
    assert result["backend"] == "numpy"
    assert result["reached_goal"] is True
    assert result["artifacts"]["summary_json"].endswith("summary.json")
    assert result["artifacts"]["trajectory_png"].endswith("trajectory.png")
    assert result["artifacts"]["animation_gif"].endswith("animation.gif")
    assert saved["seed"] == 321
    assert saved["artifacts"]["experiment_summary"] == str(manifest_path)
