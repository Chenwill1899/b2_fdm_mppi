import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_benchmark_module():
    module_path = Path("tools/benchmark_learned_fdm_mppi.py")
    spec = importlib.util.spec_from_file_location("stage5_benchmark_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_runs_handles_success_rate_stats_and_missing_values():
    module = load_benchmark_module()
    runs = [
        {
            "scenario": "standard",
            "controller": "nominal",
            "episode_id": 0,
            "seed": 123,
            "success": True,
            "failed": False,
            "final_distance": 0.4,
            "arrival_time": 10.0,
            "min_obstacle_clearance": None,
            "mean_mppi_time_ms": 12.0,
        },
        {
            "scenario": "standard",
            "controller": "nominal",
            "episode_id": 1,
            "seed": 124,
            "success": False,
            "failed": True,
            "final_distance": 1.2,
            "arrival_time": None,
            "min_obstacle_clearance": 0.2,
            "mean_mppi_time_ms": 14.0,
        },
        {
            "scenario": "standard",
            "controller": "learned",
            "episode_id": 0,
            "seed": 123,
            "success": True,
            "failed": False,
            "final_distance": 0.3,
            "arrival_time": 9.0,
            "min_obstacle_clearance": 0.25,
            "mean_mppi_time_ms": 1000.0,
        },
    ]

    aggregates = module.aggregate_runs(runs)

    assert aggregates["nominal"]["count"] == 2
    assert aggregates["nominal"]["success_rate"] == 0.5
    assert aggregates["nominal"]["final_distance_mean"] == pytest.approx(0.8)
    assert aggregates["nominal"]["final_distance_std"] == pytest.approx(0.4)
    assert aggregates["nominal"]["arrival_time_mean"] == pytest.approx(10.0)
    assert aggregates["nominal"]["min_obstacle_clearance_mean"] == pytest.approx(0.2)
    assert aggregates["learned"]["count"] == 1
    assert aggregates["learned"]["success_rate"] == 1.0


def test_compute_paired_deltas_matches_by_scenario_episode_and_seed():
    module = load_benchmark_module()
    runs = [
        {
            "scenario": "standard",
            "controller": "nominal",
            "episode_id": 0,
            "seed": 123,
            "success": True,
            "failed": False,
            "final_distance": 0.5,
            "steps": 20,
            "mean_mppi_time_ms": 10.0,
        },
        {
            "scenario": "standard",
            "controller": "learned",
            "episode_id": 0,
            "seed": 123,
            "success": True,
            "failed": False,
            "final_distance": 0.3,
            "steps": 18,
            "mean_mppi_time_ms": 900.0,
        },
        {
            "scenario": "standard",
            "controller": "learned",
            "episode_id": 1,
            "seed": 124,
            "success": True,
            "failed": False,
            "final_distance": 0.2,
            "steps": 15,
            "mean_mppi_time_ms": 850.0,
        },
    ]

    paired = module.compute_paired_deltas(runs)

    assert len(paired["pairs"]) == 1
    pair = paired["pairs"][0]
    assert pair["scenario"] == "standard"
    assert pair["episode_id"] == 0
    assert pair["seed"] == 123
    assert pair["final_distance_delta"] == pytest.approx(-0.2)
    assert pair["steps_delta"] == pytest.approx(-2.0)
    assert pair["mean_mppi_time_ms_delta"] == pytest.approx(890.0)
    assert paired["aggregate"]["final_distance_delta_mean"] == pytest.approx(-0.2)


def test_run_benchmark_writes_summary_and_configures_nominal_and_learned(tmp_path):
    module = load_benchmark_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            self.config = config
            self.controller_factory = controller_factory
            created_configs.append(config)
            results = Path(config["results"]["root"]) / config["results"]["run_name"]
            results.mkdir(parents=True, exist_ok=True)
            self.results_path = results

        def run(self):
            controller = "learned" if self.config.get("fdm", {}).get("enabled") else "nominal"
            summary = {
                "success": controller == "learned",
                "reached_goal": controller == "learned",
                "failed": False,
                "final_distance": 0.3 if controller == "learned" else 0.5,
                "steps": 18 if controller == "learned" else 20,
                "arrival_time": 1.8 if controller == "learned" else 2.0,
                "path_length": 4.0,
                "min_obstacle_clearance": 0.25,
                "mean_terrain_risk": 0.1,
                "max_terrain_risk": 0.3,
                "cumulative_terrain_risk": 1.0,
                "terrain_risk_excess": 0.2,
                "terrain_risk_excess_integral": 0.05,
                "terrain_risk_exposure_ratio": 0.4,
                "mean_cmd_real_error": 0.04,
                "mean_residual_norm": 0.05,
                "control_smoothness": 0.2,
                "control_jerk": 0.3,
                "mean_mppi_time_ms": 900.0 if controller == "learned" else 12.0,
                "max_mppi_time_ms": 930.0 if controller == "learned" else 20.0,
            }
            (self.results_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return SimpleNamespace(
                steps=summary["steps"],
                reached_goal=summary["reached_goal"],
                failed=summary["failed"],
                results_path=self.results_path,
                run_time=summary["arrival_time"],
            )

    output_dir = tmp_path / "benchmark"
    summary = module.run_benchmark(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="unit",
        output_dir=output_dir,
        episodes=1,
        base_seed=123,
        backend="numpy",
        controllers=("nominal", "learned"),
        fdm_model_dir="results/fdm_baselines/stage4_mlp_seed123_hardened",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cpu",
        fdm_residual_gain=0.75,
        mppi_overrides={
            "terrain_risk_weight": 3.0,
            "terrain_risk_power": 2.0,
            "terrain_risk_threshold": 0.25,
            "terrain_risk_mode": "excess",
        },
        learned_mppi_overrides={"goal_xy_weight": 3.5, "smooth_weight": 0.4},
        command="python3 tools/benchmark_learned_fdm_mppi.py --unit-test",
        argv=["python3", "tools/benchmark_learned_fdm_mppi.py", "--unit-test"],
        runner_cls=FakeRunner,
    )

    summary_path = output_dir / "stage5_benchmark_summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["metadata"]["command"] == "python3 tools/benchmark_learned_fdm_mppi.py --unit-test"
    assert summary["metadata"]["argv"] == ["python3", "tools/benchmark_learned_fdm_mppi.py", "--unit-test"]
    assert summary["metadata"]["backend"] == "numpy"
    assert summary["metadata"]["seeds"] == [123]
    assert summary["metadata"]["fdm_checkpoint"] == "best_model.pt"
    assert summary["metadata"]["fdm_residual_gain"] == pytest.approx(0.75)
    assert summary["metadata"]["learned_mppi_overrides"] == {
        "goal_xy_weight": 3.5,
        "smooth_weight": 0.4,
    }
    assert summary["metadata"]["mppi_overrides"] == {
        "terrain_risk_weight": 3.0,
        "terrain_risk_power": 2.0,
        "terrain_risk_threshold": 0.25,
        "terrain_risk_mode": "excess",
    }
    assert len(summary["runs"]) == 2
    assert summary["runs"][0]["cumulative_terrain_risk"] == pytest.approx(1.0)
    assert summary["aggregates"]["learned"]["success_rate"] == 1.0
    assert summary["paired_deltas"]["aggregate"]["final_distance_delta_mean"] == pytest.approx(-0.2)

    nominal_config, learned_config = created_configs
    assert nominal_config["mppi"]["backend"] == "numpy"
    assert nominal_config["results"]["enable_plots"] is False
    assert nominal_config["results"]["enable_animation"] is False
    assert "fdm" not in nominal_config or nominal_config["fdm"].get("enabled") is not True
    assert learned_config["mppi"]["backend"] == "numpy"
    assert learned_config["fdm"]["enabled"] is True
    assert learned_config["fdm"]["model_dir"] == "results/fdm_baselines/stage4_mlp_seed123_hardened"
    assert learned_config["fdm"]["checkpoint"] == "best_model.pt"
    assert learned_config["fdm"]["normalization"] == "normalization.npz"
    assert learned_config["fdm"]["device"] == "cpu"
    assert learned_config["fdm"]["residual_gain"] == pytest.approx(0.75)
    assert learned_config["mppi"]["weights"][0] == pytest.approx(3.5)
    assert learned_config["mppi"]["smooth_weight"] == pytest.approx(0.4)
    assert nominal_config["mppi"]["weights"][0] != pytest.approx(3.5)
    assert nominal_config["mppi"]["smooth_weight"] != pytest.approx(0.4)
    for config in (nominal_config, learned_config):
        assert config["mppi"]["terrain_risk_weight"] == pytest.approx(3.0)
        assert config["mppi"]["terrain_risk_power"] == pytest.approx(2.0)
        assert config["mppi"]["terrain_risk_threshold"] == pytest.approx(0.25)
        assert config["mppi"]["terrain_risk_mode"] == "excess"


def test_run_benchmark_rejects_unsupported_backend(tmp_path):
    module = load_benchmark_module()

    with pytest.raises(ValueError, match="supports only numpy, cuda, or torch"):
        module.run_benchmark(
            config_path="config/b2_omni_oracle.yaml",
            scenario_name="unit",
            output_dir=tmp_path,
            episodes=1,
            base_seed=123,
            backend="bad",
            controllers=("nominal", "learned"),
            fdm_model_dir="model",
            fdm_checkpoint="best_model.pt",
            fdm_normalization="normalization.npz",
            fdm_device="cpu",
        )


def test_run_benchmark_allows_cuda_backend_in_run_configs(tmp_path):
    module = load_benchmark_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            created_configs.append(config)
            results = Path(config["results"]["root"]) / config["results"]["run_name"]
            results.mkdir(parents=True, exist_ok=True)
            self.results_path = results

        def run(self):
            summary = {
                "success": True,
                "reached_goal": True,
                "failed": False,
                "final_distance": 0.3,
                "steps": 2,
                "arrival_time": 0.2,
                "path_length": 0.3,
                "min_obstacle_clearance": 0.4,
                "mean_terrain_risk": 0.1,
                "max_terrain_risk": 0.3,
                "cumulative_terrain_risk": 0.2,
                "terrain_risk_excess": 0.0,
                "terrain_risk_excess_integral": 0.0,
                "terrain_risk_exposure_ratio": 0.0,
                "mean_cmd_real_error": 0.0,
                "mean_residual_norm": 0.0,
                "control_smoothness": 0.0,
                "control_jerk": 0.0,
                "mean_mppi_time_ms": 1.0,
                "max_mppi_time_ms": 2.0,
            }
            (self.results_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return SimpleNamespace(
                steps=2,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.2,
            )

    module.run_benchmark(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="unit",
        output_dir=tmp_path / "benchmark",
        episodes=1,
        base_seed=123,
        backend="cuda",
        controllers=("nominal", "learned"),
        fdm_model_dir="results/fdm_baselines/stage4_mlp_seed123_hardened",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cuda",
        fdm_residual_gain=0.25,
        runner_cls=FakeRunner,
    )

    nominal_config, learned_config = created_configs
    assert nominal_config["mppi"]["backend"] == "cuda"
    assert "fdm" not in nominal_config or nominal_config["fdm"].get("enabled") is not True
    assert learned_config["mppi"]["backend"] == "cuda"
    assert learned_config["fdm"]["enabled"] is True
    assert learned_config["fdm"]["device"] == "cuda"
    assert learned_config["fdm"]["residual_gain"] == pytest.approx(0.25)


def test_run_benchmark_allows_torch_backend_in_run_configs(tmp_path):
    module = load_benchmark_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            created_configs.append(config)
            results = Path(config["results"]["root"]) / config["results"]["run_name"]
            results.mkdir(parents=True, exist_ok=True)
            self.results_path = results

        def run(self):
            summary = {
                "success": True,
                "reached_goal": True,
                "failed": False,
                "final_distance": 0.3,
                "steps": 2,
                "arrival_time": 0.2,
                "path_length": 0.3,
                "min_obstacle_clearance": 0.4,
                "mean_terrain_risk": 0.1,
                "max_terrain_risk": 0.3,
                "cumulative_terrain_risk": 0.2,
                "terrain_risk_excess": 0.0,
                "terrain_risk_excess_integral": 0.0,
                "terrain_risk_exposure_ratio": 0.0,
                "mean_cmd_real_error": 0.0,
                "mean_residual_norm": 0.0,
                "control_smoothness": 0.0,
                "control_jerk": 0.0,
                "mean_mppi_time_ms": 1.0,
                "max_mppi_time_ms": 2.0,
            }
            (self.results_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return SimpleNamespace(
                steps=2,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.2,
            )

    module.run_benchmark(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="unit",
        output_dir=tmp_path / "benchmark",
        episodes=1,
        base_seed=123,
        backend="torch",
        controllers=("nominal", "learned"),
        fdm_model_dir="results/fdm_baselines/stage4_mlp_seed123_hardened",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cpu",
        fdm_residual_gain=0.25,
        runner_cls=FakeRunner,
    )

    nominal_config, learned_config = created_configs
    assert nominal_config["mppi"]["backend"] == "torch"
    assert nominal_config["mppi"]["device"] == "cpu"
    assert "fdm" not in nominal_config or nominal_config["fdm"].get("enabled") is not True
    assert learned_config["mppi"]["backend"] == "torch"
    assert learned_config["mppi"]["device"] == "cpu"
    assert learned_config["fdm"]["enabled"] is True
    assert learned_config["fdm"]["device"] == "cpu"
    assert learned_config["fdm"]["residual_gain"] == pytest.approx(0.25)


def test_parse_mppi_overrides_rejects_unknown_cost_key():
    module = load_benchmark_module()

    with pytest.raises(ValueError, match="Unsupported learned MPPI override"):
        module.parse_learned_mppi_overrides(["bad_weight=1.0"])


def test_parse_mppi_overrides_parses_supported_cost_keys():
    module = load_benchmark_module()

    overrides = module.parse_learned_mppi_overrides(
        ["goal_xy_weight=4.0", "obstacle_weight=120.5", "yaw_rate_weight=0.03"]
    )

    assert overrides == {
        "goal_xy_weight": 4.0,
        "obstacle_weight": 120.5,
        "yaw_rate_weight": 0.03,
    }


def test_parse_shared_mppi_overrides_accepts_numeric_and_mode_values():
    module = load_benchmark_module()

    overrides = module.parse_mppi_overrides(
        [
            "terrain_risk_weight=5.0",
            "terrain_risk_power=2",
            "terrain_risk_threshold=0.3",
            "terrain_risk_mode=excess",
        ]
    )

    assert overrides == {
        "terrain_risk_weight": 5.0,
        "terrain_risk_power": 2.0,
        "terrain_risk_threshold": 0.3,
        "terrain_risk_mode": "excess",
    }


def test_parse_shared_mppi_overrides_rejects_learned_only_keys():
    module = load_benchmark_module()

    with pytest.raises(ValueError, match="Unsupported shared MPPI override"):
        module.parse_mppi_overrides(["goal_xy_weight=4.0"])
