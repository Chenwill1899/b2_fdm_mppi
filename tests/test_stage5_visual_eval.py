import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def load_visual_module():
    module_path = Path("tools/visualize_stage5_closed_loop.py")
    spec = importlib.util.spec_from_file_location("stage5_visual_eval_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_run_dir(path: Path, *, final_distance: float, steps: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "success": True,
        "reached_goal": True,
        "failed": False,
        "final_distance": final_distance,
        "steps": steps,
        "run_time": steps * 0.1,
        "path_length": 3.0,
        "min_obstacle_clearance": 0.2,
        "mean_terrain_risk": 0.1,
        "mean_cmd_real_error": 0.02,
        "mean_residual_norm": 0.03,
        "control_smoothness": 0.04,
        "control_jerk": 0.05,
        "mean_mppi_time_ms": 10.0,
        "max_mppi_time_ms": 20.0,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        {
            "step": [0, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.2 if steps == 3 else -0.2, 0.0],
            "theta": [0.0, 0.0, 0.0],
            "vx": [0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0],
            "wz": [0.0, 0.0, 0.0],
        }
    ).to_csv(path / "trajectory.csv", index=False)


def test_write_closed_loop_comparison_outputs_overlay_and_metric_deltas(tmp_path):
    module = load_visual_module()
    nominal_dir = tmp_path / "nominal"
    learned_dir = tmp_path / "learned"
    output_dir = tmp_path / "visual"
    write_run_dir(nominal_dir, final_distance=0.5, steps=4)
    write_run_dir(learned_dir, final_distance=0.3, steps=3)

    result = module.write_closed_loop_comparison(
        nominal_results_path=nominal_dir,
        learned_results_path=learned_dir,
        config_path="config/b2_omni_oracle.yaml",
        output_dir=output_dir,
        scenario_name="unit",
        seed=123,
    )

    assert Path(result["overlay_png"]).exists()
    assert Path(result["metrics_csv"]).exists()
    assert Path(result["metrics_json"]).exists()
    assert result["metrics"]["final_distance"]["nominal"] == pytest.approx(0.5)
    assert result["metrics"]["final_distance"]["learned"] == pytest.approx(0.3)
    assert result["metrics"]["final_distance"]["delta"] == pytest.approx(-0.2)
    assert result["metrics"]["steps"]["delta"] == pytest.approx(-1.0)


def test_run_visual_eval_enables_visual_outputs_and_writes_summary(tmp_path):
    module = load_visual_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            created_configs.append(config)
            self.results_path = Path(config["results"]["root"]) / config["results"]["run_name"]
            controller = "learned" if config.get("fdm", {}).get("enabled") else "nominal"
            final_distance = 0.3 if controller == "learned" else 0.5
            steps = 3 if controller == "learned" else 4
            write_run_dir(self.results_path, final_distance=final_distance, steps=steps)
            (self.results_path / "trajectory.png").write_bytes(b"fake")
            (self.results_path / "animation.gif").write_bytes(b"fake")

        def run(self):
            return SimpleNamespace(
                steps=3,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.3,
            )

    output_dir = tmp_path / "visual_eval"
    summary = module.run_visual_eval(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="unit",
        output_dir=output_dir,
        seed=123,
        backend="cuda",
        fdm_model_dir="results/fdm_baselines/stage4_mlp_seed123_hardened",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cuda",
        fdm_residual_gain=0.5,
        command="python3 tools/visualize_stage5_closed_loop.py --unit-test",
        argv=["python3", "tools/visualize_stage5_closed_loop.py", "--unit-test"],
        runner_cls=FakeRunner,
    )

    saved = json.loads((output_dir / "stage5_visual_eval_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["metadata"]["backend"] == "cuda"
    assert summary["metadata"]["seed"] == 123
    assert summary["metadata"]["fdm_residual_gain"] == pytest.approx(0.5)
    assert summary["comparison"]["metrics"]["final_distance"]["delta"] == pytest.approx(-0.2)
    assert Path(summary["comparison"]["overlay_png"]).exists()

    nominal_config, learned_config = created_configs
    assert nominal_config["results"]["enable_plots"] is True
    assert nominal_config["results"]["enable_animation"] is True
    assert "fdm" not in nominal_config or nominal_config["fdm"].get("enabled") is not True
    assert learned_config["results"]["enable_plots"] is True
    assert learned_config["results"]["enable_animation"] is True
    assert learned_config["fdm"]["enabled"] is True
    assert learned_config["fdm"]["device"] == "cuda"
    assert learned_config["fdm"]["residual_gain"] == pytest.approx(0.5)


def test_run_visual_eval_accepts_torch_backend(tmp_path):
    module = load_visual_module()

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            self.results_path = Path(config["results"]["root"]) / config["results"]["run_name"]
            controller = "learned" if config.get("fdm", {}).get("enabled") else "nominal"
            write_run_dir(
                self.results_path,
                final_distance=0.25 if controller == "learned" else 0.35,
                steps=3 if controller == "learned" else 4,
            )
            (self.results_path / "trajectory.png").write_bytes(b"fake")
            (self.results_path / "animation.gif").write_bytes(b"fake")

        def run(self):
            return SimpleNamespace(
                steps=3,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.3,
            )

    summary = module.run_visual_eval(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="unit",
        output_dir=tmp_path / "visual_eval_torch",
        seed=123,
        backend="torch",
        fdm_device="cpu",
        runner_cls=FakeRunner,
    )

    assert summary["metadata"]["backend"] == "torch"
    assert summary["metadata"]["fdm_device"] == "cpu"


def test_run_risk_aware_visual_eval_builds_four_cases(tmp_path):
    module = load_visual_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            created_configs.append(config)
            self.results_path = Path(config["results"]["root"]) / config["results"]["run_name"]
            controller = "learned" if config.get("fdm", {}).get("enabled") else "nominal"
            risk_weight = float(config["mppi"]["terrain_risk_weight"])
            write_run_dir(
                self.results_path,
                final_distance=0.3 + risk_weight * 0.01 if controller == "learned" else 0.5,
                steps=3 if controller == "learned" else 4,
            )
            (self.results_path / "trajectory.png").write_bytes(b"fake")
            (self.results_path / "animation.gif").write_bytes(b"fake")

        def run(self):
            return SimpleNamespace(
                steps=3,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.3,
            )

    output_dir = tmp_path / "risk_visual_eval"
    summary = module.run_risk_aware_visual_eval(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="two_obstacle_standard",
        output_dir=output_dir,
        seed=123,
        backend="torch",
        fdm_device="cpu",
        fdm_residual_gain=0.5,
        risk_weight=3.0,
        learned_goal_xy_weight=3.0,
        learned_smooth_weight=0.75,
        runner_cls=FakeRunner,
    )

    assert (output_dir / "stage5_e_visual_eval_summary.json").is_file()
    assert list(summary["runs"]) == [
        "nominal_risk_off",
        "nominal_risk_on",
        "learned_risk_off",
        "learned_risk_on",
    ]
    assert [config["mppi"]["terrain_risk_weight"] for config in created_configs] == [0.0, 3.0, 0.0, 3.0]
    assert created_configs[2]["fdm"]["enabled"] is True
    assert created_configs[2]["fdm"]["residual_gain"] == pytest.approx(0.5)
    assert created_configs[2]["mppi"]["weights"][0] == pytest.approx(3.0)
    assert created_configs[2]["mppi"]["smooth_weight"] == pytest.approx(0.75)
