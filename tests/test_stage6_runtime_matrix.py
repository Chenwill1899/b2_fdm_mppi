import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_runtime_matrix_module():
    module_path = Path("tools/profile_stage6_runtime_matrix.py")
    spec = importlib.util.spec_from_file_location("stage6_runtime_matrix_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeProfiledController:
    def __init__(self, case_name):
        self.case_name = case_name

    def profile_summary(self):
        multiplier = 2.0 if "learned" in self.case_name else 1.0
        return {
            "enabled": True,
            "total_calls": 2,
            "totals_ms": {
                "rollout_total_ms": 10.0 * multiplier,
                "sample_candidates_ms": 2.0 * multiplier,
                "update_distribution_ms": 4.0 * multiplier,
            },
            "means_ms": {
                "rollout_total_ms": 5.0 * multiplier,
                "sample_candidates_ms": 1.0 * multiplier,
                "update_distribution_ms": 2.0 * multiplier,
            },
            "counts": {
                "rollout_total_ms": 2,
                "sample_candidates_ms": 2,
                "update_distribution_ms": 2,
            },
        }


def test_runtime_matrix_profiles_fixed_seed_2x2_and_writes_summary(tmp_path):
    module = load_runtime_matrix_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            created_configs.append(config)
            self.config = config
            run_name = config["results"]["run_name"]
            self.controller = FakeProfiledController(run_name)
            self.results_path = Path(config["results"]["root"]) / run_name
            self.results_path.mkdir(parents=True, exist_ok=True)

        def run(self):
            run_name = self.config["results"]["run_name"]
            learned = "learned" in run_name
            summary = {
                "success": True,
                "failed": False,
                "steps": 2,
                "final_distance": 0.3 if learned else 0.4,
                "mean_mppi_time_ms": 20.0 if learned else 10.0,
                "cumulative_terrain_risk": 1.0,
            }
            (self.results_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return SimpleNamespace(
                steps=2,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.2,
            )

    summary = module.run_runtime_matrix(
        config_path="config/b2_omni_oracle_random100_dataset.yaml",
        scenario_name="unit_random",
        output_dir=tmp_path,
        episodes=1,
        steps=2,
        base_seed=123,
        backend="torch",
        device="cpu",
        risk_weight=3.0,
        fdm_model_dir="model",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_residual_gain=0.5,
        learned_goal_xy_weight=3.0,
        learned_smooth_weight=0.75,
        runner_cls=FakeRunner,
        command="python3 tools/profile_stage6_runtime_matrix.py --unit-test",
        argv=["python3", "tools/profile_stage6_runtime_matrix.py", "--unit-test"],
    )

    saved = json.loads((tmp_path / "stage6_runtime_matrix_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["metadata"]["backend"] == "torch"
    assert summary["metadata"]["device"] == "cpu"
    assert summary["metadata"]["force_steps"] is True
    assert summary["metadata"]["seeds"] == [123]
    assert {run["case"] for run in summary["runs"]} == {
        "nominal_risk_off",
        "nominal_risk_on",
        "learned_risk_off",
        "learned_risk_on",
    }
    assert all(run["profile"]["enabled"] is True for run in summary["runs"])
    assert all(run["profile_total_calls"] == 2 for run in summary["runs"])
    assert summary["profile_call_consistency"]["consistent"] is True
    assert summary["profile_call_consistency"]["expected_calls_per_run"] == 2
    assert "learned_risk_on_vs_nominal_risk_on" in summary["paired_deltas"]
    learned_vs_nominal = summary["paired_deltas"]["learned_risk_on_vs_nominal_risk_on"]["aggregate"]
    assert learned_vs_nominal["profile_mean_sample_candidates_ms_delta_mean"] == pytest.approx(1.0)
    assert learned_vs_nominal["profile_mean_update_distribution_ms_delta_mean"] == pytest.approx(2.0)

    configs_by_name = {config["results"]["run_name"]: config for config in created_configs}
    nominal_off = configs_by_name["unit_random_episode_0000_nominal_risk_off"]
    learned_on = configs_by_name["unit_random_episode_0000_learned_risk_on"]
    assert nominal_off["scenario"]["random_seed"] == 123
    assert learned_on["scenario"]["random_seed"] == 123
    assert nominal_off["mppi"]["backend"] == "torch"
    assert nominal_off["mppi"]["profile_enabled"] is True
    assert nominal_off["mppi"]["terrain_risk_weight"] == pytest.approx(0.0)
    assert "fdm" not in nominal_off
    assert learned_on["fdm"]["enabled"] is True
    assert learned_on["fdm"]["profile_enabled"] is True
    assert learned_on["fdm"]["residual_gain"] == pytest.approx(0.5)
    assert learned_on["simulation"]["disable_goal_termination"] is True
    assert learned_on["mppi"]["terrain_risk_weight"] == pytest.approx(3.0)
    assert learned_on["mppi"]["weights"][0] == pytest.approx(3.0)
    assert learned_on["mppi"]["smooth_weight"] == pytest.approx(0.75)
