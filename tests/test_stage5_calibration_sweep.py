import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_sweep_module():
    module_path = Path("tools/sweep_stage5_calibration.py")
    spec = importlib.util.spec_from_file_location("stage5_calibration_sweep_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expand_sweep_cases_builds_residual_gain_and_cost_grid_product():
    module = load_sweep_module()

    cases = module.expand_sweep_cases(
        residual_gains=[0.0, 0.5],
        cost_grid={"smooth_weight": [0.5, 1.0]},
    )

    assert [case["fdm_residual_gain"] for case in cases] == [0.0, 0.0, 0.5, 0.5]
    assert [case["learned_mppi_overrides"] for case in cases] == [
        {"smooth_weight": 0.5},
        {"smooth_weight": 1.0},
        {"smooth_weight": 0.5},
        {"smooth_weight": 1.0},
    ]
    assert len({case["case_name"] for case in cases}) == 4


def test_parse_cost_grid_entries_uses_supported_benchmark_override_keys():
    module = load_sweep_module()

    grid = module.parse_cost_grid_entries(["smooth_weight=0.5,1.0", "goal_xy_weight=2.5"])

    assert grid == {"smooth_weight": [0.5, 1.0], "goal_xy_weight": [2.5]}


def test_run_calibration_sweep_writes_case_summaries(tmp_path):
    module = load_sweep_module()
    calls = []

    def fake_benchmark(**kwargs):
        calls.append(kwargs)
        gain = kwargs["fdm_residual_gain"]
        return {
            "metadata": {"output_dir": str(kwargs["output_dir"])},
            "aggregates": {
                "learned": {"success_rate": 1.0, "final_distance_mean": 0.4 + gain},
                "nominal": {"success_rate": 1.0, "final_distance_mean": 0.5},
            },
            "paired_deltas": {"aggregate": {"final_distance_delta_mean": gain - 0.1}},
        }

    summary = module.run_calibration_sweep(
        config_path="config/b2_omni_oracle.yaml",
        scenario_name="unit",
        output_dir=tmp_path,
        episodes=1,
        base_seed=123,
        backend="numpy",
        controllers=("nominal", "learned"),
        fdm_model_dir="model",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cpu",
        residual_gains=[0.0, 0.5],
        cost_grid={},
        benchmark_fn=fake_benchmark,
        command="python3 tools/sweep_stage5_calibration.py --unit-test",
        argv=["python3", "tools/sweep_stage5_calibration.py", "--unit-test"],
    )

    saved = json.loads((tmp_path / "stage5_calibration_sweep_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert len(calls) == 2
    assert calls[0]["fdm_residual_gain"] == pytest.approx(0.0)
    assert calls[1]["fdm_residual_gain"] == pytest.approx(0.5)
    assert summary["cases"][1]["learned_final_distance_mean"] == pytest.approx(0.9)
