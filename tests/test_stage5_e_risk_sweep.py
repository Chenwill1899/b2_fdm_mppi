import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_sweep_module():
    module_path = Path("tools/sweep_stage5_e_risk_cost.py")
    spec = importlib.util.spec_from_file_location("stage5_e_risk_sweep_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expand_risk_sweep_cases_builds_config_weight_product():
    module = load_sweep_module()

    cases = module.expand_risk_sweep_cases(
        config_paths=["config/b2_omni_oracle_risk_band.yaml", "config/b2_omni_oracle_risk_island.yaml"],
        risk_weights=[0.0, 3.0],
    )

    assert [case["scenario_name"] for case in cases] == [
        "risk_band",
        "risk_band",
        "risk_island",
        "risk_island",
    ]
    assert [case["terrain_risk_weight"] for case in cases] == [0.0, 3.0, 0.0, 3.0]
    assert len({case["case_name"] for case in cases}) == 4


def test_parse_risk_weights_requires_numeric_values():
    module = load_sweep_module()

    assert module.parse_risk_weights("0,1,3.5") == [0.0, 1.0, 3.5]
    with pytest.raises(ValueError, match="At least one terrain risk weight"):
        module.parse_risk_weights("")


def test_run_risk_sweep_applies_shared_mppi_overrides_and_writes_summary(tmp_path):
    module = load_sweep_module()
    calls = []

    def fake_benchmark(**kwargs):
        calls.append(kwargs)
        weight = kwargs["mppi_overrides"]["terrain_risk_weight"]
        return {
            "metadata": {"output_dir": str(kwargs["output_dir"])},
            "aggregates": {
                "nominal": {
                    "success_rate": 1.0,
                    "final_distance_mean": 1.0 + weight,
                    "steps_mean": 10.0,
                    "cumulative_terrain_risk_mean": 5.0 - weight,
                    "terrain_risk_excess_mean": 1.0,
                    "terrain_risk_exposure_ratio_mean": 0.5,
                    "mean_mppi_time_ms_mean": 2.0,
                }
            },
            "paired_deltas": {"aggregate": {}},
        }

    summary = module.run_risk_sweep(
        config_paths=["config/b2_omni_oracle_risk_band.yaml"],
        output_dir=tmp_path,
        episodes=1,
        base_seed=123,
        backend="numpy",
        controllers=("nominal",),
        risk_weights=[0.0, 3.0],
        risk_power=2.0,
        risk_threshold=0.3,
        risk_mode="excess",
        fdm_model_dir="model",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cpu",
        benchmark_fn=fake_benchmark,
        command="python3 tools/sweep_stage5_e_risk_cost.py --unit-test",
        argv=["python3", "tools/sweep_stage5_e_risk_cost.py", "--unit-test"],
    )

    saved = json.loads((tmp_path / "stage5_e_risk_sweep_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert len(calls) == 2
    assert calls[0]["mppi_overrides"] == {
        "terrain_risk_weight": 0.0,
        "terrain_risk_power": 2.0,
        "terrain_risk_threshold": 0.3,
        "terrain_risk_mode": "excess",
    }
    assert calls[1]["mppi_overrides"]["terrain_risk_weight"] == pytest.approx(3.0)
    assert summary["cases"][1]["nominal_final_distance_mean"] == pytest.approx(4.0)
    assert summary["cases"][1]["nominal_cumulative_terrain_risk_mean"] == pytest.approx(2.0)


def test_prepare_reduced_config_writes_smoke_overrides(tmp_path):
    module = load_sweep_module()

    reduced = module.prepare_reduced_config(
        config_path="config/b2_omni_oracle_risk_band.yaml",
        output_dir=tmp_path,
        num_trajectories=32,
        time_horizon=0.7,
        max_steps=12,
        draw_num_traj=4,
    )

    import yaml

    config = yaml.safe_load(Path(reduced).read_text(encoding="utf-8"))
    assert config["mppi"]["num_trajectories"] == 32
    assert config["mppi"]["draw_num_traj"] == 4
    assert config["simulation"]["time_horizon"] == pytest.approx(0.7)
    assert config["simulation"]["max_steps"] == 12
