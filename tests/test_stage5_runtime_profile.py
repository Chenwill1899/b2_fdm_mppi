import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_profile_module():
    module_path = Path("tools/profile_stage5_learned_torch.py")
    spec = importlib.util.spec_from_file_location("stage5_runtime_profile_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeProfiledController:
    def profile_summary(self):
        return {
            "enabled": True,
            "total_calls": 2,
            "totals_ms": {"rollout_total_ms": 4.0, "fdm_inference_ms": 1.5},
            "means_ms": {"rollout_total_ms": 2.0, "fdm_inference_ms": 0.75},
        }


def test_run_profile_enables_learned_torch_profiling_and_writes_summary(tmp_path):
    module = load_profile_module()
    created_configs = []

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            created_configs.append(config)
            self.config = config
            self.controller = FakeProfiledController()
            self.results_path = Path(config["results"]["root"]) / config["results"]["run_name"]
            self.results_path.mkdir(parents=True, exist_ok=True)

        def run(self):
            (self.results_path / "summary.json").write_text(
                json.dumps({"success": True, "steps": 2, "mean_mppi_time_ms": 3.0}),
                encoding="utf-8",
            )
            return SimpleNamespace(
                steps=2,
                reached_goal=True,
                failed=False,
                results_path=self.results_path,
                run_time=0.2,
            )

    summary = module.run_profile(
        config_path="config/b2_omni_oracle.yaml",
        output_dir=tmp_path,
        steps=2,
        seed=123,
        fdm_model_dir="model",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cpu",
        fdm_residual_gain=0.5,
        runner_cls=FakeRunner,
        command="python3 tools/profile_stage5_learned_torch.py --unit-test",
        argv=["python3", "tools/profile_stage5_learned_torch.py", "--unit-test"],
    )

    saved = json.loads((tmp_path / "stage5_runtime_profile_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    config = created_configs[0]
    assert config["mppi"]["backend"] == "cuda"
    assert config["fdm"]["enabled"] is True
    assert config["fdm"]["profile_enabled"] is True
    assert config["fdm"]["residual_gain"] == pytest.approx(0.5)
    assert config["simulation"]["max_steps"] == 2
    assert summary["controller_profile"]["total_calls"] == 2
