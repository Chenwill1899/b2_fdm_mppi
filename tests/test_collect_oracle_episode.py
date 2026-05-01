import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_collect_module():
    module_path = Path("tools/collect_oracle_episode.py")
    spec = importlib.util.spec_from_file_location("collect_oracle_episode_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_oracle_episode_uses_episode_scoped_raw_results_dir(tmp_path, monkeypatch):
    module = load_collect_module()
    output_path = tmp_path / "oracle_debug" / "episodes" / "episode_000042.npz"
    captured = {}

    def fake_load_config(_config_path):
        return {
            "scenario": {},
            "oracle_residual": {},
            "mppi": {"backend": "cuda"},
            "results": {
                "root": str(tmp_path / "shared_results_root"),
                "run_name": "fixed_name",
                "timestamp_suffix": True,
            },
        }

    class FakeRunner:
        def __init__(self, config, controller_factory=None):
            captured["config"] = config
            captured["controller_factory"] = controller_factory

        def run(self):
            return SimpleNamespace(results_path=Path(captured["config"]["results"]["root"]) / "episode_000042")

    def fake_build_episode_npz(results_path, episode_id, npz_path):
        captured["build_args"] = (Path(results_path), episode_id, Path(npz_path))
        return {"episode_id": episode_id, "success": True, "failed": False, "num_transitions": 3}

    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "OmniMppiSimulationRunner", FakeRunner)
    monkeypatch.setattr(module, "build_episode_npz", fake_build_episode_npz)

    metadata = module.collect_oracle_episode(
        config_path="config.yaml",
        episode_id=42,
        seed=165,
        output_path=output_path,
        backend="numpy",
    )

    results_config = captured["config"]["results"]
    assert results_config["root"] == str(tmp_path / "oracle_debug" / "raw_results")
    assert results_config["run_name"] == "episode_000042"
    assert results_config["timestamp_suffix"] is False
    assert results_config["overwrite"] is True
    assert captured["config"]["scenario"]["random_seed"] == 165
    assert captured["config"]["oracle_residual"]["seed"] == 165
    assert captured["config"]["mppi"]["backend"] == "numpy"
    assert captured["build_args"] == (
        tmp_path / "oracle_debug" / "raw_results" / "episode_000042",
        42,
        output_path,
    )
    assert metadata["num_transitions"] == 3
