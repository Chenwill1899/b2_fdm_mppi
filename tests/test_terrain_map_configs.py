import importlib.util
import json
import sys
from pathlib import Path

from b2_fdm_mppi.config import load_config


COMPLEX_TERRAIN_CONFIGS = [
    "config/b2_omni_oracle_risk_band.yaml",
    "config/b2_omni_oracle_risk_island.yaml",
    "config/b2_omni_oracle_low_friction_patch.yaml",
    "config/b2_omni_oracle_safe_corridor.yaml",
]


def load_inspect_module():
    module_path = Path("tools/inspect_terrain_maps.py")
    spec = importlib.util.spec_from_file_location("inspect_terrain_maps_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_complex_terrain_configs_load_with_physical_patches():
    for path in COMPLEX_TERRAIN_CONFIGS:
        config = load_config(path)
        patches = config["terrain"]["patches"]
        assert config["terrain"]["enabled"] is True
        assert patches
        assert all(patch["type"] in {"ellipse", "band"} for patch in patches)
        assert config["obstacles"]["virtual"] == []


def test_inspect_terrain_maps_writes_artifacts_and_path_risk_summary(tmp_path):
    module = load_inspect_module()

    summary = module.inspect_configs(
        config_paths=[
            Path("config/b2_omni_oracle_risk_band.yaml"),
            Path("config/b2_omni_oracle_risk_island.yaml"),
        ],
        output_dir=tmp_path,
        grid_resolution=40,
        path_samples=80,
    )

    saved = json.loads((tmp_path / "terrain_map_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert {item["scenario"] for item in summary["maps"]} == {
        "b2_omni_oracle_risk_band",
        "b2_omni_oracle_risk_island",
    }
    for item in summary["maps"]:
        scenario_dir = tmp_path / item["scenario"]
        assert (scenario_dir / "terrain_risk_map.png").exists()
        assert (scenario_dir / "terrain_feature_maps.png").exists()
        assert (scenario_dir / "terrain_map_summary.json").exists()
        assert item["risk_max"] > item["risk_mean"] > item["risk_min"]
        assert item["patch_coverage_ratio"] > 0.0
        assert item["straight_path_cumulative_risk"] > item["best_detour_cumulative_risk"]
