import argparse
import importlib.util
import sys
from pathlib import Path

from b2_fdm_mppi.config import load_config


def load_run_module():
    module_path = Path("tools/run_omni_mppi.py")
    spec = importlib.util.spec_from_file_location("run_omni_mppi_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_cli_overrides_enables_learned_fdm_and_backend():
    module = load_run_module()
    config = load_config("config/b2_omni_oracle.yaml")
    args = argparse.Namespace(
        backend="numpy",
        fdm_enabled=True,
        fdm_model_dir="results/fdm_baselines/stage4_mlp_seed123_hardened",
        fdm_checkpoint="best_model.pt",
        fdm_normalization="normalization.npz",
        fdm_device="cpu",
        fdm_residual_gain=0.5,
    )

    updated = module.apply_cli_overrides(config, args)

    assert updated["mppi"]["backend"] == "numpy"
    assert updated["fdm"]["enabled"] is True
    assert updated["fdm"]["model_dir"] == "results/fdm_baselines/stage4_mlp_seed123_hardened"
    assert updated["fdm"]["checkpoint"] == "best_model.pt"
    assert updated["fdm"]["normalization"] == "normalization.npz"
    assert updated["fdm"]["device"] == "cpu"
    assert updated["fdm"]["residual_gain"] == 0.5


def test_apply_cli_overrides_leaves_fdm_disabled_by_default():
    module = load_run_module()
    config = load_config("config/b2_omni_oracle.yaml")
    args = argparse.Namespace(
        backend=None,
        fdm_enabled=False,
        fdm_model_dir=None,
        fdm_checkpoint=None,
        fdm_normalization=None,
        fdm_device=None,
        fdm_residual_gain=None,
    )

    updated = module.apply_cli_overrides(config, args)

    assert updated["mppi"]["backend"] == "cuda"
    assert "fdm" not in updated or updated["fdm"].get("enabled") is not True
