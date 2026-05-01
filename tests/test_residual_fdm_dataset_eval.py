import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from tests.test_residual_fdm_training import write_dataset


def load_training_module():
    module_path = Path("tools/train_residual_fdm.py")
    spec = importlib.util.spec_from_file_location("train_residual_fdm_tool_for_eval_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_dataset_eval_module():
    module_path = Path("tools/evaluate_residual_fdm_dataset.py")
    spec = importlib.util.spec_from_file_location("evaluate_residual_fdm_dataset_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_residual_fdm_dataset_writes_ood_metrics(tmp_path):
    training = load_training_module()
    dataset_eval = load_dataset_eval_module()
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "ood_eval"
    write_dataset(dataset_dir)
    training.train_residual_fdm(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        epochs=3,
        batch_size=8,
        hidden_dim=16,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
    )

    metrics = dataset_eval.evaluate_residual_fdm_dataset(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        checkpoint="best_model.pt",
        normalization="normalization.npz",
        device="cpu",
        command="python3 tools/evaluate_residual_fdm_dataset.py --unit-test",
    )

    assert (output_dir / "ood_residual_metrics.json").exists()
    saved = json.loads((output_dir / "ood_residual_metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert metrics["dataset_dir"] == str(dataset_dir)
    assert metrics["checkpoint_path"] == str(model_dir / "best_model.pt")
    assert metrics["normalization_path"] == str(model_dir / "normalization.npz")
    assert metrics["device"] == "cpu"
    assert metrics["command"] == "python3 tools/evaluate_residual_fdm_dataset.py --unit-test"
    for split in ("val", "test"):
        assert metrics[f"{split}_mse"] >= 0.0
        assert metrics[f"zero_residual_{split}_mse"] >= 0.0
        assert np.isfinite(metrics[f"overall_{split}_improvement_x"])
        assert set(metrics[f"per_axis_{split}_mse_reduction_pct"]) == {"vx", "vy", "wz"}
        for axis in ("vx", "vy", "wz"):
            assert np.isfinite(metrics[f"{split}_mse_{axis}"])
            assert np.isfinite(metrics[f"{split}_rmse_{axis}"])
