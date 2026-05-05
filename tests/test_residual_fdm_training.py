import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def load_training_module():
    module_path = Path("tools/train_residual_fdm.py")
    spec = importlib.util.spec_from_file_location("train_residual_fdm_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_split(path: Path, transitions: int, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.zeros((transitions, 6), dtype=np.float32)
    states[:, 0] = np.linspace(0.0, 1.0, transitions, dtype=np.float32) + offset
    states[:, 3] = 0.2 + offset
    cmd_controls = np.zeros((transitions, 3), dtype=np.float32)
    cmd_controls[:, 0] = np.linspace(-0.5, 0.5, transitions, dtype=np.float32)
    cmd_controls[:, 1] = 0.1 + offset
    terrain_features = np.zeros((transitions, 4), dtype=np.float32)
    terrain_features[:, 2] = 0.2 + offset
    terrain_features[:, 3] = 0.8
    terrain_risk = np.linspace(0.1, 0.3, transitions, dtype=np.float32)
    target = np.column_stack(
        [
            0.2 * cmd_controls[:, 0] + 0.1 * terrain_features[:, 2],
            -0.1 * cmd_controls[:, 1],
            0.05 * terrain_risk,
        ]
    ).astype(np.float32)
    np.savez(
        path,
        states=states,
        next_states=states.copy(),
        cmd_controls=cmd_controls,
        real_controls=cmd_controls + target,
        exec_residuals=target,
        oracle_residuals=target.copy(),
        terrain_features=terrain_features,
        terrain_risk=terrain_risk,
        episode_ids=np.zeros(transitions, dtype=np.int64),
        steps=np.arange(transitions, dtype=np.int64),
    )


def write_dataset(dataset_dir: Path) -> None:
    write_split(dataset_dir / "train.npz", transitions=32, offset=0.0)
    write_split(dataset_dir / "val.npz", transitions=12, offset=0.1)
    write_split(dataset_dir / "test.npz", transitions=10, offset=0.2)
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps({"total_transitions": 54}),
        encoding="utf-8",
    )
    (dataset_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "train_episode_ids": [0],
                "val_episode_ids": [1],
                "test_episode_ids": [2],
                "skipped_episode_ids": [],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "dataset_quality.json").write_text(
        json.dumps(
            {
                "pass": True,
                "num_transitions": 54,
                "zero_residual_baseline_mse": 0.01,
            }
        ),
        encoding="utf-8",
    )


def current_git_branch() -> str | None:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def test_load_residual_fdm_dataset_builds_expected_features(tmp_path):
    module = load_training_module()
    dataset_dir = tmp_path / "dataset"
    write_dataset(dataset_dir)

    arrays = module.load_residual_fdm_dataset(dataset_dir)

    assert arrays["train_features"].shape == (32, 14)
    assert arrays["train_targets"].shape == (32, 3)
    assert arrays["val_features"].shape == (12, 14)
    assert arrays["test_targets"].shape == (10, 3)


def test_load_sequence_fdm_dataset_aligns_relative_future_states_and_interleaved_features(tmp_path):
    module = load_training_module()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for split in ("train", "val", "test"):
        states = np.zeros((5, 6), dtype=np.float32)
        states[:, 0] = np.arange(5, dtype=np.float32)
        states[:, 2] = 0.1
        next_states = states.copy()
        next_states[:, 0] = np.arange(1, 6, dtype=np.float32)
        next_states[:, 1] = np.arange(10, 15, dtype=np.float32)
        next_states[:, 2] = 0.2
        cmd_controls = np.arange(15, dtype=np.float32).reshape(5, 3)
        terrain_features = np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0
        terrain_risk = np.arange(5, dtype=np.float32) / 10.0
        np.savez(
            dataset_dir / f"{split}.npz",
            states=states,
            next_states=next_states,
            cmd_controls=cmd_controls,
            real_controls=cmd_controls,
            exec_residuals=np.zeros((5, 3), dtype=np.float32),
            oracle_residuals=np.zeros((5, 3), dtype=np.float32),
            terrain_features=terrain_features,
            terrain_risk=terrain_risk,
            episode_ids=np.zeros(5, dtype=np.int64),
            steps=np.arange(5, dtype=np.int64),
        )

    arrays = module.load_sequence_fdm_dataset(dataset_dir, sequence_horizon=3)

    features = arrays["train_features"]
    targets = arrays["train_targets"]
    assert features.shape == (2, 6 + 3 + 3 * 8)
    assert targets.shape == (2, 12)
    np.testing.assert_allclose(
        targets[0].reshape(3, 4),
        np.asarray(
            [
                [1.0, 10.0, 0.1, 0.1],
                [2.0, 11.0, 0.1, 0.2],
                [3.0, 12.0, 0.1, 0.3],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    per_step = features[0, 9:].reshape(3, 8)
    assert per_step[0] == pytest.approx([0.0, 1.0, 2.0, 0.0, 0.1, 0.2, 0.3, 0.0], abs=1e-6)
    assert per_step[1] == pytest.approx([3.0, 4.0, 5.0, 0.0, 0.1, 0.2, 0.3, 0.0], abs=1e-6)


def test_train_residual_fdm_writes_checkpoint_and_metrics(tmp_path):
    module = load_training_module()
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    write_dataset(dataset_dir)

    metrics = module.train_residual_fdm(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=3,
        batch_size=8,
        hidden_dim=16,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
        command="python3 tools/train_residual_fdm.py --unit-test",
        argv=["python3", "tools/train_residual_fdm.py", "--unit-test"],
    )

    assert (output_dir / "model.pt").exists()
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "normalization.npz").exists()
    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert metrics["epochs"] == 3
    assert metrics["train_loss"][-1] >= 0.0
    assert metrics["val_loss"][-1] >= 0.0
    assert metrics["zero_residual_val_mse"] >= 0.0
    assert metrics["val_mse"] >= 0.0
    assert metrics["test_mse"] >= 0.0
    assert metrics["tensorboard_enabled"] is True
    assert metrics["tensorboard_log_dir"] == str(output_dir / "tensorboard")
    assert list((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert metrics["command"] == "python3 tools/train_residual_fdm.py --unit-test"
    assert metrics["argv"] == ["python3", "tools/train_residual_fdm.py", "--unit-test"]
    assert metrics["sys_argv"] == ["python3", "tools/train_residual_fdm.py", "--unit-test"]
    assert metrics["git_sha"]
    assert metrics["git_branch"] == current_git_branch()
    assert isinstance(metrics["git_dirty"], bool)
    assert metrics["device"] == "cpu"
    assert metrics["dataset_dir"] == str(dataset_dir)
    assert metrics["split_manifest_path"] == str(dataset_dir / "split_manifest.json")
    assert metrics["dataset_summary_path"] == str(dataset_dir / "dataset_summary.json")
    assert metrics["dataset_quality_path"] == str(dataset_dir / "dataset_quality.json")
    assert metrics["output_dir"] == str(output_dir)
    assert metrics["batch_size"] == 8
    assert metrics["hidden_dim"] == 16
    assert metrics["learning_rate"] == 1e-2
    assert metrics["weight_decay"] == 1e-5

    best_epoch = int(np.argmin(metrics["val_loss"]) + 1)
    assert metrics["best_epoch"] == best_epoch
    assert metrics["best_val_loss"] == metrics["val_loss"][best_epoch - 1]
    assert metrics["final_epoch"] == 3
    assert metrics["final_val_loss"] == metrics["val_loss"][-1]
    assert metrics["checkpoint_policy"] == "best_model.pt tracks minimum validation standardized loss; model.pt stores final epoch"
    assert metrics["best_checkpoint_path"] == str(output_dir / "best_model.pt")
    assert metrics["final_checkpoint_path"] == str(output_dir / "model.pt")

    for split in ("val", "test"):
        for axis in ("vx", "vy", "wz"):
            for name in (
                f"{split}_mse_{axis}",
                f"{split}_rmse_{axis}",
                f"zero_residual_{split}_mse_{axis}",
                f"{split}_mse_reduction_pct_{axis}",
            ):
                assert name in metrics
                assert np.isfinite(metrics[name])
        assert f"{split}_mse_relative_improvement_pct" in metrics
        assert np.isfinite(metrics[f"{split}_mse_relative_improvement_pct"])
        assert f"per_axis_{split}_mse_reduction_pct" in metrics
        assert set(metrics[f"per_axis_{split}_mse_reduction_pct"]) == {"vx", "vy", "wz"}
    assert metrics["overall_val_mse_reduction_pct"] == metrics["val_mse_relative_improvement_pct"]
    assert metrics["overall_test_mse_reduction_pct"] == metrics["test_mse_relative_improvement_pct"]
    assert np.isfinite(metrics["overall_val_improvement_x"])
    assert np.isfinite(metrics["overall_test_improvement_x"])


def test_train_residual_fdm_writes_tensorboard_to_custom_log_dir(tmp_path):
    module = load_training_module()
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    tensorboard_dir = tmp_path / "tb_custom"
    write_dataset(dataset_dir)

    metrics = module.train_residual_fdm(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        tensorboard_log_dir=tensorboard_dir,
        epochs=2,
        batch_size=8,
        hidden_dim=16,
        learning_rate=1e-2,
        seed=7,
    )

    assert metrics["tensorboard_enabled"] is True
    assert metrics["tensorboard_log_dir"] == str(tensorboard_dir)
    assert list(tensorboard_dir.glob("events.out.tfevents.*"))
