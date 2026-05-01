import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


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


def test_load_residual_fdm_dataset_builds_expected_features(tmp_path):
    module = load_training_module()
    dataset_dir = tmp_path / "dataset"
    write_dataset(dataset_dir)

    arrays = module.load_residual_fdm_dataset(dataset_dir)

    assert arrays["train_features"].shape == (32, 14)
    assert arrays["train_targets"].shape == (32, 3)
    assert arrays["val_features"].shape == (12, 14)
    assert arrays["test_targets"].shape == (10, 3)


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
    )

    assert (output_dir / "model.pt").exists()
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
