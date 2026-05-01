import json
from pathlib import Path

import numpy as np

from b2_fdm_mppi.data.oracle_dataset_validator import validate_oracle_dataset


def write_split(path: Path, episode_id: int, transitions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = np.full((transitions, 3), 0.5, dtype=np.float32)
    real = cmd + np.array([0.1, -0.2, 0.05], dtype=np.float32)
    np.savez(
        path,
        states=np.column_stack(
            [
                np.linspace(episode_id, episode_id + 1.0, transitions),
                np.linspace(0.0, 1.0, transitions),
                np.zeros(transitions),
                np.ones(transitions),
                np.zeros(transitions),
                np.zeros(transitions),
            ]
        ).astype(np.float32),
        next_states=np.column_stack(
            [
                np.linspace(episode_id + 0.1, episode_id + 1.1, transitions),
                np.linspace(0.1, 1.1, transitions),
                np.zeros(transitions),
                np.ones(transitions),
                np.zeros(transitions),
                np.zeros(transitions),
            ]
        ).astype(np.float32),
        cmd_controls=cmd,
        real_controls=real,
        exec_residuals=real - cmd,
        oracle_residuals=np.full((transitions, 3), 0.05, dtype=np.float32),
        terrain_features=np.column_stack(
            [
                np.zeros(transitions),
                np.zeros(transitions),
                np.linspace(0.1, 0.3, transitions),
                np.linspace(0.8, 0.7, transitions),
            ]
        ).astype(np.float32),
        terrain_risk=np.linspace(0.2, 0.4, transitions).astype(np.float32),
        episode_ids=np.full(transitions, episode_id, dtype=np.int64),
        steps=np.arange(transitions, dtype=np.int64),
    )


def test_validate_oracle_dataset_writes_quality_json_and_summary_png(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_split(dataset_dir / "train.npz", episode_id=0, transitions=3)
    write_split(dataset_dir / "val.npz", episode_id=1, transitions=2)
    write_split(dataset_dir / "test.npz", episode_id=2, transitions=1)
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
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"episodes": 1, "transitions": 3},
                    "val": {"episodes": 1, "transitions": 2},
                    "test": {"episodes": 1, "transitions": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    quality = validate_oracle_dataset(dataset_dir, dataset_dir)

    assert (dataset_dir / "dataset_quality.json").exists()
    assert (dataset_dir / "dataset_summary.png").exists()
    saved_quality = json.loads((dataset_dir / "dataset_quality.json").read_text(encoding="utf-8"))
    assert saved_quality == quality
    assert quality["pass"] is True
    assert quality["nan_count"] == 0
    assert quality["inf_count"] == 0
    assert quality["episode_leakage_check"]["pass"] is True
    assert quality["episode_leakage_check"]["overlaps"] == []
    assert quality["max_exec_residual_error"] == 0.0
    assert quality["mean_exec_residual_error"] == 0.0
    assert quality["zero_residual_baseline_mse"] > 0.0
    assert quality["num_transitions"] == {"train": 3, "val": 2, "test": 1}
    assert quality["num_episodes"] == {"train": 1, "val": 1, "test": 1}
    assert quality["split_shapes"]["train"]["states"] == [3, 6]


def test_validate_oracle_dataset_detects_episode_leakage(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_split(dataset_dir / "train.npz", episode_id=0, transitions=2)
    write_split(dataset_dir / "val.npz", episode_id=0, transitions=2)
    write_split(dataset_dir / "test.npz", episode_id=2, transitions=2)
    (dataset_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "train_episode_ids": [0],
                "val_episode_ids": [0],
                "test_episode_ids": [2],
                "skipped_episode_ids": [],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"episodes": 1, "transitions": 2},
                    "val": {"episodes": 1, "transitions": 2},
                    "test": {"episodes": 1, "transitions": 2},
                }
            }
        ),
        encoding="utf-8",
    )

    quality = validate_oracle_dataset(dataset_dir, dataset_dir)

    assert quality["pass"] is False
    assert quality["episode_leakage_check"]["pass"] is False
    assert quality["episode_leakage_check"]["overlaps"]
