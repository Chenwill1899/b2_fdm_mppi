import json
from pathlib import Path

import numpy as np

from b2_fdm_mppi.data.oracle_dataset import build_oracle_dataset


ARRAY_FIELDS = [
    "states",
    "next_states",
    "cmd_controls",
    "real_controls",
    "exec_residuals",
    "oracle_residuals",
    "terrain_features",
    "terrain_risk",
    "episode_ids",
    "steps",
]


def write_episode(path: Path, episode_id: int, transitions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        states=np.full((transitions, 6), episode_id, dtype=np.float32),
        next_states=np.full((transitions, 6), episode_id + 0.5, dtype=np.float32),
        cmd_controls=np.full((transitions, 3), episode_id, dtype=np.float32),
        real_controls=np.full((transitions, 3), episode_id + 1.0, dtype=np.float32),
        exec_residuals=np.ones((transitions, 3), dtype=np.float32),
        oracle_residuals=np.full((transitions, 3), 0.25, dtype=np.float32),
        terrain_features=np.full((transitions, 4), episode_id, dtype=np.float32),
        terrain_risk=np.full(transitions, episode_id, dtype=np.float32),
        episode_ids=np.full(transitions, episode_id, dtype=np.int64),
        steps=np.arange(transitions, dtype=np.int64),
        success=np.asarray(True),
        failed=np.asarray(False),
        start_goal_distance=np.asarray(5.0 + episode_id, dtype=np.float32),
        final_distance=np.asarray(0.2, dtype=np.float32),
        min_obstacle_clearance=np.asarray(1.0, dtype=np.float32),
    )


def write_manifest(input_dir: Path, rows: list[dict]) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_build_oracle_dataset_splits_by_episode_and_merges_arrays(tmp_path):
    input_dir = tmp_path / "oracle_debug"
    episodes_dir = input_dir / "episodes"
    rows = []
    for episode_id, transitions in enumerate([2, 3, 4, 5]):
        path = episodes_dir / f"episode_{episode_id:06d}.npz"
        write_episode(path, episode_id, transitions)
        rows.append(
            {
                "episode_id": episode_id,
                "seed": 100 + episode_id,
                "path": str(path),
                "success": True,
                "failed": False,
                "num_transitions": transitions,
            }
        )
    write_manifest(input_dir, rows)

    output_dir = tmp_path / "oracle_dataset"
    summary = build_oracle_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        split_seed=7,
    )

    assert (output_dir / "train.npz").exists()
    assert (output_dir / "val.npz").exists()
    assert (output_dir / "test.npz").exists()
    assert (output_dir / "split_manifest.json").exists()
    assert (output_dir / "dataset_summary.json").exists()
    split_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert sorted(
        split_manifest["train_episode_ids"]
        + split_manifest["val_episode_ids"]
        + split_manifest["test_episode_ids"]
    ) == [0, 1, 2, 3]
    assert len(split_manifest["train_episode_ids"]) == 2
    assert len(split_manifest["val_episode_ids"]) == 1
    assert len(split_manifest["test_episode_ids"]) == 1

    for split_name in ("train", "val", "test"):
        data = np.load(output_dir / f"{split_name}.npz")
        episode_ids = split_manifest[f"{split_name}_episode_ids"]
        expected_transitions = sum(rows[episode_id]["num_transitions"] for episode_id in episode_ids)
        for field in ARRAY_FIELDS:
            assert field in data.files
        assert data["states"].shape == (expected_transitions, 6)
        assert data["next_states"].shape == (expected_transitions, 6)
        assert data["cmd_controls"].shape == (expected_transitions, 3)
        assert data["real_controls"].shape == (expected_transitions, 3)
        assert data["exec_residuals"].shape == (expected_transitions, 3)
        assert data["oracle_residuals"].shape == (expected_transitions, 3)
        assert data["terrain_features"].shape == (expected_transitions, 4)
        assert data["terrain_risk"].shape == (expected_transitions,)
        assert set(data["episode_ids"].tolist()).issubset(set(episode_ids))

    saved_summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert saved_summary == summary
    assert summary["total_input_episodes"] == 4
    assert summary["usable_episodes"] == 4
    assert summary["skipped_episodes"] == 0
    assert summary["total_transitions"] == 14
    assert summary["splits"]["train"]["episodes"] == 2
    assert summary["split_seed"] == 7


def test_build_oracle_dataset_skips_failed_manifest_rows(tmp_path):
    input_dir = tmp_path / "oracle_debug"
    good_path = input_dir / "episodes" / "episode_000001.npz"
    failed_path = input_dir / "episodes" / "episode_000000.npz"
    write_episode(good_path, episode_id=1, transitions=3)
    write_manifest(
        input_dir,
        [
            {
                "episode_id": 0,
                "seed": 200,
                "path": str(failed_path),
                "success": False,
                "failed": True,
                "num_transitions": 0,
                "error": "collector failed",
            },
            {
                "episode_id": 1,
                "seed": 201,
                "path": str(good_path),
                "success": True,
                "failed": False,
                "num_transitions": 3,
            },
        ],
    )

    summary = build_oracle_dataset(
        input_dir=input_dir,
        output_dir=tmp_path / "oracle_dataset",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        split_seed=1,
    )

    assert summary["total_input_episodes"] == 2
    assert summary["usable_episodes"] == 1
    assert summary["skipped_episodes"] == 1
    train = np.load(tmp_path / "oracle_dataset" / "train.npz")
    assert train["states"].shape == (3, 6)
    assert set(train["episode_ids"].tolist()) == {1}
    val = np.load(tmp_path / "oracle_dataset" / "val.npz")
    test = np.load(tmp_path / "oracle_dataset" / "test.npz")
    assert val["states"].shape == (0, 6)
    assert test["states"].shape == (0, 6)
