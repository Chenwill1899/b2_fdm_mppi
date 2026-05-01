import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_generator_module():
    module_path = Path("tools/generate_oracle_episodes.py")
    spec = importlib.util.spec_from_file_location("generate_oracle_episodes_parallel", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mock_collector(*, config_path, episode_id, seed, output_path, backend=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        states=np.full((episode_id + 1, 2), seed, dtype=np.float32),
        actions=np.full((episode_id + 1, 2), episode_id, dtype=np.float32),
    )
    return {
        "episode_id": episode_id,
        "num_transitions": episode_id + 1,
        "success": True,
        "failed": False,
        "start_goal_distance": 10.0 + episode_id,
        "final_distance": 0.1 * episode_id,
        "min_obstacle_clearance": 2.0,
        "output_path": str(output_path),
    }


def failing_mock_collector(*, config_path, episode_id, seed, output_path, backend=None):
    if episode_id == 1:
        raise RuntimeError(f"planned failure for episode {episode_id}")
    return mock_collector(
        config_path=config_path,
        episode_id=episode_id,
        seed=seed,
        output_path=output_path,
        backend=backend,
    )


def read_manifest(output_dir):
    return [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_num_workers_one_and_two_keep_manifest_sorted_and_seed_mapping(tmp_path):
    module = load_generator_module()

    for num_workers in (1, 2):
        output_dir = tmp_path / f"oracle_workers_{num_workers}"
        summary = module.generate_oracle_episodes(
            config_path=Path("config/b2_omni_oracle_random100_dataset.yaml"),
            episodes=4,
            base_seed=123,
            output_dir=output_dir,
            backend="numpy",
            collector=mock_collector,
            num_workers=num_workers,
        )

        rows = read_manifest(output_dir)
        assert [row["episode_id"] for row in rows] == [0, 1, 2, 3]
        assert [row["seed"] for row in rows] == [123, 124, 125, 126]
        assert [row["num_transitions"] for row in rows] == [1, 2, 3, 4]
        assert all(row["success"] is True and row["failed"] is False for row in rows)
        for episode_id in range(4):
            assert (output_dir / "episodes" / f"episode_{episode_id:06d}.npz").exists()
        assert summary["total_episodes"] == 4
        assert summary["success_episodes"] == 4
        assert summary["failed_episodes"] == 0
        assert summary["total_transitions"] == 10
        assert summary["success_rate"] == 1.0


def test_parallel_failure_records_error_and_continues(tmp_path):
    module = load_generator_module()
    output_dir = tmp_path / "oracle_parallel_failure"

    summary = module.generate_oracle_episodes(
        config_path=Path("config/b2_omni_oracle_random100_dataset.yaml"),
        episodes=3,
        base_seed=500,
        output_dir=output_dir,
        backend="numpy",
        collector=failing_mock_collector,
        num_workers=2,
    )

    rows = read_manifest(output_dir)
    assert [row["episode_id"] for row in rows] == [0, 1, 2]
    assert [row["seed"] for row in rows] == [500, 501, 502]
    assert rows[0]["success"] is True
    assert rows[1]["success"] is False
    assert rows[1]["failed"] is True
    assert rows[1]["num_transitions"] == 0
    assert "planned failure for episode 1" in rows[1]["error"]
    assert rows[2]["success"] is True
    assert (output_dir / "episodes" / "episode_000000.npz").exists()
    assert not (output_dir / "episodes" / "episode_000001.npz").exists()
    assert (output_dir / "episodes" / "episode_000002.npz").exists()
    assert summary["total_episodes"] == 3
    assert summary["success_episodes"] == 2
    assert summary["failed_episodes"] == 1
    assert summary["total_transitions"] == 4
    assert summary["success_rate"] == 2 / 3
