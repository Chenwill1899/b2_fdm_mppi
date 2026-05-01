import importlib.util
import json
from pathlib import Path

import numpy as np


def load_generator_module():
    module_path = Path("tools/generate_oracle_episodes.py")
    spec = importlib.util.spec_from_file_location("generate_oracle_episodes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_oracle_episodes_writes_manifest_summary_and_npz_files(tmp_path):
    module = load_generator_module()
    calls = []

    def fake_collector(*, config_path, episode_id, seed, output_path, backend=None):
        calls.append(
            {
                "config_path": str(config_path),
                "episode_id": episode_id,
                "seed": seed,
                "output_path": str(output_path),
                "backend": backend,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, states=np.zeros((episode_id + 1, 6), dtype=np.float32))
        return {
            "episode_id": episode_id,
            "num_transitions": episode_id + 1,
            "success": True,
            "failed": False,
            "start_goal_distance": 5.0 + episode_id,
            "final_distance": 0.2,
            "min_obstacle_clearance": 1.5,
            "output_path": str(output_path),
        }

    output_dir = tmp_path / "oracle_debug"
    summary = module.generate_oracle_episodes(
        config_path=Path("config/b2_omni_oracle_random100_dataset.yaml"),
        episodes=2,
        base_seed=123,
        output_dir=output_dir,
        backend="numpy",
        collector=fake_collector,
    )

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    first_npz = output_dir / "episodes" / "episode_000000.npz"
    second_npz = output_dir / "episodes" / "episode_000001.npz"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert first_npz.exists()
    assert second_npz.exists()
    assert [call["episode_id"] for call in calls] == [0, 1]
    assert [call["seed"] for call in calls] == [123, 124]
    assert [call["backend"] for call in calls] == ["numpy", "numpy"]

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["episode_id"] == 0
    assert rows[0]["seed"] == 123
    assert rows[0]["path"] == "episodes/episode_000000.npz"
    assert rows[0]["success"] is True
    assert rows[0]["failed"] is False
    assert rows[0]["num_transitions"] == 1
    assert rows[1]["episode_id"] == 1
    assert rows[1]["seed"] == 124
    assert rows[1]["num_transitions"] == 2

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary == summary
    assert summary["total_episodes"] == 2
    assert summary["success_episodes"] == 2
    assert summary["failed_episodes"] == 0
    assert summary["total_transitions"] == 3
    assert summary["success_rate"] == 1.0
    assert summary["output_dir"] == str(output_dir)
    assert summary["config_path"] == "config/b2_omni_oracle_random100_dataset.yaml"
    assert summary["base_seed"] == 123


def test_generate_oracle_episodes_records_failure_and_continues(tmp_path):
    module = load_generator_module()

    def fake_collector(*, config_path, episode_id, seed, output_path, backend=None):
        if episode_id == 0:
            raise RuntimeError("episode failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, states=np.zeros((2, 6), dtype=np.float32))
        return {
            "episode_id": episode_id,
            "num_transitions": 2,
            "success": True,
            "failed": False,
            "start_goal_distance": 6.0,
            "final_distance": 0.3,
            "min_obstacle_clearance": 1.0,
            "output_path": str(output_path),
        }

    output_dir = tmp_path / "oracle_debug"
    summary = module.generate_oracle_episodes(
        config_path=Path("config/b2_omni_oracle_random100_dataset.yaml"),
        episodes=2,
        base_seed=200,
        output_dir=output_dir,
        collector=fake_collector,
    )

    rows = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["episode_id"] == 0
    assert rows[0]["seed"] == 200
    assert rows[0]["success"] is False
    assert rows[0]["failed"] is True
    assert rows[0]["num_transitions"] == 0
    assert "episode failed" in rows[0]["error"]
    assert rows[1]["episode_id"] == 1
    assert rows[1]["seed"] == 201
    assert rows[1]["success"] is True
    assert (output_dir / "episodes" / "episode_000001.npz").exists()
    assert summary["total_episodes"] == 2
    assert summary["success_episodes"] == 1
    assert summary["failed_episodes"] == 1
    assert summary["total_transitions"] == 2
    assert summary["success_rate"] == 0.5
