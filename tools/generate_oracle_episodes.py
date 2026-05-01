#!/usr/bin/env python3
"""Serially collect oracle episodes and write manifest/summary files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_oracle_episode import collect_oracle_episode


Collector = Callable[..., dict]


def generate_oracle_episodes(
    *,
    config_path: str | Path,
    episodes: int,
    base_seed: int,
    output_dir: str | Path,
    backend: str | None = None,
    collector: Collector = collect_oracle_episode,
) -> dict:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    episodes_dir = output_dir / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    success_episodes = 0
    failed_episodes = 0
    total_transitions = 0
    with manifest_path.open("w", encoding="utf-8") as stream:
        for episode_id in range(int(episodes)):
            seed = int(base_seed) + episode_id
            episode_path = episodes_dir / f"episode_{episode_id:06d}.npz"
            try:
                metadata = collector(
                    config_path=config_path,
                    episode_id=episode_id,
                    seed=seed,
                    output_path=episode_path,
                    backend=backend,
                )
                row = {
                    "episode_id": episode_id,
                    "seed": seed,
                    "path": str(episode_path),
                    "success": bool(metadata.get("success", False)),
                    "failed": bool(metadata.get("failed", False)),
                    "num_transitions": int(metadata.get("num_transitions", 0)),
                    "start_goal_distance": _optional_float(metadata.get("start_goal_distance")),
                    "final_distance": _optional_float(metadata.get("final_distance")),
                    "min_obstacle_clearance": _optional_float(metadata.get("min_obstacle_clearance")),
                }
            except Exception as exc:  # pragma: no cover - exact failures are exercised through tests.
                row = {
                    "episode_id": episode_id,
                    "seed": seed,
                    "path": str(episode_path),
                    "success": False,
                    "failed": True,
                    "num_transitions": 0,
                    "start_goal_distance": None,
                    "final_distance": None,
                    "min_obstacle_clearance": None,
                    "error": str(exc),
                }

            if bool(row["success"]) and not bool(row["failed"]):
                success_episodes += 1
            else:
                failed_episodes += 1
            total_transitions += int(row["num_transitions"])
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    total_episodes = int(episodes)
    summary = {
        "total_episodes": total_episodes,
        "success_episodes": success_episodes,
        "failed_episodes": failed_episodes,
        "total_transitions": total_transitions,
        "success_rate": success_episodes / total_episodes if total_episodes else 0.0,
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "base_seed": int(base_seed),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    args = parser.parse_args()

    summary = generate_oracle_episodes(
        config_path=args.config,
        episodes=args.episodes,
        base_seed=args.base_seed,
        output_dir=args.output,
        backend=args.backend,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
