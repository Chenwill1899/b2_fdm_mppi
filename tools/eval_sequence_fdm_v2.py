#!/usr/bin/env python3
"""CLI for evaluating Sequence FDM V2: run N episodes and collect metrics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.data.collect_oracle_episode import collect_oracle_episode
from b2_fdm_mppi.data.sequence_fdm_collector import _convert_tuples_to_lists, _sample_start_goal
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator


def _eval_one(
    base_config: dict,
    terrain_seed: int,
    episode_id: int,
    output_dir: Path,
    mode: str,
    model_dir: str | None,
) -> dict:
    """Run a single evaluation episode."""
    rng = np.random.default_rng(terrain_seed)
    terrain_gen = RandomTerrainGenerator(
        map_bounds=(-15.0, 15.0, -15.0, 15.0),
        num_patches_range=(3, 6),
    )
    terrain = terrain_gen.generate(seed=terrain_seed)

    # Sample obstacles
    rng_obs = np.random.default_rng(terrain_seed + 50000)
    num_obs = int(rng_obs.integers(3, 7))
    obs_list = []
    for _ in range(num_obs):
        ox = float(rng_obs.uniform(-13.0, 13.0))
        oy = float(rng_obs.uniform(-13.0, 13.0))
        radius = float(rng_obs.uniform(0.8, 2.0))
        obs_list.append([ox, oy, radius, 0.0, 0.0, 0.0, 0.0])

    # Sample start and goal
    start_xy, goal_xy = _sample_start_goal(
        rng, (-15.0, 15.0, -15.0, 15.0),
        min_distance=10.0,
        terrain=terrain,
        max_start_goal_risk=0.5,
    )

    # Build config
    config = dict(base_config)
    from b2_fdm_mppi.data.sequence_fdm_collector import _terrain_to_config
    config["terrain"] = _terrain_to_config(terrain)
    # Oracle uses CUDA backend; learned V2 controller only has torch impl
    config.setdefault("mppi", {})["backend"] = "torch" if mode == "learned" else "cuda"
    config["simulation"]["max_steps"] = 500
    config.setdefault("results", {})["enable_plots"] = False
    config.setdefault("results", {})["enable_animation"] = False
    mppi_cfg = config.setdefault("mppi", {})
    mppi_cfg["terrain_risk_weight"] = 1.0
    mppi_cfg["terrain_risk_threshold"] = 0.6
    mppi_cfg["terrain_risk_mode"] = "excess"
    config["obstacles"] = {
        "num_max": num_obs,
        "static_enabled": True,
        "virtual": obs_list,
    }
    if mode == "learned":
        if not model_dir:
            raise ValueError("--model-dir required for learned mode")
        config["sequence_fdm_v2"] = {
            "enabled": True,
            "model_dir": str(Path(model_dir).resolve()),
            "device": "cuda",
            "fdm_risk_weight": 10.0,
        }

    initial_state = list(config["simulation"]["initial_state"])
    goal_state = list(config["simulation"]["goal"])
    initial_state[0] = float(start_xy[0])
    initial_state[1] = float(start_xy[1])
    goal_state[0] = float(goal_xy[0])
    goal_state[1] = float(goal_xy[1])
    config["simulation"]["initial_state"] = initial_state
    config["simulation"]["goal"] = goal_state

    output_path = output_dir / f"eval_episode_{episode_id:03d}.npz"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(_convert_tuples_to_lists(config), tf)
        temp_path = Path(tf.name)

    try:
        metadata = collect_oracle_episode(
            config_path=temp_path,
            episode_id=episode_id,
            seed=terrain_seed,
            output_path=output_path,
        )
        return {
            "episode_id": episode_id,
            "terrain_seed": terrain_seed,
            "mode": mode,
            "success": metadata.get("success", False),
            "failed": metadata.get("failed", False),
            "num_transitions": metadata.get("num_transitions", 0),
            "final_distance": float(metadata.get("final_distance", float("nan"))),
            "start_goal_distance": float(metadata.get("start_goal_distance", float("nan"))),
            "min_obstacle_clearance": float(
                metadata.get("min_obstacle_clearance", float("nan"))
                if metadata.get("min_obstacle_clearance") is not None else float("nan")
            ),
        }
    except Exception as e:
        return {
            "episode_id": episode_id,
            "terrain_seed": terrain_seed,
            "mode": mode,
            "error": str(e),
            "success": False,
            "failed": True,
            "num_transitions": 0,
            "final_distance": float("nan"),
        }
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate oracle or learned controller on N fixed scenarios")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Base config YAML")
    parser.add_argument("--mode", choices=["oracle", "learned"], default="oracle", help="Controller mode")
    parser.add_argument("--model-dir", default=None, help="Model dir for learned mode (best_model.pt + normalization.npz)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--base-seed", type=int, default=999000, help="Base seed for test scenarios")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="Device for learned model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    results = []

    for i in range(args.num_episodes):
        terrain_seed = args.base_seed + i
        print(f"[{i + 1}/{args.num_episodes}] seed={terrain_seed} mode={args.mode}")
        result = _eval_one(
            base_config=base_config,
            terrain_seed=terrain_seed,
            episode_id=i,
            output_dir=output_dir,
            mode=args.mode,
            model_dir=args.model_dir,
        )
        print(f"  success={result['success']} failed={result['failed']} "
              f"steps={result['num_transitions']} final_dist={result['final_distance']:.2f}")
        results.append(result)

    # Aggregate
    valid = [r for r in results if "error" not in r]
    successes = sum(1 for r in valid if r["success"])
    failures = sum(1 for r in valid if r["failed"])
    crashed = sum(1 for r in results if "error" in r)
    avg_steps = float(np.mean([r["num_transitions"] for r in valid])) if valid else 0.0
    avg_final_dist = float(np.mean([r["final_distance"] for r in valid if not np.isnan(r["final_distance"])])) if valid else float("nan")
    summary = {
        "mode": args.mode,
        "num_episodes": args.num_episodes,
        "successes": successes,
        "failures": failures,
        "crashed": crashed,
        "success_rate": successes / len(valid) if valid else 0.0,
        "avg_steps": round(avg_steps, 1),
        "avg_final_distance": round(avg_final_dist, 3),
        "results": results,
    }

    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.mode.upper()} EVAL SUMMARY ===")
    print(f"  Success: {successes}/{len(valid)} ({100 * summary['success_rate']:.1f}%)")
    print(f"  Failed:  {failures}/{len(valid)}")
    print(f"  Crashed: {crashed}/{len(valid)}")
    print(f"  Avg steps: {summary['avg_steps']}")
    print(f"  Avg final distance: {summary['avg_final_distance']:.3f}")
    print(f"  Saved to: {output_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
