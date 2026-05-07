#!/usr/bin/env python3
"""Run oracle vs learned comparison on a single fixed scene with plots enabled."""

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


def _run_one(
    base_config: dict,
    terrain_seed: int,
    episode_id: int,
    output_dir: Path,
    mode: str,
    model_dir: str | None = None,
    enable_plots: bool = True,
) -> dict:
    """Run a single episode with plots enabled."""
    rng = np.random.default_rng(terrain_seed)
    terrain_gen = RandomTerrainGenerator(
        map_bounds=(-15.0, 15.0, -15.0, 15.0),
        num_patches_range=(3, 6),
    )
    terrain = terrain_gen.generate(seed=terrain_seed)

    rng_obs = np.random.default_rng(terrain_seed + 50000)
    num_obs = int(rng_obs.integers(3, 7))
    obs_list = []
    for _ in range(num_obs):
        ox = float(rng_obs.uniform(-13.0, 13.0))
        oy = float(rng_obs.uniform(-13.0, 13.0))
        radius = float(rng_obs.uniform(0.8, 2.0))
        obs_list.append([ox, oy, radius, 0.0, 0.0, 0.0, 0.0])

    start_xy, goal_xy = _sample_start_goal(
        rng, (-15.0, 15.0, -15.0, 15.0),
        min_distance=10.0,
        terrain=terrain,
        max_start_goal_risk=0.5,
    )

    config = dict(base_config)
    from b2_fdm_mppi.data.sequence_fdm_collector import _terrain_to_config
    config["terrain"] = _terrain_to_config(terrain)
    config.setdefault("mppi", {})["backend"] = "torch" if mode == "learned" else "cuda"
    config["simulation"]["max_steps"] = 500
    config.setdefault("results", {})["enable_plots"] = enable_plots
    config.setdefault("results", {})["enable_animation"] = False
    config.setdefault("results", {})["root"] = str(output_dir / mode)
    config.setdefault("results", {})["run_name"] = f"seed_{terrain_seed}_{mode}"
    config.setdefault("results", {})["overwrite"] = True
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
            raise ValueError("model_dir required for learned mode")
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

    output_path = output_dir / f"{mode}_episode_{terrain_seed}.npz"
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
        print(f"  [{mode}] success={metadata.get('success')} steps={metadata.get('num_transitions')} fd={metadata.get('final_distance', float('nan')):.2f}")
        return metadata
    except Exception as e:
        print(f"  [{mode}] FAILED: {e}")
        return {"error": str(e), "success": False}
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=999004)
    parser.add_argument("--model-dir", default="checkpoints/large_scale/round_9", help="Best model dir")
    parser.add_argument("--output-dir", default="results/single_scene_compare", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)

    print(f"Running oracle vs learned comparison for seed={args.seed}")
    print(f"Model: {args.model_dir}")

    oracle_meta = _run_one(base_config, args.seed, 0, output_dir, "oracle")
    learned_meta = _run_one(base_config, args.seed, 0, output_dir, "learned", model_dir=args.model_dir)

    # Save summary
    summary = {
        "seed": args.seed,
        "model_dir": args.model_dir,
        "oracle": {k: v for k, v in oracle_meta.items() if isinstance(v, (bool, int, float, str))},
        "learned": {k: v for k, v in learned_meta.items() if isinstance(v, (bool, int, float, str))},
    }
    with open(output_dir / f"summary_seed_{args.seed}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  Oracle:  success={oracle_meta.get('success')} steps={oracle_meta.get('num_transitions')} final_dist={oracle_meta.get('final_distance', float('nan')):.2f}")
    print(f"  Learned: success={learned_meta.get('success')} steps={learned_meta.get('num_transitions')} final_dist={learned_meta.get('final_distance', float('nan')):.2f}")


if __name__ == "__main__":
    main()
