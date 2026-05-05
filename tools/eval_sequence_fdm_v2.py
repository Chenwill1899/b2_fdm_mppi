#!/usr/bin/env python3
"""CLI for evaluating Sequence FDM V2: open-loop and closed-loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_torch import MppiOmniTorch
from b2_fdm_mppi.controllers.mppi_omni_sequence_fdm_v2_torch import MppiOmniSequenceFdmV2Torch
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator


def _load_controller(model_dir: str | None, config: dict, device: str):
    """Load either nominal or sequence FDM controller from config."""
    if model_dir is None:
        return MppiOmniTorch.from_config(config, device=device)
    dynamics = SequenceFdmDynamics.from_artifacts(Path(model_dir), device=device)
    return MppiOmniSequenceFdmV2Torch.from_config(
        config, device=device, sequence_dynamics=dynamics, fdm_risk_weight=10.0
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sequence FDM V2")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--model-dir", help="Trained model directory (for FDM controller)")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config = load_config(args.config)

    results = []
    for i in range(args.num_episodes):
        terrain_seed = args.base_seed + i
        generator = RandomTerrainGenerator(map_bounds=(-15, 15, -15, 15))
        terrain = generator.generate(seed=terrain_seed)

        # Build terrain config manually since TerrainField lacks to_config()
        config = dict(base_config)
        config["terrain"] = {
            "enabled": True,
            "slope_scale": terrain.slope_scale,
            "slope_wave": terrain.slope_wave,
            "roughness_scale": terrain.roughness_scale,
            "roughness_wave": terrain.roughness_wave,
            "friction_base": terrain.friction_base,
            "friction_slope_scale": terrain.friction_slope_scale,
            "friction_roughness_scale": terrain.friction_roughness_scale,
            "risk_weights": list(terrain.risk_weights),
            "goal_relief": dict(terrain.goal_relief),
            "noise_enabled": terrain.noise_enabled,
            "noise_seed": terrain.noise_seed,
            "noise_grid_size": list(terrain.noise_grid_size),
            "noise_scale": terrain.noise_scale,
            "noise_smooth_passes": terrain.noise_smooth_passes,
            "noise_roughness_weight": terrain.noise_roughness_weight,
            "noise_friction_weight": terrain.noise_friction_weight,
            "noise_slope_weight": terrain.noise_slope_weight,
            "noise_x_range": list(terrain.noise_x_range),
            "noise_y_range": list(terrain.noise_y_range),
            "patches": [
                {
                    "type": p["type"],
                    "center": list(p["center"]),
                    "angle": p["angle"],
                    "size": list(p["size"]),
                    "edge_width": p["edge_width"],
                    "slope_f_delta": p.get("slope_f_delta", 0.0),
                    "slope_l_delta": p.get("slope_l_delta", 0.0),
                    "roughness_delta": p.get("roughness_delta", 0.0),
                    "friction_delta": p.get("friction_delta", 0.0),
                }
                for p in (terrain.patches or [])
            ],
        }

        # Note: start/goal sampling and closed-loop runner integration
        # should be added here for full evaluation.
        result = {
            "episode": i,
            "terrain_seed": terrain_seed,
            "fdm_model": args.model_dir,
        }
        results.append(result)

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete: {len(results)} episodes")


if __name__ == "__main__":
    main()
