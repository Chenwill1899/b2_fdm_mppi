#!/usr/bin/env python3
"""Collect one oracle simulation episode and save it as an FDM npz sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.data.oracle_episode import build_episode_npz
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("scenario", {})["random_seed"] = int(args.seed)
    config.setdefault("oracle_residual", {})["seed"] = int(args.seed)

    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=args.seed),
    )
    summary = runner.run()
    metadata = build_episode_npz(summary.results_path, args.episode_id, Path(args.output))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
