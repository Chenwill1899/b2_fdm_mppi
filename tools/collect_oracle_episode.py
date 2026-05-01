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


def collect_oracle_episode(
    *,
    config_path: str | Path,
    episode_id: int,
    seed: int,
    output_path: str | Path,
    backend: str | None = None,
) -> dict:
    config = load_config(config_path)
    config.setdefault("scenario", {})["random_seed"] = int(seed)
    config.setdefault("oracle_residual", {})["seed"] = int(seed)
    if backend is not None:
        config["mppi"]["backend"] = str(backend).lower()

    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=seed),
    )
    summary = runner.run()
    return build_episode_npz(summary.results_path, episode_id, Path(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    args = parser.parse_args()

    metadata = collect_oracle_episode(
        config_path=args.config,
        episode_id=args.episode_id,
        seed=args.seed,
        output_path=args.output,
        backend=args.backend,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
