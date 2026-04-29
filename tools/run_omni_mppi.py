#!/usr/bin/env python3
"""Run the B2 omni NumPy MPPI simulation from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/b2_omni_nominal.yaml")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    config = load_config(args.config)
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: MppiOmniNumpy.from_config(config, seed=args.seed),
    )
    summary = runner.run()
    print(f"results_path={summary.results_path}")
    print(f"steps={summary.steps}")
    print(f"reached_goal={summary.reached_goal}")
    print(f"failed={summary.failed}")
    print(f"run_time={summary.run_time}")
    print(f"animation={summary.results_path / 'animation.gif'}")


if __name__ == "__main__":
    main()
