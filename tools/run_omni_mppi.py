#!/usr/bin/env python3
"""Run the B2 omni MPPI simulation from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/b2_omni_nominal.yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    parser.add_argument("--fdm-enabled", action="store_true")
    parser.add_argument("--fdm-model-dir", default=None)
    parser.add_argument("--fdm-checkpoint", default=None)
    parser.add_argument("--fdm-normalization", default=None)
    parser.add_argument("--fdm-device", default=None)
    parser.add_argument("--fdm-residual-gain", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=args.seed),
    )
    summary = runner.run()
    print(f"results_path={summary.results_path}")
    print(f"steps={summary.steps}")
    print(f"reached_goal={summary.reached_goal}")
    print(f"failed={summary.failed}")
    print(f"run_time={summary.run_time}")
    animation_path = summary.results_path / "animation.gif"
    if animation_path.exists():
        print(f"animation={animation_path}")


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.backend is not None:
        config.setdefault("mppi", {})["backend"] = str(args.backend).lower()
    if args.fdm_enabled:
        config.setdefault("fdm", {})["enabled"] = True
    if args.fdm_model_dir is not None:
        config.setdefault("fdm", {})["model_dir"] = args.fdm_model_dir
    if args.fdm_checkpoint is not None:
        config.setdefault("fdm", {})["checkpoint"] = args.fdm_checkpoint
    if args.fdm_normalization is not None:
        config.setdefault("fdm", {})["normalization"] = args.fdm_normalization
    if args.fdm_device is not None:
        config.setdefault("fdm", {})["device"] = args.fdm_device
    fdm_residual_gain = getattr(args, "fdm_residual_gain", None)
    if fdm_residual_gain is not None:
        config.setdefault("fdm", {})["residual_gain"] = float(fdm_residual_gain)
    return config


if __name__ == "__main__":
    main()
