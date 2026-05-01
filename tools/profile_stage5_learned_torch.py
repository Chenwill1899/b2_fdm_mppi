#!/usr/bin/env python3
"""Profile Stage 5 learned-FDM Torch MPPI runtime buckets."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller
from tools.benchmark_learned_fdm_mppi import current_git_metadata, shell_join


def run_profile(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    steps: int,
    seed: int,
    fdm_model_dir: str | Path,
    fdm_checkpoint: str | Path = "best_model.pt",
    fdm_normalization: str | Path = "normalization.npz",
    fdm_device: str | None = None,
    fdm_residual_gain: float = 1.0,
    runner_cls=OmniMppiSimulationRunner,
    command: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config = _profile_config(
        load_config(config_path),
        output_dir=output_dir,
        steps=steps,
        fdm_model_dir=fdm_model_dir,
        fdm_checkpoint=fdm_checkpoint,
        fdm_normalization=fdm_normalization,
        fdm_device=fdm_device or "cuda",
        fdm_residual_gain=fdm_residual_gain,
    )
    runner = runner_cls(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(
            config,
            seed=seed,
        ),
    )
    simulation_summary = runner.run()
    controller_profile = _controller_profile(runner.controller)
    run_summary_path = Path(simulation_summary.results_path) / "summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config": str(config_path),
            "output_dir": str(output_dir),
            "steps": int(steps),
            "seed": int(seed),
            "fdm_model_dir": str(fdm_model_dir),
            "fdm_checkpoint": str(fdm_checkpoint),
            "fdm_normalization": str(fdm_normalization),
            "fdm_device": str(fdm_device or "cuda"),
            "fdm_residual_gain": float(fdm_residual_gain),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **current_git_metadata(),
        },
        "run": {
            "results_path": str(simulation_summary.results_path),
            "steps": int(simulation_summary.steps),
            "reached_goal": bool(simulation_summary.reached_goal),
            "failed": bool(simulation_summary.failed),
            "run_time": float(simulation_summary.run_time),
            "summary_json": str(run_summary_path),
            "summary": run_summary,
        },
        "controller_profile": controller_profile,
    }
    (output_dir / "stage5_runtime_profile_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _profile_config(
    base_config: dict,
    *,
    output_dir: Path,
    steps: int,
    fdm_model_dir: str | Path,
    fdm_checkpoint: str | Path,
    fdm_normalization: str | Path,
    fdm_device: str,
    fdm_residual_gain: float,
) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault("simulation", {})["world_mode"] = "oracle"
    config.setdefault("simulation", {})["max_steps"] = int(steps)
    config.setdefault("mppi", {})["backend"] = "cuda"
    config["fdm"] = {
        **config.get("fdm", {}),
        "enabled": True,
        "model_dir": str(fdm_model_dir),
        "checkpoint": str(fdm_checkpoint),
        "normalization": str(fdm_normalization),
        "device": str(fdm_device),
        "residual_gain": float(fdm_residual_gain),
        "profile_enabled": True,
    }
    config["results"] = {
        **config.get("results", {}),
        "root": str(output_dir / "runs"),
        "run_name": "learned_torch_profile",
        "timestamp_suffix": False,
        "overwrite": True,
        "enable_plots": False,
        "enable_animation": False,
    }
    return config


def _controller_profile(controller) -> dict:
    profile_summary = getattr(controller, "profile_summary", None)
    if callable(profile_summary):
        return profile_summary()
    return {
        "enabled": False,
        "total_calls": 0,
        "totals_ms": {},
        "means_ms": {},
        "counts": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/b2_omni_oracle.yaml")
    parser.add_argument("--output", default="results/stage5_profile/standard_seed123_cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-device", default="cuda")
    parser.add_argument("--fdm-residual-gain", type=float, default=1.0)
    args = parser.parse_args()

    summary = run_profile(
        config_path=args.config,
        output_dir=args.output,
        steps=args.steps,
        seed=args.seed,
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_device=args.fdm_device,
        fdm_residual_gain=args.fdm_residual_gain,
        command=shell_join([sys.executable, *sys.argv]),
        argv=[sys.executable, *sys.argv],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
