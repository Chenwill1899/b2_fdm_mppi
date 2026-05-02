#!/usr/bin/env python3
"""Run seed-123 oracle Stage 5 parameter variants with GIF output enabled."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller
from tools.benchmark_learned_fdm_mppi import prepare_run_config


CASES = [
    {
        "case": "nominal",
        "label": "Nominal CUDA",
        "controller": "nominal",
        "residual_gain": None,
        "overrides": {},
    },
    {
        "case": "default_g10",
        "label": "Default learned g=1.0",
        "controller": "learned",
        "residual_gain": 1.0,
        "overrides": {},
    },
    {
        "case": "efficiency_g05_goal35_smooth10",
        "label": "Efficiency 0.5/3.5/1.0",
        "controller": "learned",
        "residual_gain": 0.5,
        "overrides": {"goal_xy_weight": 3.5, "smooth_weight": 1.0},
    },
    {
        "case": "balanced_g05_goal30_smooth075",
        "label": "Balanced 0.5/3.0/0.75",
        "controller": "learned",
        "residual_gain": 0.5,
        "overrides": {"goal_xy_weight": 3.0, "smooth_weight": 0.75},
    },
]

SUMMARY_FIELDS = [
    "case",
    "label",
    "controller",
    "seed",
    "success",
    "failed",
    "final_distance",
    "steps",
    "arrival_time",
    "path_length",
    "min_obstacle_clearance",
    "mean_terrain_risk",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
    "max_mppi_time_ms",
    "results_path",
    "animation_gif",
]


def _enable_visual_outputs(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("results", {})["enable_plots"] = True
    config.setdefault("results", {})["enable_animation"] = True
    return config


def _load_summary(results_path: Path) -> dict[str, Any]:
    return json.loads((results_path / "summary.json").read_text(encoding="utf-8"))


def run_cases(
    *,
    config_path: Path,
    output_dir: Path,
    seed: int,
    backend: str,
    fdm_model_dir: Path,
    fdm_checkpoint: str,
    fdm_normalization: str,
    fdm_device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(config_path)
    records = []

    for case in CASES:
        scenario_name = f"stage5_seed123_{case['case']}"
        residual_gain = case["residual_gain"] if case["residual_gain"] is not None else 1.0
        run_config = prepare_run_config(
            base_config,
            output_dir=output_dir,
            scenario_name=scenario_name,
            controller=case["controller"],
            episode_id=0,
            seed=seed,
            backend=backend,
            fdm_model_dir=fdm_model_dir,
            fdm_checkpoint=fdm_checkpoint,
            fdm_normalization=fdm_normalization,
            fdm_device=fdm_device,
            fdm_residual_gain=float(residual_gain),
            learned_mppi_overrides=case["overrides"],
        )
        run_config = _enable_visual_outputs(run_config)
        runner = OmniMppiSimulationRunner(
            run_config,
            controller_factory=lambda *, config, runner, seed=seed: create_omni_controller(
                config,
                seed=seed,
            ),
        )
        summary = runner.run()
        results_path = Path(summary.results_path)
        summary_json = _load_summary(results_path)
        gif_path = results_path / "animation.gif"
        copied_gif = gifs_dir / f"{case['case']}_seed{seed}_animation.gif"
        if gif_path.exists():
            shutil.copy2(gif_path, copied_gif)
        record = {
            "case": case["case"],
            "label": case["label"],
            "controller": case["controller"],
            "seed": seed,
            "results_path": str(results_path),
            "animation_gif": str(copied_gif) if copied_gif.exists() else "",
            "overrides": case["overrides"],
            "residual_gain": case["residual_gain"],
        }
        for field in SUMMARY_FIELDS:
            if field not in record:
                record[field] = summary_json.get(field)
        records.append(record)

    summary = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "backend": backend,
        "fdm_model_dir": str(fdm_model_dir),
        "fdm_checkpoint": fdm_checkpoint,
        "fdm_normalization": fdm_normalization,
        "fdm_device": fdm_device,
        "cases": records,
    }
    (output_dir / "seed123_param_gif_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(records)[SUMMARY_FIELDS].to_csv(
        output_dir / "seed123_param_gif_summary.csv", index=False
    )
    write_index(output_dir, records)
    return summary


def write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        gif_rel = Path(record["animation_gif"]).relative_to(output_dir) if record.get("animation_gif") else None
        rows.append(
            f"""
    <section>
      <h2>{record['label']}</h2>
      <p>final_distance={float(record['final_distance']):.4f}, steps={int(record['steps'])}, risk={float(record['mean_terrain_risk']):.4f}, smooth={float(record['control_smoothness']):.6f}</p>
      {f'<img src="{gif_rel.as_posix()}" alt="{record["label"]} animation GIF">' if gif_rel else '<p>GIF missing.</p>'}
    </section>
"""
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 5 seed123 parameter GIFs</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; max-width: 1180px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0 28px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Stage 5 seed123 parameter GIFs</h1>
  <p>Fixed scene: <code>config/b2_omni_oracle.yaml</code>, <code>seed=123</code>. Each trajectory stops when it enters the configured goal tolerance, so it may not end exactly on the goal marker.</p>
  {''.join(rows)}
  <p>CSV: <a href="seed123_param_gif_summary.csv">seed123_param_gif_summary.csv</a></p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/b2_omni_oracle.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage6_result_package/seed123_oracle_param_gifs"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "numpy"])
    parser.add_argument("--fdm-model-dir", type=Path, default=Path("results/fdm_baselines/stage4_mlp_seed123_hardened"))
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_cases(
        config_path=args.config,
        output_dir=args.output,
        seed=args.seed,
        backend=args.backend,
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_device=args.fdm_device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
