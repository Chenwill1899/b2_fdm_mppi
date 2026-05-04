#!/usr/bin/env python3
"""3-round iterative training: collect data → train → repeat."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add package to path
import torch


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{desc}")
    print(f"{'=' * 60}")
    env = {**os.environ, "PYTHONPATH": "/home/test/docker/prjcwl/b2_fdm_mppi"}
    t0 = time.time()
    result = subprocess.run(cmd, cwd="/home/test/docker/prjcwl/b2_fdm_mppi", env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED after {elapsed:.0f}s")
        sys.exit(1)
    print(f"Done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="3-round iterative training")
    parser.add_argument("--base-config", default="configs/smoke.yaml")
    parser.add_argument("--episodes", type=int, nargs="+", default=[100, 100, 200],
                        help="Episodes per round (3 values for 3 rounds)")
    parser.add_argument("--num-trajectories", type=int, default=3)
    parser.add_argument("--data-root", default="data/iterative")
    parser.add_argument("--checkpoint-root", default="checkpoints/iterative")
    parser.add_argument("--map-bounds", type=float, nargs=4, default=[-15.0, 15.0, -15.0, 15.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed for round 1 (each round offsets by 100000)")
    args = parser.parse_args()

    base_config = args.base_config
    num_rounds = 3
    device = args.device

    # Horizon/epochs/lr per phase
    horizons = [5, 10, 20]
    epochs = [50, 50, 100]
    lrs = [1e-3, 5e-4, 1e-4]
    curriculum = list(zip(horizons, epochs, lrs))

    summary = []

    for round_idx in range(1, num_rounds + 1):
        print(f"\n{'#' * 60}")
        print(f"# ROUND {round_idx}/{num_rounds}")
        print(f"{'#' * 60}")

        data_dir = Path(args.data_root) / f"round_{round_idx}"
        ckpt_dir = Path(args.checkpoint_root) / f"round_{round_idx}"
        data_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        episodes = args.episodes[round_idx - 1]
        seed = args.base_seed + (round_idx - 1) * 100000

        # --- Collect data ---
        t0 = time.time()
        run([
            sys.executable, "tools/collect_sequence_fdm_v2_data.py",
            "--base-config", base_config,
            "--episodes", str(episodes),
            "--num-trajectories", str(args.num_trajectories),
            "--output-dir", str(data_dir),
            "--base-seed", str(seed),
            "--workers", "16",
            "--map-bounds",
            str(args.map_bounds[0]), str(args.map_bounds[1]),
            str(args.map_bounds[2]), str(args.map_bounds[3]),
        ], f"[Round {round_idx}] Collecting {episodes} episodes × {args.num_trajectories} trajectories")
        collect_time = time.time() - t0

        # --- Train ---
        train_cmd = [
            sys.executable, "tools/train_sequence_fdm_v2.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(ckpt_dir),
            "--device", device,
            "--batch-size", str(args.batch_size),
        ]
        if round_idx > 1:
            prev_ckpt = Path(args.checkpoint_root) / f"round_{round_idx - 1}"
            train_cmd += ["--resume", str(prev_ckpt)]
            print(f"  Resuming from: {prev_ckpt}")

        t0 = time.time()
        run(train_cmd, f"[Round {round_idx}] Training")
        train_time = time.time() - t0

        # Load metrics
        metrics_path = ckpt_dir / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            best_loss = metrics.get("best_val_loss", "N/A")
        else:
            best_loss = "N/A"

        summary.append({
            "round": round_idx,
            "episodes": episodes,
            "num_trajectories": args.num_trajectories,
            "collect_time_s": round(collect_time, 1),
            "train_time_s": round(train_time, 1),
            "best_val_loss": best_loss,
            "data_dir": str(data_dir),
            "checkpoint_dir": str(ckpt_dir),
        })

        print(f"\n  Round {round_idx} summary:")
        print(f"    Episodes: {episodes} × {args.num_trajectories} = {episodes * args.num_trajectories} trajectories")
        print(f"    Collect: {collect_time:.0f}s, Train: {train_time:.0f}s")
        print(f"    Best val loss: {best_loss}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("ITERATIVE TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for s in summary:
        print(f"  Round {s['round']}: {s['episodes']} eps × {s['num_trajectories']} traj, "
              f"collect={s['collect_time_s']}s, train={s['train_time_s']}s, "
              f"best_loss={s['best_val_loss']}")
    total_time = sum(s["collect_time_s"] + s["train_time_s"] for s in summary)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    summary_path = Path(args.checkpoint_root) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
