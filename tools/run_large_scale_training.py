#!/usr/bin/env python3
"""Large-scale iterative training: 5 rounds, 500 eps/round, 3 traj/episode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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


def _detect_round_state(data_root: Path, checkpoint_root: Path, num_rounds: int) -> dict[int, dict]:
    """Detect what each round has completed."""
    state = {}
    for r in range(1, num_rounds + 1):
        data_dir = data_root / f"round_{r}"
        ckpt_dir = checkpoint_root / f"round_{r}"
        npz_files = list(data_dir.glob("episode_*_traj_00.npz")) if data_dir.exists() else []
        ckpt_exists = (ckpt_dir / "best_model.pt").exists()
        training_done = (ckpt_dir / "training_metrics.json").exists()
        state[r] = {
            "data_count": len(npz_files),
            "has_checkpoint": ckpt_exists,
            "training_done": training_done,
        }
    return state


def main():
    parser = argparse.ArgumentParser(description="Large-scale iterative training (5 rounds)")
    parser.add_argument("--base-config", default="configs/smoke.yaml")
    parser.add_argument("--resume-from-seed", type=int, default=None,
                        help="Episode seed to resume from (detects and resumes partial rounds)")
    parser.add_argument("--resume-round", type=int, default=None,
                        help="Resume from a specific round (skips completed rounds)")
    parser.add_argument("--episodes", type=int, nargs="+", default=[500]*10,
                        help="Episodes per round (default: 10 rounds x 500)")
    parser.add_argument("--num-trajectories", type=int, default=3)
    parser.add_argument("--data-root", default="data/large_scale_v2")
    parser.add_argument("--checkpoint-root", default="checkpoints/large_scale_v2")
    parser.add_argument("--map-bounds", type=float, nargs=4, default=[-15.0, 15.0, -15.0, 15.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size (default: 512 for GPU-resident training)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Collection workers (default: 16 for dual-GPU)")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed for round 1 (each round offsets by 100000)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of test scenes for pre/post eval")
    args = parser.parse_args()

    base_config = args.base_config
    num_rounds = len(args.episodes)
    device = args.device

    start_round = args.resume_round if args.resume_round else 1

    state = _detect_round_state(Path(args.data_root), Path(args.checkpoint_root), num_rounds)
    print("Current round state:")
    for r in range(1, num_rounds + 1):
        s = state[r]
        print(f"  Round {r}: {s['data_count']} episodes collected, "
              f"checkpoint={'✓' if s['has_checkpoint'] else '✗'}, "
              f"training={'✓' if s['training_done'] else '✗'}")
    print(f"  Starting from round {start_round}\n")

    summary = []

    for round_idx in range(start_round, num_rounds + 1):
        print(f"\n{'#' * 60}")
        print(f"# ROUND {round_idx}/{num_rounds}")
        print(f"{'#' * 60}")

        data_dir = Path(args.data_root) / f"round_{round_idx}"
        ckpt_dir = Path(args.checkpoint_root) / f"round_{round_idx}"
        data_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        episodes = args.episodes[round_idx - 1]
        seed = args.base_seed + (round_idx - 1) * 100000

        eval_seed = 999000
        eval_dir = Path(args.checkpoint_root) / f"eval_round_{round_idx}"
        pre_eval_dir = eval_dir / "pre"
        pre_eval_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "tools/eval_sequence_fdm_v2.py",
            "--config", base_config,
            "--mode", "oracle",
            "--num-episodes", str(args.eval_episodes),
            "--base-seed", str(eval_seed),
            "--output-dir", str(pre_eval_dir),
        ], f"[Round {round_idx}] Pre-eval (oracle) on {args.eval_episodes} test scenes")

        collect_cmd = [
            sys.executable, "tools/collect_sequence_fdm_v2_data.py",
            "--base-config", base_config,
            "--episodes", str(episodes),
            "--num-trajectories", str(args.num_trajectories),
            "--output-dir", str(data_dir),
            "--base-seed", str(seed),
            "--workers", str(args.workers),
            "--map-bounds",
            str(args.map_bounds[0]), str(args.map_bounds[1]),
            str(args.map_bounds[2]), str(args.map_bounds[3]),
        ]
        if round_idx > 1:
            prev_ckpt = Path(args.checkpoint_root) / f"round_{round_idx - 1}"
            collect_cmd += ["--learned-model-dir", str(prev_ckpt), "--learned-traj-ratio", "0.8"]
            print(f"  Learned model: {prev_ckpt} (80% trajectories)")
        t0 = time.time()
        run(collect_cmd, f"[Round {round_idx}] Collecting {episodes} episodes x {args.num_trajectories} trajectories")
        collect_time = time.time() - t0

        # Cumulative training: use all data from round_1 to round_{round_idx}
        all_data_dirs = [str(Path(args.data_root) / f"round_{r}") for r in range(1, round_idx + 1)]
        train_cmd = [
            sys.executable, "tools/train_sequence_fdm_v2.py",
            "--data-dir", *all_data_dirs,
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

        post_eval_dir = eval_dir / "post"
        post_eval_dir.mkdir(parents=True, exist_ok=True)
        post_eval_cmd = [
            sys.executable, "tools/eval_sequence_fdm_v2.py",
            "--config", base_config,
            "--mode", "learned",
            "--model-dir", str(ckpt_dir),
            "--num-episodes", str(args.eval_episodes),
            "--base-seed", str(eval_seed),
            "--output-dir", str(post_eval_dir),
        ]
        run(post_eval_cmd, f"[Round {round_idx}] Post-eval (learned) on {args.eval_episodes} test scenes")

        metrics_path = ckpt_dir / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            best_loss = metrics.get("best_val_loss", "N/A")
        else:
            best_loss = "N/A"

        pre_eval = {}
        post_eval = {}
        pre_eval_path = pre_eval_dir / "eval_summary.json"
        post_eval_path = post_eval_dir / "eval_summary.json"
        if pre_eval_path.exists():
            with open(pre_eval_path) as f:
                pre_eval = json.load(f)
        if post_eval_path.exists():
            with open(post_eval_path) as f:
                post_eval = json.load(f)

        summary.append({
            "round": round_idx,
            "episodes": episodes,
            "num_trajectories": args.num_trajectories,
            "collect_time_s": round(collect_time, 1),
            "train_time_s": round(train_time, 1),
            "best_val_loss": best_loss,
            "pre_eval": {
                "success_rate": pre_eval.get("success_rate", "N/A"),
                "avg_steps": pre_eval.get("avg_steps", "N/A"),
                "avg_final_distance": pre_eval.get("avg_final_distance", "N/A"),
            },
            "post_eval": {
                "success_rate": post_eval.get("success_rate", "N/A"),
                "avg_steps": post_eval.get("avg_steps", "N/A"),
                "avg_final_distance": post_eval.get("avg_final_distance", "N/A"),
            },
            "data_dir": str(data_dir),
            "checkpoint_dir": str(ckpt_dir),
        })

        print(f"\n  Round {round_idx} summary:")
        print(f"    Episodes: {episodes} x {args.num_trajectories} = {episodes * args.num_trajectories} trajectories")
        print(f"    Collect: {collect_time:.0f}s, Train: {train_time:.0f}s")
        print(f"    Best val loss: {best_loss}")
        pre_sr = pre_eval.get('success_rate', 'N/A')
        post_sr = post_eval.get('success_rate', 'N/A')
        print(f"    Pre-eval (oracle):   success_rate={pre_sr}")
        print(f"    Post-eval (learned): success_rate={post_sr}")

    print(f"\n{'=' * 60}")
    print("LARGE-SCALE TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for s in summary:
        pre = s.get("pre_eval", {})
        post = s.get("post_eval", {})
        print(f"  Round {s['round']}: {s['episodes']} eps x {s['num_trajectories']} traj, "
              f"collect={s['collect_time_s']}s, train={s['train_time_s']}s, "
              f"best_loss={s['best_val_loss']}, "
              f"pre_sr={pre.get('success_rate', 'N/A')} post_sr={post.get('success_rate', 'N/A')}")
    total_time = sum(s["collect_time_s"] + s["train_time_s"] for s in summary)
    total_traj = sum(s["episodes"] * s["num_trajectories"] for s in summary)
    print(f"  Total trajectories: {total_traj}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min, {total_time/3600:.1f} h)")

    summary_path = Path(args.checkpoint_root) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
