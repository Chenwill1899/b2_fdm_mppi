#!/usr/bin/env python3
"""Parallel collection of sequence FDM V2 training episodes with dynamic GPU worker scaling."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# CUDA requires 'spawn' start method for multiprocessing
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from b2_fdm_mppi.data.sequence_fdm_collector import collect_sequence_fdm_episode


def _collect_one(args: tuple) -> list[dict]:
    (base_config_path, episode_id, terrain_seed, output_dir,
     map_bounds, num_trajectories, learned_model_dir,
     learned_traj_ratio, learned_device) = args

    # 瓶颈2: 根据 episode_id 奇偶性绑定 GPU，实现双 GPU 负载均衡
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(episode_id) % 2)

    try:
        return collect_sequence_fdm_episode(
            base_config_path=base_config_path,
            episode_id=episode_id,
            terrain_seed=terrain_seed,
            output_dir=output_dir,
            map_bounds=map_bounds,
            num_trajectories=num_trajectories,
            learned_model_dir=learned_model_dir,
            learned_traj_ratio=learned_traj_ratio,
            learned_device=learned_device,
        )
    except Exception as e:
        return [{"episode_id": episode_id, "error": str(e), "success": False}]


def _get_gpu_memory_info():
    """Return (free_mb_total, total_mb_total) across all visible GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        total_free = 0
        total_mem = 0
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            total_free += free
            total_mem += total
        return total_free / (1024 * 1024), total_mem / (1024 * 1024)
    except Exception:
        return None, None


def _estimate_workers_from_gpu(free_mb: float | None, total_mb: float | None) -> int:
    """Estimate safe number of parallel workers based on available GPU memory.

    With N GPUs, we return workers proportional to total free memory,
    but cap at 8 workers per GPU to avoid context-switch thrashing.
    """
    if free_mb is None:
        return 1
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 1

    # Heuristic: each episode needs ~500MB GPU memory
    # Reserve 2GB headroom per GPU for system / spikes
    usable = max(0, free_mb - 1536 * num_gpus)
    workers = max(1, int(usable / 400))
    # Cap at 8 per GPU to avoid thrashing
    return min(workers, 8 * num_gpus)


def main():
    parser = argparse.ArgumentParser(description="Collect sequence FDM V2 training episodes")
    parser.add_argument("--base-config", required=True, help="Base YAML config path")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to collect")
    parser.add_argument("--output-dir", required=True, help="Output directory for episodes")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed for terrain generation")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (auto if unset)")
    parser.add_argument("--map-bounds", type=float, nargs=4, default=[-15.0, 15.0, -15.0, 15.0])
    parser.add_argument("--batch-size", type=int, default=10, help="Deprecated: no longer used, kept for CLI compatibility")
    parser.add_argument("--num-trajectories", type=int, default=1,
                        help="Number of MPPI trajectories per episode (default: 1)")
    parser.add_argument("--learned-model-dir", type=str, default=None,
                        help="Directory with best_model.pt + normalization.npz for learned collection")
    parser.add_argument("--learned-traj-ratio", type=float, default=0.5,
                        help="Fraction of trajectories per episode using learned model (0.0-1.0)")
    parser.add_argument("--learned-device", type=str, default="cuda",
                        help="Device for learned model inference")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    free_mb, total_mb = _get_gpu_memory_info()
    if free_mb is not None:
        print(f"GPU memory: {free_mb:.0f}MB free / {total_mb:.0f}MB total")
    else:
        print("GPU not available, falling back to CPU (1 worker)")

    max_workers = args.workers
    if max_workers is None:
        max_workers = _estimate_workers_from_gpu(free_mb, total_mb)
    print(f"Using up to {max_workers} parallel workers")

    results = []
    completed = 0
    failed = 0
    t_start = time.time()

    # 瓶颈4: 一次性创建所有任务，只启动一次 ProcessPoolExecutor
    # 避免每 batch 销毁/重建 worker 进程的开销
    all_ids = list(range(args.episodes))
    tasks = [
        (
            args.base_config,
            args.base_seed + i,
            args.base_seed + i,
            str(output_dir),
            tuple(args.map_bounds),
            args.num_trajectories,
            args.learned_model_dir,
            args.learned_traj_ratio,
            args.learned_device,
        )
        for i in all_ids
    ]

    print(f"\n[Collection] {args.episodes} episodes with {max_workers} workers "
          f"({max_workers // 2} per GPU)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_collect_one, t): t for t in tasks}
        for future in as_completed(futures):
            result_list = future.result()
            for result in result_list:
                if result.get("error"):
                    failed += 1
                    print(f"  FAILED episode {result['episode_id']}: {result['error']}")
                else:
                    completed += 1
                    status = "SUCCESS" if result.get("success") else "FAILURE"
                    print(f"  {status} episode {result['episode_id']}: "
                          f"steps={result.get('num_transitions', '?')}, "
                          f"final_dist={result.get('final_distance', '?'):.2f}")
            results.extend(result_list)

            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = f"{(args.episodes - completed) / rate:.0f}s" if rate > 0 else "N/A"
            if completed % 30 == 0 or completed == args.episodes:
                print(f"  Progress: {completed}/{args.episodes} completed, {failed} failed, "
                      f"{rate:.1f} eps/s, ETA {eta}")

    # Final summary
    success_count = sum(1 for r in results if r.get("success") and not r.get("error"))
    failure_count = sum(1 for r in results if not r.get("success") and not r.get("error"))

    summary = {
        "total": len(results),
        "success": success_count,
        "failure": failure_count,
        "failed": failed,
        "elapsed_seconds": time.time() - t_start,
        "output_dir": str(output_dir),
        "episodes": args.episodes,
        "trajectories_per_episode": args.num_trajectories,
        "learned_model_dir": args.learned_model_dir,
        "learned_traj_ratio": args.learned_traj_ratio,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"\n=== Collection Complete ===")
    print(f"Total trajectories: {summary['total']}, Success: {summary['success']}, "
          f"Failure: {summary['failure']}, Crashed: {summary['failed']}")
    print(f"Episodes: {summary['episodes']} x {summary['trajectories_per_episode']} trajectories/episode")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
