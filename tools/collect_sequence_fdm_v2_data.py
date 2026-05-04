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


def _collect_one(args: tuple) -> dict:
    base_config_path, episode_id, terrain_seed, output_dir, map_bounds = args
    try:
        return collect_sequence_fdm_episode(
            base_config_path=base_config_path,
            episode_id=episode_id,
            terrain_seed=terrain_seed,
            output_dir=output_dir,
            map_bounds=map_bounds,
        )
    except Exception as e:
        return {"episode_id": episode_id, "error": str(e), "success": False}


def _get_gpu_memory_info():
    """Return (free_mb, total_mb) for GPU 0."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info(0)
        return free / (1024 * 1024), total / (1024 * 1024)
    except Exception:
        return None, None


def _estimate_workers_from_gpu(free_mb: float | None, total_mb: float | None) -> int:
    """Estimate safe number of parallel workers based on available GPU memory."""
    if free_mb is None:
        return 1
    # Heuristic: each episode needs ~500MB GPU memory for MPPI torch backend
    # Reserve 2GB headroom for system / spikes
    usable = max(0, free_mb - 2048)
    workers = max(1, int(usable / 500))
    return min(workers, 8)  # Cap at 8 to avoid CPU contention


def main():
    parser = argparse.ArgumentParser(description="Collect sequence FDM V2 training episodes")
    parser.add_argument("--base-config", required=True, help="Base YAML config path")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to collect")
    parser.add_argument("--output-dir", required=True, help="Output directory for episodes")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed for terrain generation")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (auto if unset)")
    parser.add_argument("--map-bounds", type=float, nargs=4, default=[-15.0, 15.0, -15.0, 15.0])
    parser.add_argument("--batch-size", type=int, default=10, help="Episodes per batch before rechecking GPU")
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

    # Process in batches to allow dynamic worker adjustment
    remaining = list(range(args.episodes))
    batch_num = 0

    while remaining:
        batch_num += 1
        # Recheck GPU memory before each batch
        free_mb, total_mb = _get_gpu_memory_info()
        current_max = _estimate_workers_from_gpu(free_mb, total_mb)
        if args.workers is not None:
            current_max = min(current_max, args.workers)

        batch_size = min(args.batch_size, len(remaining))
        batch_ids = remaining[:batch_size]
        remaining = remaining[batch_size:]

        tasks = [
            (
                args.base_config,
                args.base_seed + i,
                args.base_seed + i,
                str(output_dir),
                tuple(args.map_bounds),
            )
            for i in batch_ids
        ]

        print(f"\n[Batch {batch_num}] Collecting episodes {batch_ids[0]}-{batch_ids[-1]} "
              f"with {current_max} workers (GPU free: {free_mb:.0f}MB)")

        batch_results = []
        with ProcessPoolExecutor(max_workers=current_max) as executor:
            futures = {executor.submit(_collect_one, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                batch_results.append(result)
                if result.get("error"):
                    failed += 1
                    print(f"  FAILED episode {result['episode_id']}: {result['error']}")
                else:
                    completed += 1
                    status = "SUCCESS" if result.get("success") else "FAILURE"
                    print(f"  {status} episode {result['episode_id']}: "
                          f"steps={result.get('num_transitions', '?')}, "
                          f"final_dist={result.get('final_distance', '?'):.2f}")

        results.extend(batch_results)
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = f"{(args.episodes - completed) / rate:.0f}s" if rate > 0 else "N/A"
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
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"\n=== Collection Complete ===")
    print(f"Total: {summary['total']}, Success: {summary['success']}, "
          f"Failure: {summary['failure']}, Crashed: {summary['failed']}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
