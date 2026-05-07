#!/usr/bin/env python3
"""Compare oracle vs learned trajectories from Round 9 eval on same scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from b2_fdm_mppi.data.sequence_fdm_collector import _sample_start_goal
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator


def _reconstruct_scene(seed: int, map_bounds: tuple = (-15.0, 15.0, -15.0, 15.0)):
    """Reconstruct terrain, obstacles, start, goal from fixed seed."""
    rng = np.random.default_rng(seed)
    terrain_gen = RandomTerrainGenerator(map_bounds=map_bounds, num_patches_range=(3, 6))
    terrain = terrain_gen.generate(seed=seed)

    rng_obs = np.random.default_rng(seed + 50000)
    num_obs = int(rng_obs.integers(3, 7))
    obs_list = []
    for _ in range(num_obs):
        ox = float(rng_obs.uniform(-13.0, 13.0))
        oy = float(rng_obs.uniform(-13.0, 13.0))
        radius = float(rng_obs.uniform(0.8, 2.0))
        obs_list.append((ox, oy, radius))

    start_xy, goal_xy = _sample_start_goal(
        rng, map_bounds, min_distance=10.0, terrain=terrain, max_start_goal_risk=0.5
    )
    return terrain, obs_list, start_xy, goal_xy


def _draw_terrain_background(ax, terrain, map_bounds, grid_res=200):
    """Draw terrain risk as background heatmap."""
    x_min, x_max, y_min, y_max = map_bounds
    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(grid_res):
        for j in range(grid_res):
            Z[i, j] = terrain.risk_cost(float(X[i, j]), float(Y[i, j]))
    im = ax.imshow(
        Z, extent=[x_min, x_max, y_min, y_max], origin="lower",
        cmap="YlOrRd", vmin=0, vmax=1, alpha=0.4, aspect="auto",
    )
    return im


def plot_comparison(
    episode_id: int,
    seed: int,
    pre_dir: Path,
    post_dir: Path,
    output_dir: Path,
    map_bounds: tuple = (-15.0, 15.0, -15.0, 15.0),
):
    """Load oracle & learned trajectories and plot side-by-side."""
    pre_npz = np.load(pre_dir / f"eval_episode_{episode_id:03d}.npz", allow_pickle=True)
    post_npz = np.load(post_dir / f"eval_episode_{episode_id:03d}.npz", allow_pickle=True)

    oracle_states = pre_npz["states"]
    learned_states = post_npz["states"]

    oracle_success = bool(pre_npz["success"].item())
    learned_success = bool(post_npz["success"].item())
    oracle_steps = len(oracle_states)
    learned_steps = len(learned_states)
    oracle_fd = float(pre_npz["final_distance"].item())
    learned_fd = float(post_npz["final_distance"].item())

    # Reconstruct scene
    terrain, obs_list, start_xy, goal_xy = _reconstruct_scene(seed, map_bounds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, title, states, success, steps, fd in [
        (axes[0], "Oracle", oracle_states, oracle_success, oracle_steps, oracle_fd),
        (axes[1], "Learned (Round 9)", learned_states, learned_success, learned_steps, learned_fd),
    ]:
        _draw_terrain_background(ax, terrain, map_bounds)

        # Obstacles
        for ox, oy, radius in obs_list:
            circ = Circle((ox, oy), radius, color="black", fill=True, alpha=0.6)
            ax.add_patch(circ)

        # Trajectory
        color = "tab:blue" if success else "tab:red"
        ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2, label="trajectory")
        ax.scatter(states[0, 0], states[0, 1], color="green", s=150, marker="o", zorder=5, label="start")
        ax.scatter(goal_xy[0], goal_xy[1], color="red", s=150, marker="*", zorder=5, label="goal")

        status = "SUCCESS" if success else "FAIL"
        ax.set_title(f"{title}\n{status} | steps={steps} | final_dist={fd:.2f}", fontsize=12)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(map_bounds[0], map_bounds[1])
        ax.set_ylim(map_bounds[2], map_bounds[3])
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Episode {episode_id} (seed={seed})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"compare_episode_{episode_id:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare oracle vs learned trajectories")
    parser.add_argument("--round-dir", default="checkpoints/large_scale/eval_round_9", help="Eval round directory")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Episode IDs to plot (default: auto-select)")
    parser.add_argument("--output-dir", default="results/compare_oracle_learned", help="Output directory for plots")
    args = parser.parse_args()

    round_dir = Path(args.round_dir)
    pre_dir = round_dir / "pre"
    post_dir = round_dir / "post"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    with open(pre_dir / "eval_summary.json") as f:
        pre_summary = json.load(f)
    with open(post_dir / "eval_summary.json") as f:
        post_summary = json.load(f)

    pre_results = {r["episode_id"]: r for r in pre_summary["results"]}
    post_results = {r["episode_id"]: r for r in post_summary["results"]}

    if args.episodes:
        selected = args.episodes
    else:
        # Auto-select interesting episodes
        selected = []
        # 1. Learned success, oracle fail
        for ep in range(20):
            if not pre_results[ep]["success"] and post_results[ep]["success"]:
                selected.append(ep)
        # 2. Both success but different steps
        for ep in range(20):
            if pre_results[ep]["success"] and post_results[ep]["success"]:
                if abs(pre_results[ep]["num_transitions"] - post_results[ep]["num_transitions"]) > 50:
                    selected.append(ep)
        # 3. Both fail but different final distance
        for ep in range(20):
            if not pre_results[ep]["success"] and not post_results[ep]["success"]:
                if abs(pre_results[ep]["final_distance"] - post_results[ep]["final_distance"]) > 5:
                    selected.append(ep)
        selected = sorted(set(selected))[:6]

    print(f"Plotting {len(selected)} episodes: {selected}")
    for ep in selected:
        seed = pre_results[ep]["terrain_seed"]
        plot_comparison(ep, seed, pre_dir, post_dir, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
