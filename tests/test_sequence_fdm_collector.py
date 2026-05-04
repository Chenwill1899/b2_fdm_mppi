import tempfile
from pathlib import Path

import numpy as np
import pytest

from b2_fdm_mppi.data.sequence_fdm_collector import (
    _mark_binary_risk,
    _sample_start_goal,
    _terrain_to_config,
    collect_sequence_fdm_episode,
)


def test_mark_binary_risk():
    terrain_risk = np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32)
    binary = _mark_binary_risk(terrain_risk, threshold=0.6)
    expected = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    assert np.array_equal(binary, expected)


def test_mark_binary_risk_all_safe():
    terrain_risk = np.array([0.1, 0.2, 0.3, 0.1], dtype=np.float32)
    binary = _mark_binary_risk(terrain_risk, threshold=0.6)
    expected = np.array([0, 0, 0, 0], dtype=np.float32)
    assert np.array_equal(binary, expected)


def test_sample_start_goal():
    rng = np.random.default_rng(42)
    start, goal = _sample_start_goal(rng, (-10, 10, -10, 10), min_distance=5.0)
    assert np.linalg.norm(goal - start) >= 5.0


def test_terrain_to_config():
    from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator

    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(2, 3))
    terrain = gen.generate(seed=42)
    cfg = _terrain_to_config(terrain)

    from b2_fdm_mppi.core.terrain import TerrainField

    terrain2 = TerrainField.from_config(cfg)
    assert terrain2.enabled == terrain.enabled
    assert len(terrain2.patches) == len(terrain.patches)
    for x, y in [(-5, -5), (0, 0), (5, 5)]:
        assert abs(terrain2.risk_cost(x, y) - terrain.risk_cost(x, y)) < 1e-6


def test_collect_sequence_fdm_episode(tmp_path, monkeypatch):
    """End-to-end test that verifies the collector orchestration."""
    from b2_fdm_mppi.data import sequence_fdm_collector as collector_module

    # Create a minimal fake base config
    base_config = {
        "simulation": {
            "sampling_rate": 10.0,
            "time_horizon": 2.5,
            "goal": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "initial_state": [0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
            "minimum_distance": 0.45,
            "max_steps": 80,
            "without_heading": True,
            "print_out": False,
            "world_mode": "oracle",
        },
        "mppi": {
            "backend": "numpy",
            "state_dim": 6,
            "control_dim": 3,
            "num_trajectories": 256,
            "draw_num_traj": 20,
            "exploration_variance": 1200.0,
            "weights": [2.5, 2.5, 0.5, 0.0, 0.0, 0.0],
            "dist_type": 2,
            "std_normal": [0.22, 0.16, 0.16],
            "std_slack": 0.1,
            "lambda": 0.7,
            "obstacle_weight": 300.0,
            "obstacle_soft_weight": 0.5,
            "obstacle_influence_dist": 1.2,
            "cbf_weight": 0.0,
            "control_weight": 0.02,
            "smooth_weight": 1.0,
            "lateral_weight": 0.2,
            "yaw_rate_weight": 0.05,
            "accel_weight": 0.45,
            "jerk_weight": 0.0,
            "terrain_risk_weight": 0.0,
            "terrain_risk_power": 2.0,
            "terrain_risk_threshold": 0.0,
            "terrain_risk_mode": "excess",
            "beta_1": 0.5,
            "beta_2": 0.8,
            "beta_3": 0.5,
            "sg_window": 51,
            "sg_poly_order": 3,
            "soft_cbf": False,
        },
        "robot": {
            "state_dim": 6,
            "max_vx": 1.5,
            "max_vy": 0.1,
            "max_wz": 1,
            "max_ax": 0.8,
            "max_ay": 0.2,
            "max_awz": 0.8,
            "velocity_lag_beta": 0.35,
            "radius": 0.6,
            "safety_dist": 0.25,
            "obstacle_state_num": 7,
        },
        "cbf": {
            "enabled": False,
            "type": 0,
            "dcbf_alpha": 0.1,
            "dcbf_weight": 1000000.0,
            "slack_weight": 100.0,
            "check_dist": 6.0,
            "max_slack_vari": 1.0,
            "atau": 0.2,
        },
        "obstacles": {
            "num_max": 5,
            "static_enabled": True,
            "virtual": [],
        },
        "results": {
            "root": "./results/sim_results",
            "run_name": "fdm_mppi_smoke_latest",
            "timestamp_suffix": False,
            "overwrite": True,
            "enable_plots": True,
            "enable_animation": True,
        },
        "execution": {
            "filter_enabled": False,
            "filter_alpha": 0.0,
        },
        "terrain": {
            "enabled": True,
            "friction_base": 0.72,
            "slope_scale": 0.10,
            "roughness_scale": 0.25,
            "friction_slope_scale": 0.18,
            "friction_roughness_scale": 0.10,
            "goal_relief": {
                "enabled": True,
                "center": [1.2, 0.0],
                "sigma": [2.0, 1.2],
                "strength": 0.75,
                "floor": 0.25,
            },
        },
        "oracle_residual": {
            "enabled": True,
            "alpha": 0.35,
            "residual_scale": 0.35,
            "noise_std": 0.012,
            "max_residual_ratio": 0.3,
            "seed": 123,
        },
    }

    base_config_path = tmp_path / "base_config.yaml"
    import yaml

    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)

    output_dir = tmp_path / "episodes"
    output_path = output_dir / "episode_000007_traj_00.npz"

    # Fake collect_oracle_episode to avoid full simulation
    def fake_collect_oracle_episode(*, config_path, episode_id, seed, output_path):
        # Create a minimal fake NPZ that matches what the collector expects
        num_transitions = 5
        np.savez_compressed(
            output_path,
            states=np.zeros((num_transitions, 6), dtype=np.float32),
            next_states=np.zeros((num_transitions, 6), dtype=np.float32),
            cmd_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            real_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            exec_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            oracle_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            terrain_features=np.zeros((num_transitions, 4), dtype=np.float32),
            terrain_risk=np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32),
            episode_ids=np.full(num_transitions, episode_id, dtype=np.int64),
            steps=np.arange(num_transitions, dtype=np.int64),
            success=np.asarray(True),
            failed=np.asarray(False),
            start_goal_distance=np.asarray(15.0, dtype=np.float32),
            final_distance=np.asarray(0.5, dtype=np.float32),
            min_obstacle_clearance=np.asarray(1.0, dtype=np.float32),
        )
        return {
            "episode_id": episode_id,
            "num_transitions": num_transitions,
            "success": True,
            "failed": False,
            "start_goal_distance": 15.0,
            "final_distance": 0.5,
            "min_obstacle_clearance": 1.0,
            "results_path": str(tmp_path / "raw_results" / f"episode_{episode_id:06d}"),
            "output_path": str(output_path),
        }

    monkeypatch.setattr(
        collector_module, "collect_oracle_episode", fake_collect_oracle_episode
    )

    result = collect_sequence_fdm_episode(
        base_config_path=base_config_path,
        episode_id=7,
        terrain_seed=42,
        output_dir=output_dir,
        map_bounds=(-10, 10, -10, 10),
        min_start_goal_distance=5.0,
        risk_threshold=0.6,
        num_patches_range=(2, 3),
    )

    # Returns list of 1 dict even for num_trajectories=1
    assert isinstance(result, list)
    assert len(result) == 1
    metadata = result[0]

    assert metadata["episode_id"] == 7
    assert metadata["terrain_seed"] == 42
    assert "start_xy" in metadata
    assert "goal_xy" in metadata
    assert "binary_risk" in metadata
    assert metadata["traj_idx"] == 0

    # Verify the saved NPZ contains the augmented fields
    data = np.load(output_path)
    assert "binary_risk" in data
    assert "terrain_seed" in data
    assert "start_xy" in data
    assert "goal_xy" in data
    assert "traj_idx" in data
    expected_binary = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    assert np.array_equal(data["binary_risk"], expected_binary)
    assert int(data["terrain_seed"]) == 42
    assert int(data["traj_idx"]) == 0


def test_collect_multitrajectory_output_structure(tmp_path, monkeypatch):
    """num_trajectories=3 → 3 NPZ files exist, traj_idx 0/1/2."""
    from b2_fdm_mppi.data import sequence_fdm_collector as collector_module

    base_config = {
        "simulation": {
            "sampling_rate": 10.0,
            "time_horizon": 2.5,
            "goal": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "initial_state": [0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
            "minimum_distance": 0.45,
            "max_steps": 80,
            "without_heading": True,
            "print_out": False,
            "world_mode": "oracle",
        },
        "mppi": {
            "backend": "numpy",
            "state_dim": 6,
            "control_dim": 3,
            "num_trajectories": 256,
            "draw_num_traj": 20,
            "exploration_variance": 1200.0,
            "weights": [2.5, 2.5, 0.5, 0.0, 0.0, 0.0],
            "dist_type": 2,
            "std_normal": [0.22, 0.16, 0.16],
            "std_slack": 0.1,
            "lambda": 0.7,
            "obstacle_weight": 300.0,
            "obstacle_soft_weight": 0.5,
            "obstacle_influence_dist": 1.2,
            "cbf_weight": 0.0,
            "control_weight": 0.02,
            "smooth_weight": 1.0,
            "lateral_weight": 0.2,
            "yaw_rate_weight": 0.05,
            "accel_weight": 0.45,
            "jerk_weight": 0.0,
            "terrain_risk_weight": 0.0,
            "terrain_risk_power": 2.0,
            "terrain_risk_threshold": 0.0,
            "terrain_risk_mode": "excess",
            "beta_1": 0.5,
            "beta_2": 0.8,
            "beta_3": 0.5,
            "sg_window": 51,
            "sg_poly_order": 3,
            "soft_cbf": False,
        },
        "robot": {
            "state_dim": 6,
            "max_vx": 1.5,
            "max_vy": 0.1,
            "max_wz": 1,
            "max_ax": 0.8,
            "max_ay": 0.2,
            "max_awz": 0.8,
            "velocity_lag_beta": 0.35,
            "radius": 0.6,
            "safety_dist": 0.25,
            "obstacle_state_num": 7,
        },
        "cbf": {
            "enabled": False,
            "type": 0,
            "dcbf_alpha": 0.1,
            "dcbf_weight": 1000000.0,
            "slack_weight": 100.0,
            "check_dist": 6.0,
            "max_slack_vari": 1.0,
            "atau": 0.2,
        },
        "obstacles": {
            "num_max": 5,
            "static_enabled": True,
            "virtual": [],
        },
        "results": {
            "root": "./results/sim_results",
            "run_name": "fdm_mppi_smoke_latest",
            "timestamp_suffix": False,
            "overwrite": True,
            "enable_plots": True,
            "enable_animation": True,
        },
        "execution": {
            "filter_enabled": False,
            "filter_alpha": 0.0,
        },
        "terrain": {
            "enabled": True,
            "friction_base": 0.72,
            "slope_scale": 0.10,
            "roughness_scale": 0.25,
            "friction_slope_scale": 0.18,
            "friction_roughness_scale": 0.10,
            "goal_relief": {
                "enabled": True,
                "center": [1.2, 0.0],
                "sigma": [2.0, 1.2],
                "strength": 0.75,
                "floor": 0.25,
            },
        },
        "oracle_residual": {
            "enabled": True,
            "alpha": 0.35,
            "residual_scale": 0.35,
            "noise_std": 0.012,
            "max_residual_ratio": 0.3,
            "seed": 123,
        },
    }

    import yaml

    base_config_path = tmp_path / "base_config.yaml"
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)

    output_dir = tmp_path / "episodes"

    def fake_collect_oracle_episode(*, config_path, episode_id, seed, output_path):
        num_transitions = 5
        np.savez_compressed(
            output_path,
            states=np.zeros((num_transitions, 6), dtype=np.float32),
            next_states=np.zeros((num_transitions, 6), dtype=np.float32),
            cmd_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            real_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            exec_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            oracle_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            terrain_features=np.zeros((num_transitions, 4), dtype=np.float32),
            terrain_risk=np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32),
            episode_ids=np.full(num_transitions, episode_id, dtype=np.int64),
            steps=np.arange(num_transitions, dtype=np.int64),
            success=np.asarray(True),
            failed=np.asarray(False),
            start_goal_distance=np.asarray(15.0, dtype=np.float32),
            final_distance=np.asarray(0.5, dtype=np.float32),
            min_obstacle_clearance=np.asarray(1.0, dtype=np.float32),
        )
        return {
            "episode_id": episode_id,
            "num_transitions": num_transitions,
            "success": True,
            "failed": False,
            "start_goal_distance": 15.0,
            "final_distance": 0.5,
            "min_obstacle_clearance": 1.0,
            "results_path": str(tmp_path / "raw_results" / f"episode_{episode_id:06d}"),
            "output_path": str(output_path),
        }

    monkeypatch.setattr(
        collector_module, "collect_oracle_episode", fake_collect_oracle_episode
    )

    result = collect_sequence_fdm_episode(
        base_config_path=base_config_path,
        episode_id=7,
        terrain_seed=42,
        output_dir=output_dir,
        map_bounds=(-10, 10, -10, 10),
        min_start_goal_distance=5.0,
        risk_threshold=0.6,
        num_patches_range=(2, 3),
        num_trajectories=3,
    )

    # Expect 3 NPZ files: episode_000007_traj_00.npz, _traj_01.npz, _traj_02.npz
    assert (output_dir / "episode_000007_traj_00.npz").exists()
    assert (output_dir / "episode_000007_traj_01.npz").exists()
    assert (output_dir / "episode_000007_traj_02.npz").exists()
    # No single-episode file
    assert not (output_dir / "episode_000007.npz").exists()


def test_collect_multitrajectory_same_terrain(tmp_path, monkeypatch):
    """All 3 files share same terrain_seed."""
    from b2_fdm_mppi.data import sequence_fdm_collector as collector_module

    base_config = {
        "simulation": {
            "sampling_rate": 10.0,
            "time_horizon": 2.5,
            "goal": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "initial_state": [0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
            "minimum_distance": 0.45,
            "max_steps": 80,
            "without_heading": True,
            "print_out": False,
            "world_mode": "oracle",
        },
        "mppi": {
            "backend": "numpy",
            "state_dim": 6,
            "control_dim": 3,
            "num_trajectories": 256,
            "draw_num_traj": 20,
            "exploration_variance": 1200.0,
            "weights": [2.5, 2.5, 0.5, 0.0, 0.0, 0.0],
            "dist_type": 2,
            "std_normal": [0.22, 0.16, 0.16],
            "std_slack": 0.1,
            "lambda": 0.7,
            "obstacle_weight": 300.0,
            "obstacle_soft_weight": 0.5,
            "obstacle_influence_dist": 1.2,
            "cbf_weight": 0.0,
            "control_weight": 0.02,
            "smooth_weight": 1.0,
            "lateral_weight": 0.2,
            "yaw_rate_weight": 0.05,
            "accel_weight": 0.45,
            "jerk_weight": 0.0,
            "terrain_risk_weight": 0.0,
            "terrain_risk_power": 2.0,
            "terrain_risk_threshold": 0.0,
            "terrain_risk_mode": "excess",
            "beta_1": 0.5,
            "beta_2": 0.8,
            "beta_3": 0.5,
            "sg_window": 51,
            "sg_poly_order": 3,
            "soft_cbf": False,
        },
        "robot": {
            "state_dim": 6,
            "max_vx": 1.5,
            "max_vy": 0.1,
            "max_wz": 1,
            "max_ax": 0.8,
            "max_ay": 0.2,
            "max_awz": 0.8,
            "velocity_lag_beta": 0.35,
            "radius": 0.6,
            "safety_dist": 0.25,
            "obstacle_state_num": 7,
        },
        "cbf": {
            "enabled": False,
            "type": 0,
            "dcbf_alpha": 0.1,
            "dcbf_weight": 1000000.0,
            "slack_weight": 100.0,
            "check_dist": 6.0,
            "max_slack_vari": 1.0,
            "atau": 0.2,
        },
        "obstacles": {
            "num_max": 5,
            "static_enabled": True,
            "virtual": [],
        },
        "results": {
            "root": "./results/sim_results",
            "run_name": "fdm_mppi_smoke_latest",
            "timestamp_suffix": False,
            "overwrite": True,
            "enable_plots": True,
            "enable_animation": True,
        },
        "execution": {
            "filter_enabled": False,
            "filter_alpha": 0.0,
        },
        "terrain": {
            "enabled": True,
            "friction_base": 0.72,
            "slope_scale": 0.10,
            "roughness_scale": 0.25,
            "friction_slope_scale": 0.18,
            "friction_roughness_scale": 0.10,
            "goal_relief": {
                "enabled": True,
                "center": [1.2, 0.0],
                "sigma": [2.0, 1.2],
                "strength": 0.75,
                "floor": 0.25,
            },
        },
        "oracle_residual": {
            "enabled": True,
            "alpha": 0.35,
            "residual_scale": 0.35,
            "noise_std": 0.012,
            "max_residual_ratio": 0.3,
            "seed": 123,
        },
    }

    import yaml

    base_config_path = tmp_path / "base_config.yaml"
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)

    output_dir = tmp_path / "episodes"

    def fake_collect_oracle_episode(*, config_path, episode_id, seed, output_path):
        num_transitions = 5
        np.savez_compressed(
            output_path,
            states=np.zeros((num_transitions, 6), dtype=np.float32),
            next_states=np.zeros((num_transitions, 6), dtype=np.float32),
            cmd_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            real_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            exec_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            oracle_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            terrain_features=np.zeros((num_transitions, 4), dtype=np.float32),
            terrain_risk=np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32),
            episode_ids=np.full(num_transitions, episode_id, dtype=np.int64),
            steps=np.arange(num_transitions, dtype=np.int64),
            success=np.asarray(True),
            failed=np.asarray(False),
            start_goal_distance=np.asarray(15.0, dtype=np.float32),
            final_distance=np.asarray(0.5, dtype=np.float32),
            min_obstacle_clearance=np.asarray(1.0, dtype=np.float32),
        )
        return {
            "episode_id": episode_id,
            "num_transitions": num_transitions,
            "success": True,
            "failed": False,
            "start_goal_distance": 15.0,
            "final_distance": 0.5,
            "min_obstacle_clearance": 1.0,
            "results_path": str(tmp_path / "raw_results" / f"episode_{episode_id:06d}"),
            "output_path": str(output_path),
        }

    monkeypatch.setattr(
        collector_module, "collect_oracle_episode", fake_collect_oracle_episode
    )

    collect_sequence_fdm_episode(
        base_config_path=base_config_path,
        episode_id=7,
        terrain_seed=42,
        output_dir=output_dir,
        map_bounds=(-10, 10, -10, 10),
        min_start_goal_distance=5.0,
        risk_threshold=0.6,
        num_patches_range=(2, 3),
        num_trajectories=3,
    )

    # Load all 3 NPZ files and verify they share the same terrain_seed
    seeds = []
    for traj_idx in range(3):
        path = output_dir / f"episode_000007_traj_{traj_idx:02d}.npz"
        data = np.load(path)
        seeds.append(int(data["terrain_seed"]))
    assert seeds[0] == seeds[1] == seeds[2] == 42


def test_collect_multitrajectory_different_start_goal(tmp_path, monkeypatch):
    """Start/goal differ across trajectories."""
    from b2_fdm_mppi.data import sequence_fdm_collector as collector_module

    base_config = {
        "simulation": {
            "sampling_rate": 10.0,
            "time_horizon": 2.5,
            "goal": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "initial_state": [0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
            "minimum_distance": 0.45,
            "max_steps": 80,
            "without_heading": True,
            "print_out": False,
            "world_mode": "oracle",
        },
        "mppi": {
            "backend": "numpy",
            "state_dim": 6,
            "control_dim": 3,
            "num_trajectories": 256,
            "draw_num_traj": 20,
            "exploration_variance": 1200.0,
            "weights": [2.5, 2.5, 0.5, 0.0, 0.0, 0.0],
            "dist_type": 2,
            "std_normal": [0.22, 0.16, 0.16],
            "std_slack": 0.1,
            "lambda": 0.7,
            "obstacle_weight": 300.0,
            "obstacle_soft_weight": 0.5,
            "obstacle_influence_dist": 1.2,
            "cbf_weight": 0.0,
            "control_weight": 0.02,
            "smooth_weight": 1.0,
            "lateral_weight": 0.2,
            "yaw_rate_weight": 0.05,
            "accel_weight": 0.45,
            "jerk_weight": 0.0,
            "terrain_risk_weight": 0.0,
            "terrain_risk_power": 2.0,
            "terrain_risk_threshold": 0.0,
            "terrain_risk_mode": "excess",
            "beta_1": 0.5,
            "beta_2": 0.8,
            "beta_3": 0.5,
            "sg_window": 51,
            "sg_poly_order": 3,
            "soft_cbf": False,
        },
        "robot": {
            "state_dim": 6,
            "max_vx": 1.5,
            "max_vy": 0.1,
            "max_wz": 1,
            "max_ax": 0.8,
            "max_ay": 0.2,
            "max_awz": 0.8,
            "velocity_lag_beta": 0.35,
            "radius": 0.6,
            "safety_dist": 0.25,
            "obstacle_state_num": 7,
        },
        "cbf": {
            "enabled": False,
            "type": 0,
            "dcbf_alpha": 0.1,
            "dcbf_weight": 1000000.0,
            "slack_weight": 100.0,
            "check_dist": 6.0,
            "max_slack_vari": 1.0,
            "atau": 0.2,
        },
        "obstacles": {
            "num_max": 5,
            "static_enabled": True,
            "virtual": [],
        },
        "results": {
            "root": "./results/sim_results",
            "run_name": "fdm_mppi_smoke_latest",
            "timestamp_suffix": False,
            "overwrite": True,
            "enable_plots": True,
            "enable_animation": True,
        },
        "execution": {
            "filter_enabled": False,
            "filter_alpha": 0.0,
        },
        "terrain": {
            "enabled": True,
            "friction_base": 0.72,
            "slope_scale": 0.10,
            "roughness_scale": 0.25,
            "friction_slope_scale": 0.18,
            "friction_roughness_scale": 0.10,
            "goal_relief": {
                "enabled": True,
                "center": [1.2, 0.0],
                "sigma": [2.0, 1.2],
                "strength": 0.75,
                "floor": 0.25,
            },
        },
        "oracle_residual": {
            "enabled": True,
            "alpha": 0.35,
            "residual_scale": 0.35,
            "noise_std": 0.012,
            "max_residual_ratio": 0.3,
            "seed": 123,
        },
    }

    import yaml

    base_config_path = tmp_path / "base_config.yaml"
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)

    output_dir = tmp_path / "episodes"

    def fake_collect_oracle_episode(*, config_path, episode_id, seed, output_path):
        num_transitions = 5
        np.savez_compressed(
            output_path,
            states=np.zeros((num_transitions, 6), dtype=np.float32),
            next_states=np.zeros((num_transitions, 6), dtype=np.float32),
            cmd_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            real_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            exec_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            oracle_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            terrain_features=np.zeros((num_transitions, 4), dtype=np.float32),
            terrain_risk=np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32),
            episode_ids=np.full(num_transitions, episode_id, dtype=np.int64),
            steps=np.arange(num_transitions, dtype=np.int64),
            success=np.asarray(True),
            failed=np.asarray(False),
            start_goal_distance=np.asarray(15.0, dtype=np.float32),
            final_distance=np.asarray(0.5, dtype=np.float32),
            min_obstacle_clearance=np.asarray(1.0, dtype=np.float32),
        )
        return {
            "episode_id": episode_id,
            "num_transitions": num_transitions,
            "success": True,
            "failed": False,
            "start_goal_distance": 15.0,
            "final_distance": 0.5,
            "min_obstacle_clearance": 1.0,
            "results_path": str(tmp_path / "raw_results" / f"episode_{episode_id:06d}"),
            "output_path": str(output_path),
        }

    monkeypatch.setattr(
        collector_module, "collect_oracle_episode", fake_collect_oracle_episode
    )

    collect_sequence_fdm_episode(
        base_config_path=base_config_path,
        episode_id=7,
        terrain_seed=42,
        output_dir=output_dir,
        map_bounds=(-10, 10, -10, 10),
        min_start_goal_distance=5.0,
        risk_threshold=0.6,
        num_patches_range=(2, 3),
        num_trajectories=3,
    )

    # Load all 3 NPZ files and verify start_xy differs between at least 2 of 3
    start_xy_list = []
    for traj_idx in range(3):
        path = output_dir / f"episode_000007_traj_{traj_idx:02d}.npz"
        data = np.load(path)
        start_xy_list.append(data["start_xy"])

    # At least two trajectories must have different start positions
    different_count = sum(
        1 for i in range(3) for j in range(i + 1, 3)
        if not np.array_equal(start_xy_list[i], start_xy_list[j])
    )
    assert different_count >= 2, (
        f"Expected start_xy to differ across trajectories, but got: {start_xy_list}"
    )


def test_backward_compat_num_trajectories_1(tmp_path, monkeypatch):
    """num_trajectories=1 default → returns list of 1 dict, single file."""
    from b2_fdm_mppi.data import sequence_fdm_collector as collector_module

    base_config = {
        "simulation": {
            "sampling_rate": 10.0,
            "time_horizon": 2.5,
            "goal": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "initial_state": [0.01, 0.01, 0.0, 0.0, 0.0, 0.0],
            "minimum_distance": 0.45,
            "max_steps": 80,
            "without_heading": True,
            "print_out": False,
            "world_mode": "oracle",
        },
        "mppi": {
            "backend": "numpy",
            "state_dim": 6,
            "control_dim": 3,
            "num_trajectories": 256,
            "draw_num_traj": 20,
            "exploration_variance": 1200.0,
            "weights": [2.5, 2.5, 0.5, 0.0, 0.0, 0.0],
            "dist_type": 2,
            "std_normal": [0.22, 0.16, 0.16],
            "std_slack": 0.1,
            "lambda": 0.7,
            "obstacle_weight": 300.0,
            "obstacle_soft_weight": 0.5,
            "obstacle_influence_dist": 1.2,
            "cbf_weight": 0.0,
            "control_weight": 0.02,
            "smooth_weight": 1.0,
            "lateral_weight": 0.2,
            "yaw_rate_weight": 0.05,
            "accel_weight": 0.45,
            "jerk_weight": 0.0,
            "terrain_risk_weight": 0.0,
            "terrain_risk_power": 2.0,
            "terrain_risk_threshold": 0.0,
            "terrain_risk_mode": "excess",
            "beta_1": 0.5,
            "beta_2": 0.8,
            "beta_3": 0.5,
            "sg_window": 51,
            "sg_poly_order": 3,
            "soft_cbf": False,
        },
        "robot": {
            "state_dim": 6,
            "max_vx": 1.5,
            "max_vy": 0.1,
            "max_wz": 1,
            "max_ax": 0.8,
            "max_ay": 0.2,
            "max_awz": 0.8,
            "velocity_lag_beta": 0.35,
            "radius": 0.6,
            "safety_dist": 0.25,
            "obstacle_state_num": 7,
        },
        "cbf": {
            "enabled": False,
            "type": 0,
            "dcbf_alpha": 0.1,
            "dcbf_weight": 1000000.0,
            "slack_weight": 100.0,
            "check_dist": 6.0,
            "max_slack_vari": 1.0,
            "atau": 0.2,
        },
        "obstacles": {
            "num_max": 5,
            "static_enabled": True,
            "virtual": [],
        },
        "results": {
            "root": "./results/sim_results",
            "run_name": "fdm_mppi_smoke_latest",
            "timestamp_suffix": False,
            "overwrite": True,
            "enable_plots": True,
            "enable_animation": True,
        },
        "execution": {
            "filter_enabled": False,
            "filter_alpha": 0.0,
        },
        "terrain": {
            "enabled": True,
            "friction_base": 0.72,
            "slope_scale": 0.10,
            "roughness_scale": 0.25,
            "friction_slope_scale": 0.18,
            "friction_roughness_scale": 0.10,
            "goal_relief": {
                "enabled": True,
                "center": [1.2, 0.0],
                "sigma": [2.0, 1.2],
                "strength": 0.75,
                "floor": 0.25,
            },
        },
        "oracle_residual": {
            "enabled": True,
            "alpha": 0.35,
            "residual_scale": 0.35,
            "noise_std": 0.012,
            "max_residual_ratio": 0.3,
            "seed": 123,
        },
    }

    import yaml

    base_config_path = tmp_path / "base_config.yaml"
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)

    output_dir = tmp_path / "episodes"

    def fake_collect_oracle_episode(*, config_path, episode_id, seed, output_path):
        num_transitions = 5
        np.savez_compressed(
            output_path,
            states=np.zeros((num_transitions, 6), dtype=np.float32),
            next_states=np.zeros((num_transitions, 6), dtype=np.float32),
            cmd_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            real_controls=np.zeros((num_transitions, 3), dtype=np.float32),
            exec_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            oracle_residuals=np.zeros((num_transitions, 3), dtype=np.float32),
            terrain_features=np.zeros((num_transitions, 4), dtype=np.float32),
            terrain_risk=np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32),
            episode_ids=np.full(num_transitions, episode_id, dtype=np.int64),
            steps=np.arange(num_transitions, dtype=np.int64),
            success=np.asarray(True),
            failed=np.asarray(False),
            start_goal_distance=np.asarray(15.0, dtype=np.float32),
            final_distance=np.asarray(0.5, dtype=np.float32),
            min_obstacle_clearance=np.asarray(1.0, dtype=np.float32),
        )
        return {
            "episode_id": episode_id,
            "num_transitions": num_transitions,
            "success": True,
            "failed": False,
            "start_goal_distance": 15.0,
            "final_distance": 0.5,
            "min_obstacle_clearance": 1.0,
            "results_path": str(tmp_path / "raw_results" / f"episode_{episode_id:06d}"),
            "output_path": str(output_path),
        }

    monkeypatch.setattr(
        collector_module, "collect_oracle_episode", fake_collect_oracle_episode
    )

    result = collect_sequence_fdm_episode(
        base_config_path=base_config_path,
        episode_id=7,
        terrain_seed=42,
        output_dir=output_dir,
        map_bounds=(-10, 10, -10, 10),
        min_start_goal_distance=5.0,
        risk_threshold=0.6,
        num_patches_range=(2, 3),
        num_trajectories=1,
    )

    # When num_trajectories=1, result should be a list of 1 dict
    assert isinstance(result, list)
    assert len(result) == 1
    # Single NPZ file named with traj_00 suffix
    assert (output_dir / "episode_000007_traj_00.npz").exists()
