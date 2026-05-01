import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_eval_module():
    module_path = Path("tools/evaluate_residual_fdm_rollout.py")
    spec = importlib.util.spec_from_file_location("evaluate_residual_fdm_rollout_tool", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConstantTerrain:
    def feature(self, x: float, y: float) -> np.ndarray:
        return np.array([0.1, -0.05, 0.2, 0.8], dtype=np.float32)

    def risk_cost(self, x: float, y: float, *, features=None) -> float:
        return 0.3


class SimpleRobot:
    state_dim = 6

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = dt
        self.max_vx = 10.0
        self.max_vy = 10.0
        self.max_wz = 10.0

    def clip_control(self, control):
        return np.asarray(control, dtype=np.float32)

    def update_state(self, state, control):
        next_state = np.asarray(state, dtype=np.float32).copy()
        control = np.asarray(control, dtype=np.float32)
        next_state[:3] += control * self.dt
        next_state[3:] = control
        return next_state


def test_compute_rollout_metrics_reports_learned_improvement():
    module = load_eval_module()
    oracle_states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    nominal_states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    learned_states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.9, 0.0, 0.0],
            [1.8, 0.0, 0.0, 0.9, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    oracle_residuals = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    learned_residuals = np.array([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)

    metrics = module.compute_rollout_metrics(
        oracle_states=oracle_states,
        nominal_states=nominal_states,
        learned_states=learned_states,
        oracle_residuals=oracle_residuals,
        learned_residuals=learned_residuals,
    )

    assert metrics["nominal_ade_xy"] > metrics["learned_ade_xy"]
    assert metrics["nominal_fde_xy"] > metrics["learned_fde_xy"]
    assert metrics["learned_vs_nominal_ade_improvement_pct"] > 0.0
    assert metrics["residual_mse"] < metrics["zero_residual_mse"]


def test_replay_controls_uses_predicted_residuals():
    module = load_eval_module()
    robot = SimpleRobot(dt=1.0)
    terrain = ConstantTerrain()
    initial_state = np.zeros(6, dtype=np.float32)
    cmd_controls = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    replay = module.replay_controls(
        initial_state=initial_state,
        cmd_controls=cmd_controls,
        robot=robot,
        terrain=terrain,
        residual_predictor=lambda state, command, features, risk: np.array([0.5, 0.0, 0.0], dtype=np.float32),
    )

    assert replay["states"].shape == (3, 6)
    assert replay["predicted_residuals"].shape == (2, 3)
    assert np.allclose(replay["real_controls"][:, 0], 1.5)
    assert np.allclose(replay["states"][-1, 0], 3.0)


def test_save_rollout_comparison_gif_writes_nonempty_file(tmp_path):
    module = load_eval_module()
    config = {
        "simulation": {
            "goal": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "map_size": [4.0, 4.0],
            "map_origin": [-1.0, -2.0],
        },
        "robot": {"radius": 0.2, "safety_dist": 0.1},
        "obstacles": {
            "virtual": [[1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0]],
        },
        "terrain": {"enabled": True, "slope_scale": 0.1, "roughness_scale": 0.2},
        "visualization": {"terrain_grid_resolution": 12},
    }
    oracle_states = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    nominal_states = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, -0.1, 0.0, 0.6, -0.1, 0.0]],
        dtype=np.float32,
    )
    learned_states = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.9, 0.0, 0.0]],
        dtype=np.float32,
    )

    module._save_rollout_comparison_gif(
        output_dir=tmp_path,
        config=config,
        oracle_states=oracle_states,
        nominal_states=nominal_states,
        learned_states=learned_states,
        fps=2,
        max_frames=4,
    )

    gif_path = tmp_path / "rollout_compare.gif"
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_parameter_snapshot_records_safety_distance():
    module = load_eval_module()
    config = {
        "robot": {"radius": 0.6, "safety_dist": 0.25},
        "obstacles": {
            "virtual": [
                [6.0, -0.5, 0.4, 0.0, 0.0, 0.0, 0.0],
                [12.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
        },
    }

    snapshot = module.parameter_snapshot(config)

    assert snapshot["robot_radius"] == 0.6
    assert snapshot["robot_safety_dist"] == 0.25
    assert snapshot["obstacle_radii"] == [0.4, 0.5]
    assert snapshot["visualized_safety_boundary_radii"] == [1.25, 1.35]
