"""Simulation runner for the NumPy B2 omnidirectional MPPI controller."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.simulation.results_path import create_results_path
from b2_fdm_mppi.visualization.utils import map_axis_limits

try:
    from b2_fdm_mppi.controllers.mppi_omni_cuda import MppiOmniCuda
except Exception:  # pragma: no cover - exercised on machines without CUDA/PyCUDA.
    MppiOmniCuda = None


ControllerFactory = Callable[..., object]


def create_omni_controller(config: dict, seed: int = 123) -> object:
    backend = str(config["mppi"].get("backend", "numpy")).lower()
    if backend == "cuda":
        if MppiOmniCuda is None:
            raise RuntimeError("mppi.backend is 'cuda' but PyCUDA controller is unavailable")
        return MppiOmniCuda.from_config(config, seed=seed)
    if backend == "numpy":
        return MppiOmniNumpy.from_config(config, seed=seed)
    raise ValueError(f"Unsupported omni MPPI backend: {backend}")


@dataclass(frozen=True)
class OmniSimulationSummary:
    steps: int
    reached_goal: bool
    failed: bool
    results_path: Path
    run_time: float


def goal_reached_xy(state: np.ndarray, target: np.ndarray, minimum_distance: float) -> bool:
    return bool(np.linalg.norm(target[:2] - state[:2]) < minimum_distance - 0.1)


class OmniMppiSimulationRunner:
    def __init__(
        self,
        config: dict,
        controller_factory: ControllerFactory | None = None,
        logger=None,
    ) -> None:
        self.config = config
        self.logger = logger
        sim = config["simulation"]
        robot_cfg = config["robot"]

        self.hz = float(sim["sampling_rate"])
        self.dt = 1.0 / self.hz
        self.max_steps = int(sim["max_steps"])
        self.minimum_distance = float(sim["minimum_distance"])
        self.state = np.asarray(sim["initial_state"], dtype=np.float32)
        self.init_pose = np.copy(self.state)
        self.goal = np.asarray(sim["goal"], dtype=np.float32)
        self.obstacles = np.asarray(config["obstacles"].get("virtual", []), dtype=np.float32).reshape(-1, 7)
        execution_cfg = config.get("execution", {})
        self.filter_enabled = bool(execution_cfg.get("filter_enabled", False))
        self.filter_alpha = float(execution_cfg.get("filter_alpha", 0.0))
        self.filter_alpha = float(np.clip(self.filter_alpha, 0.0, 1.0))
        self.rollout_max_control = np.asarray(
            [robot_cfg["max_vx"], robot_cfg["max_vy"], robot_cfg["max_wz"]],
            dtype=np.float32,
        )
        self.rollout_max_accel = np.asarray(
            [
                robot_cfg.get("max_ax", 1000.0),
                robot_cfg.get("max_ay", 1000.0),
                robot_cfg.get("max_awz", 1000.0),
            ],
            dtype=np.float32,
        )
        self.rollout_velocity_lag_beta = float(np.clip(robot_cfg.get("velocity_lag_beta", 0.0), 0.0, 1.0))
        self.robot = OmniB2(
            self.dt,
            float(robot_cfg["max_vx"]),
            float(robot_cfg["max_vy"]),
            float(robot_cfg["max_wz"]),
        )
        self.controller = (controller_factory or self._default_controller_factory)(
            config=config,
            runner=self,
        )
        self.results_path = create_results_path(config["results"])
        self.state_history: list[np.ndarray] = []
        self.control_history: list[np.ndarray] = []
        self.raw_control_history: list[np.ndarray] = []
        self.previous_exec_control = np.zeros(3, dtype=np.float32)
        self.mppi_time_history: list[float] = []
        self.min_cost_history: list[float] = []
        self.optimal_u_history: list[np.ndarray] = []
        self.sample_u_history: list[np.ndarray] = []
        self.failed = False

    def run(self) -> OmniSimulationSummary:
        steps = 0
        while steps < self.max_steps and not goal_reached_xy(self.state, self.goal, self.minimum_distance):
            self.step()
            steps += 1
            if self.failed:
                break
        self._save_results()
        if self.config["results"].get("enable_plots", True):
            self._plot_results()
        return OmniSimulationSummary(
            steps=steps,
            reached_goal=goal_reached_xy(self.state, self.goal, self.minimum_distance),
            failed=self.failed,
            results_path=self.results_path,
            run_time=steps * self.dt,
        )

    def step(self) -> np.ndarray:
        start = time.time()
        raw_u, optimal_u, sample_u, _normalizer, min_cost = self.controller.compute_control(
            self.state,
            [None, None, None, self.goal, self.obstacles, len(self.obstacles)],
        )
        elapsed_ms = (time.time() - start) * 1000.0
        raw_u = np.asarray(raw_u, dtype=np.float32)
        u = self._execution_control(raw_u)

        self.state_history.append(np.copy(self.state))
        self.raw_control_history.append(np.copy(raw_u))
        self.control_history.append(np.copy(u))
        self.mppi_time_history.append(float(elapsed_ms))
        self.min_cost_history.append(float(min_cost))
        self.optimal_u_history.append(np.copy(optimal_u))
        self.sample_u_history.append(np.copy(sample_u))

        if np.isnan(np.sum(self.state)) or np.isnan(np.sum(u)):
            self.failed = True
            return u
        self.state = self.robot.update_state(self.state, u)
        return u

    def _execution_control(self, raw_u: np.ndarray) -> np.ndarray:
        if not self.filter_enabled:
            self.previous_exec_control = np.asarray(raw_u, dtype=np.float32).copy()
            return self.previous_exec_control.copy()
        filtered = self.filter_alpha * self.previous_exec_control + (1.0 - self.filter_alpha) * raw_u
        self.previous_exec_control = np.asarray(filtered, dtype=np.float32)
        return self.previous_exec_control.copy()

    def _summary_metrics(self) -> dict:
        final_distance = float(np.linalg.norm(self.goal[:2] - self.state[:2]))
        path_length = self._path_length()
        mean_time = float(np.mean(self.mppi_time_history)) if self.mppi_time_history else 0.0
        max_time = float(np.max(self.mppi_time_history)) if self.mppi_time_history else 0.0
        success = goal_reached_xy(self.state, self.goal, self.minimum_distance)
        return {
            "success": success,
            "reached_goal": success,
            "failed": self.failed,
            "init_pose": self.init_pose.tolist(),
            "goal": self.goal.tolist(),
            "steps": len(self.state_history),
            "final_distance": final_distance,
            "path_length": path_length,
            "arrival_time": len(self.state_history) * self.dt if success else None,
            "run_time": len(self.state_history) * self.dt,
            "mean_mppi_time_ms": mean_time,
            "max_mppi_time_ms": max_time,
            "min_obstacle_clearance": self._min_obstacle_clearance(),
            "controls_csv": "executed_controls",
            "raw_controls_csv": "raw_controls",
            **self._control_metrics(),
            **self._sample_coverage_metrics(),
        }

    def _save_results(self) -> None:
        self._save_config()
        self._save_trajectory()
        self._save_controls()
        self._save_raw_controls()
        self._save_obstacles()
        pd.DataFrame({"mppi_time_ms": self.mppi_time_history}).to_csv(
            self.results_path / "time_results.csv", index=False
        )
        pd.DataFrame({"min_cost": self.min_cost_history}).to_csv(
            self.results_path / "costs.csv", index=False
        )
        summary = self._summary_metrics()
        (self.results_path / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with (self.results_path / "test_summary.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, sort_keys=False)

    def _save_config(self) -> None:
        with (self.results_path / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.config, stream, sort_keys=False)

    def _save_trajectory(self) -> None:
        rows = []
        for idx, state in enumerate(self.state_history):
            rows.append(
                {
                    "step": idx,
                    "x": state[0],
                    "y": state[1],
                    "theta": state[2],
                    "vx": state[3],
                    "vy": state[4],
                    "wz": state[5],
                    "x_des": self.goal[0],
                    "y_des": self.goal[1],
                    "theta_des": self.goal[2],
                }
            )
        pd.DataFrame(rows).to_csv(self.results_path / "trajectory.csv", index=False)
        pd.DataFrame(rows).rename(columns={"vx": "dx", "vy": "dy"}).to_csv(
            self.results_path / "results.csv", index=False
        )

    def _save_controls(self) -> None:
        rows = []
        for idx, control in enumerate(self.control_history):
            rows.append({"step": idx, "vx_cmd": control[0], "vy_cmd": control[1], "wz_cmd": control[2]})
        pd.DataFrame(rows).to_csv(self.results_path / "controls.csv", index=False)

    def _save_raw_controls(self) -> None:
        rows = []
        for idx, control in enumerate(self.raw_control_history):
            rows.append({"step": idx, "vx_cmd": control[0], "vy_cmd": control[1], "wz_cmd": control[2]})
        pd.DataFrame(rows).to_csv(self.results_path / "raw_controls.csv", index=False)

    def _save_obstacles(self) -> None:
        rows = []
        for _ in self.state_history:
            row = {}
            for idx, obstacle in enumerate(self.obstacles):
                row.update(
                    {
                        f"x{idx}": obstacle[0],
                        f"y{idx}": obstacle[1],
                        f"r{idx}": obstacle[2],
                        f"theta{idx}": obstacle[4],
                        f"dx{idx}": obstacle[5],
                        f"dy{idx}": obstacle[6],
                    }
                )
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.results_path / "obs_results.csv", index=False)

    def _plot_results(self) -> None:
        self._plot_trajectory()
        if self.config["results"].get("enable_animation", True):
            self._save_animation()

    def _plot_trajectory(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        states = np.asarray(self.state_history)
        if len(states):
            ax.plot(states[:, 0], states[:, 1], color="tab:blue", label="trajectory")
        self._draw_obstacles(ax)
        ax.scatter([self.init_pose[0]], [self.init_pose[1]], color="green", label="start")
        ax.scatter([self.goal[0]], [self.goal[1]], color="purple", label="goal")
        xlim, ylim = map_axis_limits()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.savefig(self.results_path / "trajectory.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_animation(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        states = np.asarray(self.state_history)
        fig, ax = plt.subplots(figsize=(6, 6))
        xlim, ylim = map_axis_limits()

        def update(frame):
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            self._draw_obstacles(ax)
            ax.scatter([self.init_pose[0]], [self.init_pose[1]], color="green", label="start")
            ax.scatter([self.goal[0]], [self.goal[1]], color="purple", label="goal")
            if frame >= 0 and len(states):
                self._draw_predicted_rollouts(ax, states[frame], frame)
                ax.plot(states[: frame + 1, 0], states[: frame + 1, 1], color="tab:blue")
                ax.scatter([states[frame, 0]], [states[frame, 1]], color="red")
            ax.legend(loc="upper right")

        frames = max(1, len(states))
        animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        animation.save(self.results_path / "animation.gif", writer=PillowWriter(fps=5))
        plt.close(fig)

    def _draw_predicted_rollouts(self, ax, state: np.ndarray, frame: int) -> None:
        if frame < len(self.sample_u_history):
            sampled_controls = self.sample_u_history[frame]
            max_draw = min(50, len(sampled_controls))
            for control_sequence in sampled_controls[:max_draw]:
                predicted = self._predict_trajectory(state, control_sequence)
                ax.plot(predicted[:, 0], predicted[:, 1], color="black", alpha=0.08, linewidth=0.8)
        if frame < len(self.optimal_u_history):
            predicted = self._predict_trajectory(state, self.optimal_u_history[frame])
            ax.plot(
                predicted[:, 0],
                predicted[:, 1],
                color="orange",
                alpha=0.75,
                linewidth=1.4,
                label="optimized rollout",
            )

    def _predict_trajectory(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        predicted = [np.asarray(state, dtype=np.float32).copy()]
        rollout_state = predicted[0].copy()
        prev_real = rollout_state[3:].copy()
        max_delta = self.rollout_max_accel * self.dt
        for command in controls:
            command = np.clip(np.asarray(command, dtype=np.float32), -self.rollout_max_control, self.rollout_max_control)
            lagged = self.rollout_velocity_lag_beta * prev_real + (1.0 - self.rollout_velocity_lag_beta) * command
            delta = np.clip(lagged - prev_real, -max_delta, max_delta)
            control = np.clip(prev_real + delta, -self.rollout_max_control, self.rollout_max_control)
            rollout_state = self.robot.update_state(rollout_state, control)
            predicted.append(rollout_state.copy())
            prev_real = control
        return np.asarray(predicted, dtype=np.float32)

    def _draw_obstacles(self, ax) -> None:
        import matplotlib.patches as patches

        robot_radius = float(self.config["robot"]["radius"])
        safety_dist = float(self.config["robot"]["safety_dist"])
        for obstacle in self.obstacles:
            x, y, radius = obstacle[:3]
            ax.add_patch(patches.Circle((x, y), radius, color="gray", alpha=0.4))
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    radius + robot_radius + safety_dist,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=1.0,
                )
            )

    def _path_length(self) -> float:
        if not self.state_history:
            return 0.0
        positions = [state[:2] for state in self.state_history]
        positions.append(self.state[:2])
        deltas = np.diff(np.asarray(positions, dtype=np.float32), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _min_obstacle_clearance(self) -> float | None:
        if len(self.obstacles) == 0 or not self.state_history:
            return None
        states = np.asarray(self.state_history, dtype=np.float32)
        min_clearance = np.inf
        robot_radius = float(self.config["robot"]["radius"])
        for obstacle in self.obstacles:
            clearance = (
                np.linalg.norm(states[:, :2] - obstacle[:2], axis=1)
                - float(obstacle[2])
                - robot_radius
            )
            min_clearance = min(min_clearance, float(np.min(clearance)))
        return min_clearance

    def _control_metrics(self) -> dict:
        if not self.control_history:
            return {
                "control_smoothness": 0.0,
                "smooth_vx": 0.0,
                "smooth_vy": 0.0,
                "smooth_wz": 0.0,
                "control_jerk": 0.0,
                "jerk_vx": 0.0,
                "jerk_vy": 0.0,
                "jerk_wz": 0.0,
                "acceleration_cost": 0.0,
                "lateral_usage": 0.0,
                "yaw_rate_usage": 0.0,
                "vx_variance": 0.0,
                "vy_variance": 0.0,
                "wz_variance": 0.0,
            }
        controls = np.asarray(self.control_history, dtype=np.float64)
        variances = np.var(controls, axis=0)
        if len(controls) >= 2:
            deltas = np.diff(controls, axis=0)
            smooth_by_channel = np.mean(deltas * deltas, axis=0)
            smoothness = float(np.mean(np.sum(deltas * deltas, axis=1)))
            accelerations = deltas / self.dt
            acceleration_cost = float(np.sum(accelerations * accelerations))
        else:
            smooth_by_channel = np.zeros(3, dtype=np.float64)
            smoothness = 0.0
            acceleration_cost = 0.0
        if len(controls) >= 3:
            jerks = controls[2:] - 2.0 * controls[1:-1] + controls[:-2]
            jerk_by_channel = np.mean(jerks * jerks, axis=0)
            jerk = float(np.mean(np.sum(jerks * jerks, axis=1)))
        else:
            jerk_by_channel = np.zeros(3, dtype=np.float64)
            jerk = 0.0
        return {
            "control_smoothness": smoothness,
            "smooth_vx": float(smooth_by_channel[0]),
            "smooth_vy": float(smooth_by_channel[1]),
            "smooth_wz": float(smooth_by_channel[2]),
            "control_jerk": jerk,
            "jerk_vx": float(jerk_by_channel[0]),
            "jerk_vy": float(jerk_by_channel[1]),
            "jerk_wz": float(jerk_by_channel[2]),
            "acceleration_cost": acceleration_cost,
            "lateral_usage": float(np.mean(controls[:, 1] * controls[:, 1])),
            "yaw_rate_usage": float(np.mean(controls[:, 2] * controls[:, 2])),
            "vx_variance": float(variances[0]),
            "vy_variance": float(variances[1]),
            "wz_variance": float(variances[2]),
        }

    def _sample_coverage_metrics(self) -> dict:
        empty = {
            "sample_terminal_y_std_mean": 0.0,
            "sample_terminal_y_range_mean": 0.0,
            "sample_terminal_spread_mean": 0.0,
            "sample_terminal_x_range_mean": 0.0,
        }
        if not self.sample_u_history or not self.state_history:
            return empty
        y_stds = []
        y_ranges = []
        spreads = []
        x_ranges = []
        for frame, sampled_controls in enumerate(self.sample_u_history):
            if frame >= len(self.state_history) or len(sampled_controls) == 0:
                continue
            terminals = []
            for control_sequence in sampled_controls:
                predicted = self._predict_trajectory(self.state_history[frame], control_sequence)
                terminals.append(predicted[-1, :2])
            terminal_xy = np.asarray(terminals, dtype=np.float64)
            if terminal_xy.size == 0:
                continue
            y_values = terminal_xy[:, 1]
            x_values = terminal_xy[:, 0]
            center = np.mean(terminal_xy, axis=0)
            y_stds.append(float(np.std(y_values)))
            y_ranges.append(float(np.max(y_values) - np.min(y_values)))
            x_ranges.append(float(np.max(x_values) - np.min(x_values)))
            spreads.append(float(np.mean(np.linalg.norm(terminal_xy - center[None, :], axis=1))))
        if not y_stds:
            return empty
        return {
            "sample_terminal_y_std_mean": float(np.mean(y_stds)),
            "sample_terminal_y_range_mean": float(np.mean(y_ranges)),
            "sample_terminal_spread_mean": float(np.mean(spreads)),
            "sample_terminal_x_range_mean": float(np.mean(x_ranges)),
        }

    def _default_controller_factory(self, *, config: dict, runner: "OmniMppiSimulationRunner") -> object:
        return create_omni_controller(config, seed=123)
