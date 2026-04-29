"""NumPy MPPI controller for the B2 omnidirectional SE(2) nominal model."""

from __future__ import annotations

import numpy as np

from b2_fdm_mppi.core.omni_b2 import OmniB2


class MppiOmniNumpy:
    def __init__(
        self,
        *,
        dt: float,
        horizon_steps: int,
        num_samples: int,
        lambda_: float,
        noise_std: np.ndarray,
        max_vx: float,
        max_vy: float,
        max_wz: float,
        goal_xy_weight: float = 5.0,
        yaw_weight: float = 0.2,
        control_weight: float = 0.01,
        obstacle_weight: float = 25.0,
        robot_radius: float = 0.6,
        safety_dist: float = 0.3,
        draw_num_traj: int = 50,
        seed: int | None = None,
    ) -> None:
        self.dt = float(dt)
        self.horizon_steps = int(horizon_steps)
        self.num_samples = int(num_samples)
        self.lambda_ = float(lambda_)
        self.noise_std = np.asarray(noise_std, dtype=np.float32)
        self.max_control = np.asarray([max_vx, max_vy, max_wz], dtype=np.float32)
        self.goal_xy_weight = float(goal_xy_weight)
        self.yaw_weight = float(yaw_weight)
        self.control_weight = float(control_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.robot_radius = float(robot_radius)
        self.safety_dist = float(safety_dist)
        self.draw_num_traj = min(int(draw_num_traj), self.num_samples)
        self.rng = np.random.default_rng(seed)
        self.nominal_u = np.zeros((self.horizon_steps, 3), dtype=np.float32)
        self.model = OmniB2(self.dt, max_vx, max_vy, max_wz)

    @classmethod
    def from_config(cls, config: dict, seed: int | None = None) -> "MppiOmniNumpy":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        return cls(
            dt=dt,
            horizon_steps=horizon_steps,
            num_samples=int(mppi["num_trajectories"]),
            lambda_=float(mppi["lambda"]),
            noise_std=np.asarray(mppi["std_normal"], dtype=np.float32),
            max_vx=float(robot["max_vx"]),
            max_vy=float(robot["max_vy"]),
            max_wz=float(robot["max_wz"]),
            goal_xy_weight=float(mppi["weights"][0]),
            yaw_weight=float(mppi["weights"][2]),
            robot_radius=float(robot["radius"]),
            safety_dist=float(robot["safety_dist"]),
            draw_num_traj=int(mppi["draw_num_traj"]),
            seed=seed,
        )

    def compute_control(self, state: np.ndarray, cost_params):
        goal = np.asarray(cost_params[3], dtype=np.float32)
        obstacles = np.asarray(cost_params[4], dtype=np.float32).reshape(-1, 7)
        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=(self.num_samples, self.horizon_steps, 3),
        ).astype(np.float32)
        candidates = np.clip(
            self.nominal_u[None, :, :] + noise,
            -self.max_control,
            self.max_control,
        )
        costs = np.array(
            [self.trajectory_cost(state, candidate, goal, obstacles) for candidate in candidates],
            dtype=np.float32,
        )
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / max(self.lambda_, 1e-6))
        normalizer = float(np.sum(weights))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            weights = np.full(self.num_samples, 1.0 / self.num_samples, dtype=np.float32)
            normalizer = 1.0
        else:
            weights = weights / normalizer
        self.nominal_u = np.tensordot(weights, candidates, axes=(0, 0)).astype(np.float32)
        self.nominal_u = np.clip(self.nominal_u, -self.max_control, self.max_control)
        control = self.nominal_u[0].copy()
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].copy()
        self._shift_nominal_controls()
        return control, optimal_u, sample_u, normalizer, min_cost

    def trajectory_cost(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> float:
        states = self.model.rollout(initial_state, controls)
        final_state = states[-1]
        xy_error = final_state[:2] - goal[:2]
        yaw_error = self._angle_diff(float(final_state[2]), float(goal[2]))
        goal_cost = self.goal_xy_weight * float(np.dot(xy_error, xy_error))
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * float(np.sum(controls * controls))
        obstacle_cost = self._obstacle_cost(states[1:], obstacles)
        return goal_cost + yaw_cost + control_cost + obstacle_cost

    def _obstacle_cost(self, states: np.ndarray, obstacles: np.ndarray) -> float:
        if obstacles.size == 0:
            return 0.0
        total = 0.0
        for obstacle in obstacles:
            ox, oy, radius = obstacle[:3]
            clearance = (
                np.linalg.norm(states[:, :2] - np.array([ox, oy], dtype=np.float32), axis=1)
                - float(radius)
                - self.robot_radius
            )
            margin = self.safety_dist - clearance
            violations = margin[margin > 0.0]
            if violations.size:
                total += self.obstacle_weight * float(np.sum(violations * violations))
        return total

    def _shift_nominal_controls(self) -> None:
        self.nominal_u[:-1] = self.nominal_u[1:]
        self.nominal_u[-1] = 0.0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)
