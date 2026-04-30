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
        smooth_weight: float = 0.2,
        obstacle_weight: float = 25.0,
        obstacle_soft_weight: float = 0.0,
        obstacle_influence_dist: float = 0.0,
        max_ax: float = 1000.0,
        max_ay: float = 1000.0,
        max_awz: float = 1000.0,
        velocity_lag_beta: float = 0.0,
        lateral_weight: float = 0.0,
        yaw_rate_weight: float = 0.0,
        accel_weight: float = 0.0,
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
        self.smooth_weight = float(smooth_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_soft_weight = float(obstacle_soft_weight)
        self.obstacle_influence_dist = float(obstacle_influence_dist)
        self.max_accel = np.asarray([max_ax, max_ay, max_awz], dtype=np.float32)
        self.velocity_lag_beta = float(np.clip(velocity_lag_beta, 0.0, 1.0))
        self.lateral_weight = float(lateral_weight)
        self.yaw_rate_weight = float(yaw_rate_weight)
        self.accel_weight = float(accel_weight)
        self.robot_radius = float(robot_radius)
        self.safety_dist = float(safety_dist)
        self.draw_num_traj = min(int(draw_num_traj), self.num_samples)
        self.rng = np.random.default_rng(seed)
        self.nominal_u = np.zeros((self.horizon_steps, 3), dtype=np.float32)
        self.previous_control = np.zeros(3, dtype=np.float32)
        self.model = OmniB2(self.dt, max_vx, max_vy, max_wz)

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        **overrides,
    ) -> "MppiOmniNumpy":
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
            goal_xy_weight=float(overrides.get("goal_xy_weight", mppi["weights"][0])),
            yaw_weight=float(overrides.get("yaw_weight", mppi["weights"][2])),
            control_weight=float(overrides.get("control_weight", mppi.get("control_weight", 0.01))),
            smooth_weight=float(overrides.get("smooth_weight", mppi.get("smooth_weight", 0.2))),
            obstacle_weight=float(overrides.get("obstacle_weight", mppi.get("obstacle_weight", 25.0))),
            obstacle_soft_weight=float(
                overrides.get("obstacle_soft_weight", mppi.get("obstacle_soft_weight", 0.0))
            ),
            obstacle_influence_dist=float(
                overrides.get("obstacle_influence_dist", mppi.get("obstacle_influence_dist", 0.0))
            ),
            max_ax=float(overrides.get("max_ax", robot.get("max_ax", 1000.0))),
            max_ay=float(overrides.get("max_ay", robot.get("max_ay", 1000.0))),
            max_awz=float(overrides.get("max_awz", robot.get("max_awz", 1000.0))),
            velocity_lag_beta=float(overrides.get("velocity_lag_beta", robot.get("velocity_lag_beta", 0.0))),
            lateral_weight=float(overrides.get("lateral_weight", mppi.get("lateral_weight", 0.0))),
            yaw_rate_weight=float(overrides.get("yaw_rate_weight", mppi.get("yaw_rate_weight", 0.0))),
            accel_weight=float(overrides.get("accel_weight", mppi.get("accel_weight", 0.0))),
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
        costs = self.trajectory_cost_batch(state, candidates, goal, obstacles)
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
        command = self.nominal_u[0].copy()
        control = self._apply_velocity_response(state, command).astype(np.float32)
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].copy()
        self.previous_control = command.copy()
        self._shift_nominal_controls()
        return control, optimal_u, sample_u, normalizer, min_cost

    def trajectory_cost(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> float:
        states, real_controls = self._rollout_batch(
            initial_state,
            controls[None, :, :],
            return_controls=True,
        )
        states = states[0]
        real_controls = real_controls[0]
        final_state = states[-1]
        xy_error = final_state[:2] - goal[:2]
        yaw_error = self._angle_diff(float(final_state[2]), float(goal[2]))
        goal_cost = self.goal_xy_weight * float(np.dot(xy_error, xy_error))
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * float(np.sum(controls * controls))
        smooth_cost = self.smooth_weight * float(
            np.sum(np.diff(np.vstack([self.previous_control, controls]), axis=0) ** 2)
        )
        accel = np.diff(np.vstack([np.asarray(initial_state, dtype=np.float32)[3:], real_controls]), axis=0) / self.dt
        accel_cost = self.accel_weight * float(np.sum(accel * accel))
        lateral_cost = self.lateral_weight * float(np.sum(real_controls[:, 1] * real_controls[:, 1]))
        yaw_rate_cost = self.yaw_rate_weight * float(np.sum(real_controls[:, 2] * real_controls[:, 2]))
        obstacle_cost = self._obstacle_cost(states[1:], obstacles)
        return goal_cost + yaw_cost + control_cost + smooth_cost + accel_cost + lateral_cost + yaw_rate_cost + obstacle_cost

    def trajectory_cost_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        states, real_controls = self._rollout_batch(initial_state, controls, return_controls=True)
        final_states = states[:, -1, :]
        xy_error = final_states[:, :2] - goal[:2]
        yaw_error = self._angle_diff_array(final_states[:, 2], float(goal[2]))
        goal_cost = self.goal_xy_weight * np.sum(xy_error * xy_error, axis=1)
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * np.sum(controls * controls, axis=(1, 2))
        previous = np.broadcast_to(self.previous_control, (controls.shape[0], 1, 3))
        control_deltas = np.diff(np.concatenate([previous, controls], axis=1), axis=1)
        smooth_cost = self.smooth_weight * np.sum(control_deltas * control_deltas, axis=(1, 2))
        initial_velocity = np.broadcast_to(
            np.asarray(initial_state, dtype=np.float32)[3:],
            (controls.shape[0], 1, 3),
        )
        accel = np.diff(np.concatenate([initial_velocity, real_controls], axis=1), axis=1) / self.dt
        accel_cost = self.accel_weight * np.sum(accel * accel, axis=(1, 2))
        lateral_cost = self.lateral_weight * np.sum(real_controls[:, :, 1] * real_controls[:, :, 1], axis=1)
        yaw_rate_cost = self.yaw_rate_weight * np.sum(real_controls[:, :, 2] * real_controls[:, :, 2], axis=1)
        obstacle_cost = self._obstacle_cost_batch(states[:, 1:, :], obstacles)
        return (
            goal_cost
            + yaw_cost
            + control_cost
            + smooth_cost
            + accel_cost
            + lateral_cost
            + yaw_rate_cost
            + obstacle_cost
        ).astype(np.float32)

    def _rollout_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        return_controls: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        controls = np.clip(np.asarray(controls, dtype=np.float32), -self.max_control, self.max_control)
        num_samples, horizon_steps, _ = controls.shape
        states = np.zeros((num_samples, horizon_steps + 1, 6), dtype=np.float32)
        states[:, 0, :] = np.asarray(initial_state, dtype=np.float32)
        real_controls = np.zeros((num_samples, horizon_steps, 3), dtype=np.float32)
        prev_real = np.broadcast_to(states[:, 0, 3:], (num_samples, 3)).copy()
        max_delta = self.max_accel * self.dt
        for step in range(horizon_steps):
            prev = states[:, step, :]
            command = controls[:, step, :]
            lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
            delta = np.clip(lagged - prev_real, -max_delta, max_delta)
            control = np.clip(prev_real + delta, -self.max_control, self.max_control)
            real_controls[:, step, :] = control
            theta = prev[:, 2]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            vx = control[:, 0]
            vy = control[:, 1]
            wz = control[:, 2]
            states[:, step + 1, 0] = prev[:, 0] + (vx * cos_theta - vy * sin_theta) * self.dt
            states[:, step + 1, 1] = prev[:, 1] + (vx * sin_theta + vy * cos_theta) * self.dt
            states[:, step + 1, 2] = prev[:, 2] + wz * self.dt
            states[:, step + 1, 3] = vx
            states[:, step + 1, 4] = vy
            states[:, step + 1, 5] = wz
            prev_real = control
        if return_controls:
            return states, real_controls
        return states

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
            if self.obstacle_soft_weight > 0.0 and self.obstacle_influence_dist > self.safety_dist:
                far_mask = (clearance > self.safety_dist) & (clearance < self.obstacle_influence_dist)
                if np.any(far_mask):
                    soft_margin = self.obstacle_influence_dist - clearance[far_mask]
                    total += self.obstacle_soft_weight * float(np.sum(soft_margin * soft_margin))
        return total

    def _obstacle_cost_batch(self, states: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        costs = np.zeros(states.shape[0], dtype=np.float32)
        if obstacles.size == 0:
            return costs
        for obstacle in obstacles:
            center = obstacle[:2].astype(np.float32)
            clearance = (
                np.linalg.norm(states[:, :, :2] - center[None, None, :], axis=2)
                - float(obstacle[2])
                - self.robot_radius
            )
            margin = np.maximum(self.safety_dist - clearance, 0.0)
            costs += self.obstacle_weight * np.sum(margin * margin, axis=1)
            if self.obstacle_soft_weight > 0.0 and self.obstacle_influence_dist > self.safety_dist:
                soft_margin = np.maximum(self.obstacle_influence_dist - clearance, 0.0)
                soft_margin = np.where(clearance > self.safety_dist, soft_margin, 0.0)
                costs += self.obstacle_soft_weight * np.sum(soft_margin * soft_margin, axis=1)
        return costs

    def _shift_nominal_controls(self) -> None:
        self.nominal_u[:-1] = self.nominal_u[1:]
        self.nominal_u[-1] = 0.0

    def _apply_velocity_response(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        prev_real = np.asarray(state, dtype=np.float32)[3:]
        command = np.clip(np.asarray(command, dtype=np.float32), -self.max_control, self.max_control)
        lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
        delta = np.clip(lagged - prev_real, -self.max_accel * self.dt, self.max_accel * self.dt)
        return np.clip(prev_real + delta, -self.max_control, self.max_control)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _angle_diff_array(a: np.ndarray, b: float) -> np.ndarray:
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi
