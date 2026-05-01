"""Torch batched MPPI controller using learned residual FDM rollout dynamics."""

from __future__ import annotations

import numpy as np
import torch

from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from b2_fdm_mppi.core.learned_residual_dynamics import LearnedResidualDynamics
from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.terrain import TerrainField


class LearnedFdmMppiOmniTorch(MppiOmniNumpy):
    """Reference learned-FDM MPPI backend with batched Torch rollout/cost.

    This keeps the MPPI control API compatible with the NumPy controller while
    moving the expensive sampled rollout to Torch tensors. Use `device="cuda"`
    for the Stage 5-B benchmark path.
    """

    def __init__(
        self,
        *args,
        learned_dynamics: LearnedResidualDynamics,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.learned_dynamics = learned_dynamics
        self.terrain = terrain if terrain is not None else getattr(learned_dynamics, "terrain", TerrainField())
        self.torch_device = torch.device(device)
        if self.torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Learned FDM Torch backend requested CUDA, but torch.cuda.is_available() is false")
        self.max_control_t = torch.as_tensor(self.max_control, dtype=torch.float32, device=self.torch_device)
        self.max_accel_t = torch.as_tensor(self.max_accel, dtype=torch.float32, device=self.torch_device)
        self.noise_std_t = torch.as_tensor(self.noise_std, dtype=torch.float32, device=self.torch_device)
        self.previous_control_t = torch.zeros(3, dtype=torch.float32, device=self.torch_device)
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(int(kwargs.get("seed", 0) or 0))
        self._setup_learned_tensors()
        self._setup_terrain_tensors()

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        *,
        learned_dynamics: LearnedResidualDynamics | None = None,
        **overrides,
    ) -> "LearnedFdmMppiOmniTorch":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        fdm = config.get("fdm", {})
        device = str(fdm.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        terrain = TerrainField.from_config(config.get("terrain"))
        if learned_dynamics is None:
            dynamics_robot = OmniB2(
                dt,
                float(robot["max_vx"]),
                float(robot["max_vy"]),
                float(robot["max_wz"]),
            )
            learned_dynamics = LearnedResidualDynamics.from_artifacts(
                fdm["model_dir"],
                robot=dynamics_robot,
                terrain=terrain,
                checkpoint=fdm.get("checkpoint", "best_model.pt"),
                normalization=fdm.get("normalization", "normalization.npz"),
                device=device,
            )
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
            jerk_weight=float(overrides.get("jerk_weight", mppi.get("jerk_weight", 0.0))),
            robot_radius=float(robot["radius"]),
            safety_dist=float(robot["safety_dist"]),
            draw_num_traj=int(mppi["draw_num_traj"]),
            seed=seed,
            learned_dynamics=learned_dynamics,
            terrain=terrain,
            device=device,
        )

    def compute_control(self, state: np.ndarray, cost_params):
        goal = torch.as_tensor(np.asarray(cost_params[3], dtype=np.float32), device=self.torch_device)
        obstacles = torch.as_tensor(
            np.asarray(cost_params[4], dtype=np.float32).reshape(-1, 7),
            device=self.torch_device,
        )
        nominal = torch.as_tensor(self.nominal_u, dtype=torch.float32, device=self.torch_device)
        noise = torch.randn(
            (self.num_samples, self.horizon_steps, 3),
            dtype=torch.float32,
            device=self.torch_device,
            generator=self.generator,
        ) * self.noise_std_t
        candidates = torch.clamp(nominal.unsqueeze(0) + noise, -self.max_control_t, self.max_control_t)
        costs = self._trajectory_cost_batch_torch(state, candidates, goal, obstacles)
        min_cost_t = torch.min(costs)
        weights = torch.exp(-(costs - min_cost_t) / max(self.lambda_, 1e-6))
        normalizer_t = torch.sum(weights)
        if not bool(torch.isfinite(normalizer_t).item()) or float(normalizer_t.detach().cpu()) <= 0.0:
            weights = torch.full((self.num_samples,), 1.0 / self.num_samples, dtype=torch.float32, device=self.torch_device)
            normalizer_t = torch.tensor(1.0, dtype=torch.float32, device=self.torch_device)
        else:
            weights = weights / normalizer_t
        nominal = torch.sum(weights[:, None, None] * candidates, dim=0)
        nominal = torch.clamp(nominal, -self.max_control_t, self.max_control_t)
        self.nominal_u = nominal.detach().cpu().numpy().astype(np.float32)
        command = self.nominal_u[0].copy()
        control = self._apply_velocity_response(state, command).astype(np.float32)
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].detach().cpu().numpy().astype(np.float32)
        self.previous_control = command.copy()
        self.previous_control_t = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device)
        self._shift_nominal_controls()
        return (
            control,
            optimal_u,
            sample_u,
            float(normalizer_t.detach().cpu()),
            float(min_cost_t.detach().cpu()),
        )

    def trajectory_cost_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        controls_t = torch.as_tensor(np.asarray(controls, dtype=np.float32), device=self.torch_device)
        controls_t = torch.clamp(controls_t, -self.max_control_t, self.max_control_t)
        goal_t = torch.as_tensor(np.asarray(goal, dtype=np.float32), device=self.torch_device)
        obstacles_t = torch.as_tensor(np.asarray(obstacles, dtype=np.float32).reshape(-1, 7), device=self.torch_device)
        costs = self._trajectory_cost_batch_torch(initial_state, controls_t, goal_t, obstacles_t)
        return costs.detach().cpu().numpy().astype(np.float32)

    def _rollout_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        return_controls: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        controls_t = torch.as_tensor(np.asarray(controls, dtype=np.float32), device=self.torch_device)
        states_t, real_controls_t = self._rollout_batch_torch(initial_state, controls_t)
        states = states_t.detach().cpu().numpy().astype(np.float32)
        real_controls = real_controls_t.detach().cpu().numpy().astype(np.float32)
        if return_controls:
            return states, real_controls
        return states

    def _trajectory_cost_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
    ) -> torch.Tensor:
        controls = torch.clamp(controls, -self.max_control_t, self.max_control_t)
        states, real_controls = self._rollout_batch_torch(initial_state, controls)
        final_states = states[:, -1, :]
        xy_error = final_states[:, :2] - goal[:2]
        yaw_error = self._angle_diff_torch(final_states[:, 2], goal[2])
        goal_cost = self.goal_xy_weight * torch.sum(xy_error * xy_error, dim=1)
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * torch.sum(controls * controls, dim=(1, 2))
        previous = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device).view(1, 1, 3)
        previous = previous.expand(controls.shape[0], 1, 3)
        control_deltas = torch.diff(torch.cat([previous, controls], dim=1), dim=1)
        smooth_cost = self.smooth_weight * torch.sum(control_deltas * control_deltas, dim=(1, 2))
        initial_velocity = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32)[3:],
            dtype=torch.float32,
            device=self.torch_device,
        ).view(1, 1, 3)
        initial_velocity = initial_velocity.expand(controls.shape[0], 1, 3)
        accel = torch.diff(torch.cat([initial_velocity, real_controls], dim=1), dim=1) / self.dt
        accel_cost = self.accel_weight * torch.sum(accel * accel, dim=(1, 2))
        jerk = torch.diff(torch.cat([initial_velocity, real_controls], dim=1), n=2, dim=1)
        jerk_cost = self.jerk_weight * torch.sum(jerk * jerk, dim=(1, 2))
        lateral_cost = self.lateral_weight * torch.sum(real_controls[:, :, 1] * real_controls[:, :, 1], dim=1)
        yaw_rate_cost = self.yaw_rate_weight * torch.sum(real_controls[:, :, 2] * real_controls[:, :, 2], dim=1)
        obstacle_cost = self._obstacle_cost_batch_torch(states[:, 1:, :], obstacles)
        return (
            goal_cost
            + yaw_cost
            + control_cost
            + smooth_cost
            + accel_cost
            + jerk_cost
            + lateral_cost
            + yaw_rate_cost
            + obstacle_cost
        ).to(torch.float32)

    def _rollout_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        controls = torch.clamp(controls.to(torch.float32), -self.max_control_t, self.max_control_t)
        num_samples, horizon_steps, _ = controls.shape
        states = torch.zeros((num_samples, horizon_steps + 1, 6), dtype=torch.float32, device=self.torch_device)
        states[:, 0, :] = torch.as_tensor(np.asarray(initial_state, dtype=np.float32), device=self.torch_device)
        real_controls = torch.zeros((num_samples, horizon_steps, 3), dtype=torch.float32, device=self.torch_device)
        prev_real = states[:, 0, 3:].clone()
        max_delta = self.max_accel_t * self.dt
        with torch.no_grad():
            for step in range(horizon_steps):
                prev = states[:, step, :]
                command = controls[:, step, :]
                lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
                delta = torch.clamp(lagged - prev_real, -max_delta, max_delta)
                response_command = torch.clamp(prev_real + delta, -self.max_control_t, self.max_control_t)
                residual = self._predict_residual_torch(prev, response_command)
                control = torch.clamp(response_command + residual, -self.max_control_t, self.max_control_t)
                real_controls[:, step, :] = control
                theta = prev[:, 2]
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
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
        return states, real_controls

    def _predict_residual_torch(self, states: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        custom_predictor = getattr(self.learned_dynamics, "predict_residual_torch", None)
        if custom_predictor is not None:
            return custom_predictor(states, commands).to(dtype=torch.float32, device=self.torch_device)
        features, risks = self._terrain_features_torch(states)
        fdm_features = torch.cat([states, commands, features, risks[:, None]], dim=1)
        standardized = (fdm_features - self.feature_mean_t) / self.feature_std_t
        pred = self.learned_dynamics.model(standardized)
        return pred * self.target_std_t + self.target_mean_t

    def _obstacle_cost_batch_torch(self, states: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        costs = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
        if obstacles.numel() == 0:
            return costs
        centers = obstacles[:, :2].to(torch.float32)
        radii = obstacles[:, 2].to(torch.float32)
        deltas = states[:, :, None, :2] - centers[None, None, :, :]
        clearance = torch.linalg.norm(deltas, dim=3) - radii[None, None, :] - self.robot_radius
        margin = torch.clamp(self.safety_dist - clearance, min=0.0)
        costs += self.obstacle_weight * torch.sum(margin * margin, dim=(1, 2))
        if self.obstacle_soft_weight > 0.0 and self.obstacle_influence_dist > self.safety_dist:
            soft_margin = torch.clamp(self.obstacle_influence_dist - clearance, min=0.0)
            soft_margin = torch.where(clearance > self.safety_dist, soft_margin, torch.zeros_like(soft_margin))
            costs += self.obstacle_soft_weight * torch.sum(soft_margin * soft_margin, dim=(1, 2))
        return costs

    def _terrain_features_torch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = states[:, 0]
        y = states[:, 1]
        if not self.terrain.enabled:
            features = torch.zeros((states.shape[0], 4), dtype=torch.float32, device=self.torch_device)
            risks = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
            return features, risks
        slope_f = self.terrain.slope_scale * (torch.sin(self.terrain.slope_wave * x) + 0.35 * torch.cos(0.5 * y))
        slope_l = self.terrain.slope_scale * (torch.cos(self.terrain.slope_wave * y) + 0.35 * torch.sin(0.5 * x))
        roughness = self.terrain.roughness_scale * (0.5 + 0.5 * torch.sin(self.terrain.roughness_wave * (x + y)))
        friction = self.terrain.friction_base - self.terrain.friction_slope_scale * (
            0.5 * (torch.abs(slope_f) + torch.abs(slope_l))
        )
        friction = friction - self.terrain.friction_roughness_scale * roughness
        if self.noise_grid_t is not None:
            noise = self._bilinear_sample_torch(self.noise_grid_t, x, y)
            grad_x = self._bilinear_sample_torch(self.noise_grad_x_t, x, y)
            grad_y = self._bilinear_sample_torch(self.noise_grad_y_t, x, y)
            roughness = roughness + self.terrain.noise_roughness_weight * noise
            friction = friction - self.terrain.noise_friction_weight * noise
            slope_f = slope_f + self.terrain.noise_slope_weight * grad_x
            slope_l = slope_l + self.terrain.noise_slope_weight * grad_y
        roughness = torch.clamp(roughness, 0.0, 1.0)
        friction = torch.clamp(friction, 0.2, 1.0)
        attenuation = self._goal_relief_attenuation_torch(x, y)
        slope_f = slope_f * attenuation
        slope_l = slope_l * attenuation
        roughness = roughness * attenuation
        friction = 1.0 - attenuation * (1.0 - friction)
        features = torch.stack([slope_f, slope_l, roughness, friction], dim=1).to(torch.float32)
        w0, w1, w2, w3 = self.terrain.risk_weights
        risks = w0 * torch.abs(slope_f) + w1 * torch.abs(slope_l) + w2 * roughness + w3 * (1.0 - friction)
        return features, risks.to(torch.float32)

    def _bilinear_sample_torch(self, grid: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_min, x_max = self.terrain.noise_x_range
        y_min, y_max = self.terrain.noise_y_range
        cols = int(grid.shape[1])
        rows = int(grid.shape[0])
        gx = torch.clamp((x - x_min) / max(1e-6, x_max - x_min), 0.0, 1.0) * (cols - 1)
        gy = torch.clamp((y - y_min) / max(1e-6, y_max - y_min), 0.0, 1.0) * (rows - 1)
        x0 = torch.floor(gx).to(torch.long)
        y0 = torch.floor(gy).to(torch.long)
        x1 = torch.clamp(x0 + 1, max=cols - 1)
        y1 = torch.clamp(y0 + 1, max=rows - 1)
        tx = (gx - x0.to(torch.float32)).to(torch.float32)
        ty = (gy - y0.to(torch.float32)).to(torch.float32)
        top = (1.0 - tx) * grid[y0, x0] + tx * grid[y0, x1]
        bottom = (1.0 - tx) * grid[y1, x0] + tx * grid[y1, x1]
        return ((1.0 - ty) * top + ty * bottom).to(torch.float32)

    def _goal_relief_attenuation_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        relief = self.terrain.goal_relief
        if not bool(relief.get("enabled", False)):
            return torch.ones_like(x, dtype=torch.float32, device=self.torch_device)
        center = relief.get("center", [0.0, 0.0])
        sigma = relief.get("sigma", [2.0, 1.2])
        sigma_x = max(1e-6, float(sigma[0]))
        sigma_y = max(1e-6, float(sigma[1]))
        strength = float(np.clip(relief.get("strength", 0.75), 0.0, 1.0))
        floor = float(np.clip(relief.get("floor", 0.25), 0.0, 1.0))
        dx = (x - float(center[0])) / sigma_x
        dy = (y - float(center[1])) / sigma_y
        gaussian = torch.exp(-0.5 * (dx * dx + dy * dy))
        return torch.clamp(1.0 - strength * gaussian, min=floor)

    def _setup_learned_tensors(self) -> None:
        model = getattr(self.learned_dynamics, "model", None)
        if model is not None:
            model.to(self.torch_device)
            model.eval()
        if hasattr(self.learned_dynamics, "device"):
            self.learned_dynamics.device = self.torch_device
        if hasattr(self.learned_dynamics, "feature_mean"):
            self.feature_mean_t = torch.as_tensor(
                self.learned_dynamics.feature_mean,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.feature_std_t = torch.as_tensor(
                self.learned_dynamics.feature_std,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.target_mean_t = torch.as_tensor(
                self.learned_dynamics.target_mean,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.target_std_t = torch.as_tensor(
                self.learned_dynamics.target_std,
                dtype=torch.float32,
                device=self.torch_device,
            )

    def _setup_terrain_tensors(self) -> None:
        if getattr(self.terrain, "_noise_grid", None) is None:
            self.noise_grid_t = None
            self.noise_grad_x_t = None
            self.noise_grad_y_t = None
            return
        self.noise_grid_t = torch.as_tensor(self.terrain._noise_grid, dtype=torch.float32, device=self.torch_device)
        self.noise_grad_x_t = torch.as_tensor(
            self.terrain._noise_grad_x,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.noise_grad_y_t = torch.as_tensor(
            self.terrain._noise_grad_y,
            dtype=torch.float32,
            device=self.torch_device,
        )

    @staticmethod
    def _angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + torch.pi, 2.0 * torch.pi) - torch.pi
