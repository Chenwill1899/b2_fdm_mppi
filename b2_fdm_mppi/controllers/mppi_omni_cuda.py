"""PyCUDA MPPI controller for the B2 omnidirectional SE(2) nominal model."""

from __future__ import annotations

import numpy as np

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray

from b2_fdm_mppi.core.omni_b2 import OmniB2


CUDA_SOURCE = r"""
__device__ float clamp_value(float value, float limit) {
    if (value > limit) {
        return limit;
    }
    if (value < -limit) {
        return -limit;
    }
    return value;
}

__device__ float angle_diff(float a, float b) {
    const float pi = 3.14159265358979323846f;
    return fmodf(a - b + pi, 2.0f * pi) - pi;
}

extern "C" __global__ void omni_costs(
    const float *initial_state,
    const float *controls,
    const float *previous_control,
    const float *goal,
    const float *obstacles,
    float *costs,
    int num_samples,
    int horizon_steps,
    int obstacle_count,
    float dt,
    float max_vx,
    float max_vy,
    float max_wz,
    float goal_xy_weight,
    float yaw_weight,
    float control_weight,
    float smooth_weight,
    float obstacle_weight,
    float cbf_weight,
    float cbf_alpha,
    float robot_radius,
    float safety_dist
) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= num_samples) {
        return;
    }

    float x = initial_state[0];
    float y = initial_state[1];
    float theta = initial_state[2];
    float prev_u0 = previous_control[0];
    float prev_u1 = previous_control[1];
    float prev_u2 = previous_control[2];
    float cost = 0.0f;

    for (int step = 0; step < horizon_steps; ++step) {
        int offset = (sample * horizon_steps + step) * 3;
        float vx = clamp_value(controls[offset], max_vx);
        float vy = clamp_value(controls[offset + 1], max_vy);
        float wz = clamp_value(controls[offset + 2], max_wz);

        cost += control_weight * (vx * vx + vy * vy + wz * wz);
        float du0 = vx - prev_u0;
        float du1 = vy - prev_u1;
        float du2 = wz - prev_u2;
        cost += smooth_weight * (du0 * du0 + du1 * du1 + du2 * du2);

        float old_x = x;
        float old_y = y;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        x += (vx * cos_theta - vy * sin_theta) * dt;
        y += (vx * sin_theta + vy * cos_theta) * dt;
        theta += wz * dt;

        for (int obs_index = 0; obs_index < obstacle_count; ++obs_index) {
            int obs_offset = obs_index * 7;
            float ox = obstacles[obs_offset];
            float oy = obstacles[obs_offset + 1];
            float radius = obstacles[obs_offset + 2];

            float dx = x - ox;
            float dy = y - oy;
            float clearance = sqrtf(dx * dx + dy * dy) - radius - robot_radius;
            float margin = safety_dist - clearance;
            if (margin > 0.0f) {
                cost += obstacle_weight * margin * margin;
            }

            if (cbf_weight > 0.0f) {
                float old_dx = old_x - ox;
                float old_dy = old_y - oy;
                float h = sqrtf(old_dx * old_dx + old_dy * old_dy) - radius - robot_radius - safety_dist;
                float h_next = clearance - safety_dist;
                float cbf_violation = -(h_next - h + cbf_alpha * h);
                if (cbf_violation > 0.0f) {
                    cost += cbf_weight * cbf_violation * cbf_violation;
                }
            }
        }

        prev_u0 = vx;
        prev_u1 = vy;
        prev_u2 = wz;
    }

    float xy0 = x - goal[0];
    float xy1 = y - goal[1];
    float yaw_error = angle_diff(theta, goal[2]);
    cost += goal_xy_weight * (xy0 * xy0 + xy1 * xy1);
    cost += yaw_weight * yaw_error * yaw_error;
    costs[sample] = cost;
}
"""


class MppiOmniCuda:
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
        cbf_weight: float = 0.0,
        cbf_alpha: float = 0.1,
        robot_radius: float = 0.6,
        safety_dist: float = 0.3,
        draw_num_traj: int = 50,
        seed: int | None = None,
        block_dim: int = 128,
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
        self.cbf_weight = float(cbf_weight)
        self.cbf_alpha = float(cbf_alpha)
        self.robot_radius = float(robot_radius)
        self.safety_dist = float(safety_dist)
        self.draw_num_traj = min(int(draw_num_traj), self.num_samples)
        self.block_dim = int(block_dim)
        self.rng = np.random.default_rng(seed)
        self.nominal_u = np.zeros((self.horizon_steps, 3), dtype=np.float32)
        self.previous_control = np.zeros(3, dtype=np.float32)
        self.model = OmniB2(self.dt, max_vx, max_vy, max_wz)
        self.module = SourceModule(CUDA_SOURCE, no_extern_c=True)
        self.cost_kernel = self.module.get_function("omni_costs")

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        **overrides,
    ) -> "MppiOmniCuda":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        cbf = config["cbf"]
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
            cbf_weight=float(overrides.get("cbf_weight", mppi.get("cbf_weight", 0.0))),
            cbf_alpha=float(overrides.get("cbf_alpha", cbf.get("dcbf_alpha", 0.1))),
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
        ).astype(np.float32)
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
        control = self.nominal_u[0].copy()
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].copy()
        self.previous_control = control.copy()
        self._shift_nominal_controls()
        return control, optimal_u, sample_u, normalizer, min_cost

    def trajectory_cost_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        controls = np.clip(np.asarray(controls, dtype=np.float32), -self.max_control, self.max_control)
        controls = np.ascontiguousarray(controls)
        num_samples = int(controls.shape[0])
        obstacles = np.ascontiguousarray(np.asarray(obstacles, dtype=np.float32).reshape(-1, 7))
        costs_gpu = gpuarray.empty(num_samples, dtype=np.float32)
        grid_dim = ((num_samples + self.block_dim - 1) // self.block_dim, 1, 1)
        self.cost_kernel(
            cuda.In(np.asarray(initial_state, dtype=np.float32)),
            cuda.In(controls),
            cuda.In(np.asarray(self.previous_control, dtype=np.float32)),
            cuda.In(np.asarray(goal, dtype=np.float32)),
            cuda.In(obstacles),
            costs_gpu,
            np.int32(num_samples),
            np.int32(self.horizon_steps),
            np.int32(len(obstacles)),
            np.float32(self.dt),
            np.float32(self.max_control[0]),
            np.float32(self.max_control[1]),
            np.float32(self.max_control[2]),
            np.float32(self.goal_xy_weight),
            np.float32(self.yaw_weight),
            np.float32(self.control_weight),
            np.float32(self.smooth_weight),
            np.float32(self.obstacle_weight),
            np.float32(self.cbf_weight),
            np.float32(self.cbf_alpha),
            np.float32(self.robot_radius),
            np.float32(self.safety_dist),
            block=(self.block_dim, 1, 1),
            grid=grid_dim,
        )
        return costs_gpu.get()

    def _shift_nominal_controls(self) -> None:
        self.nominal_u[:-1] = self.nominal_u[1:]
        self.nominal_u[-1] = 0.0
