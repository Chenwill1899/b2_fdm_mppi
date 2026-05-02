"""Torch MPPI controller using learned residual FDM rollout dynamics."""

from __future__ import annotations

import numpy as np
import torch

from b2_fdm_mppi.controllers.mppi_omni_torch import MppiOmniTorch
from b2_fdm_mppi.core.learned_residual_dynamics import LearnedResidualDynamics
from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.terrain import TerrainField


class LearnedFdmMppiOmniTorch(MppiOmniTorch):
    """Learned-FDM MPPI backend sharing Torch rollout and cost with nominal Torch."""

    def __init__(
        self,
        *args,
        learned_dynamics: LearnedResidualDynamics,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        residual_gain: float = 1.0,
        profile_enabled: bool = False,
        **kwargs,
    ) -> None:
        shared_terrain = terrain if terrain is not None else getattr(learned_dynamics, "terrain", TerrainField())
        super().__init__(*args, terrain=shared_terrain, device=device, profile_enabled=profile_enabled, **kwargs)
        self.learned_dynamics = learned_dynamics
        self.residual_gain = float(residual_gain)
        self._setup_learned_tensors()

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
        device = str(fdm.get("device", mppi.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
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
            terrain_risk_weight=float(overrides.get("terrain_risk_weight", mppi.get("terrain_risk_weight", 0.0))),
            terrain_risk_power=float(overrides.get("terrain_risk_power", mppi.get("terrain_risk_power", 2.0))),
            terrain_risk_threshold=float(
                overrides.get("terrain_risk_threshold", mppi.get("terrain_risk_threshold", 0.0))
            ),
            terrain_risk_mode=str(overrides.get("terrain_risk_mode", mppi.get("terrain_risk_mode", "excess"))),
            robot_radius=float(robot["radius"]),
            safety_dist=float(robot["safety_dist"]),
            draw_num_traj=int(mppi["draw_num_traj"]),
            seed=seed,
            learned_dynamics=learned_dynamics,
            terrain=terrain,
            device=device,
            residual_gain=float(overrides.get("residual_gain", fdm.get("residual_gain", 1.0))),
            profile_enabled=bool(overrides.get("profile_enabled", fdm.get("profile_enabled", False))),
        )

    def _predict_residual_torch(self, states: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        custom_predictor = getattr(self.learned_dynamics, "predict_residual_torch", None)
        if custom_predictor is not None:
            profile_start = self._profile_start()
            residual = custom_predictor(states, commands).to(dtype=torch.float32, device=self.torch_device)
            self._profile_stop("fdm_inference_ms", profile_start)
            return self.residual_gain * residual
        profile_start = self._profile_start()
        features, risks = self._terrain_features_torch(states)
        self._profile_stop("terrain_features_ms", profile_start)
        profile_start = self._profile_start()
        fdm_features = torch.cat([states, commands, features, risks[:, None]], dim=1)
        standardized = (fdm_features - self.feature_mean_t) / self.feature_std_t
        pred = self.learned_dynamics.model(standardized)
        residual = pred * self.target_std_t + self.target_mean_t
        self._profile_stop("fdm_inference_ms", profile_start)
        return self.residual_gain * residual

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
