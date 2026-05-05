"""Torch MPPI controller using Sequence FDM V2 as a risk-aware cost evaluator.

This controller does NOT replace the nominal dynamics rollout. Instead:
1. MPPI still uses the real nominal dynamics (MppiOmniTorch._rollout_batch_torch)
   to compute accurate state trajectories for goal/obstacle/terrain costs.
2. The V2 model provides an ADDITIONAL risk cost term based on its learned
   prediction of terrain risk along the trajectory.
3. The final cost = base_cost (from real dynamics) + fdm_risk_cost (from V2).

This avoids the compounding error problem because the trajectory positions
used for obstacle/goal costs come from the accurate real dynamics, while the
V2 model only influences trajectory selection via its risk assessment.
"""

from __future__ import annotations

import numpy as np
import torch

from b2_fdm_mppi.controllers.mppi_omni_torch import MppiOmniTorch
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.core.terrain_grid import sample_terrain_risk_grid_torch


class MppiOmniSequenceFdmV2Torch(MppiOmniTorch):
    """MPPI controller that adds Sequence FDM V2 risk cost on top of real dynamics."""

    def __init__(
        self,
        *args,
        sequence_dynamics: SequenceFdmDynamics,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        fdm_risk_weight: float = 10.0,
        profile_enabled: bool = False,
        **kwargs,
    ) -> None:
        shared_terrain = terrain if terrain is not None else getattr(sequence_dynamics, "terrain", TerrainField())
        super().__init__(*args, terrain=shared_terrain, device=device, profile_enabled=profile_enabled, **kwargs)
        self.sequence_dynamics = sequence_dynamics
        self.fdm_risk_weight = float(fdm_risk_weight)
        # Ensure model is on correct device
        self.sequence_dynamics.model.to(self.torch_device)

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        **overrides,
    ) -> "MppiOmniSequenceFdmV2Torch":
        """Create controller from config, auto-adjusting horizon to match V2 model."""
        sequence_dynamics = overrides.pop("sequence_dynamics")
        device = overrides.pop("device", "cuda")
        fdm_risk_weight = overrides.pop("fdm_risk_weight", 10.0)
        profile_enabled = overrides.pop("profile_enabled", False)

        # Adjust config time_horizon to match V2 model's horizon_steps
        adj_config = dict(config)
        sim = dict(adj_config["simulation"])
        h_v2 = int(sequence_dynamics.horizon_steps)
        sim["time_horizon"] = h_v2 / float(sim["sampling_rate"])
        adj_config["simulation"] = sim

        # Build via parent and reclass
        instance = MppiOmniTorch.from_config(adj_config, seed=seed, **overrides)
        instance.__class__ = cls
        instance.sequence_dynamics = sequence_dynamics
        instance.fdm_risk_weight = fdm_risk_weight
        instance.sequence_dynamics.model.to(instance.torch_device)
        return instance

    def _trajectory_cost_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        path: torch.Tensor | None = None,
        costmap: dict | None = None,
    ) -> torch.Tensor:
        """Compute cost using REAL dynamics rollout + V2 risk prediction.

        Steps:
        1. Run nominal dynamics rollout (accurate positions for obstacle/goal)
        2. Compute base cost (goal, obstacle, smoothness, etc.)
        3. Run V2 model to predict per-step risk logits
        4. Add risk cost as an extra term
        5. Return total cost
        """
        controls = torch.clamp(controls, -self.max_control_t, self.max_control_t)
        num_samples = int(controls.shape[0])
        H = self.horizon_steps

        # 1. Base cost from real dynamics rollout (inherited from MppiOmniTorch)
        base_cost = super()._trajectory_cost_batch_torch(initial_state, controls, goal, obstacles)

        # 2. V2 risk prediction (additional cost term only)
        profile_start = self._profile_start()
        x0 = float(initial_state[0])
        y0 = float(initial_state[1])
        terrain_grid = sample_terrain_risk_grid_torch(self.terrain, x0, y0, size=9, span=18.0)
        terrain_grid = terrain_grid.unsqueeze(0).expand(num_samples, -1)

        state_t = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32).reshape(1, 6),
            dtype=torch.float32,
            device=self.torch_device,
        ).expand(num_samples, -1)

        with torch.no_grad():
            _pred_states, pred_risk_logits = self.sequence_dynamics.predict_torch(state_t, controls, terrain_grid)

        pred_risk = torch.sigmoid(pred_risk_logits)
        fdm_risk_cost = self.fdm_risk_weight * torch.sum(pred_risk, dim=1)
        self._profile_stop("fdm_inference_ms", profile_start)

        return (base_cost + fdm_risk_cost).to(torch.float32)
