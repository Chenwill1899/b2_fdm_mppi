"""MPPI integration adapter for the High-Level FDM.

This module is the contract surface that MPPI consumes. It intentionally
stays small and opinionated:

    HighLevelFdmRollout
        Wraps a trained HighLevelFdm. Batched forward prediction of
        trajectories and risks for one `(history, map)` state and a batch
        of candidate control sequences.

    high_level_fdm_cost
        Converts (risk_prob, pose_pred, goal_xy_body) into a scalar cost
        per MPPI sample. The MPPI outer loop adds this to its existing
        cost terms. As the High-Level FDM becomes more accurate, the
        existing handwritten costs (local obstacle, lateral velocity,
        smoothness, etc.) can be down-weighted.

The MPPI side is *not* modified in this PR. Keeping the contract here
means the rollout model and cost term can be unit-tested without touching
the cube demo's launch graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from b2_fdm_mppi.high_level_fdm.model import HighLevelFdm
from b2_fdm_mppi.high_level_fdm.schema import (
    CONTROL_DIM,
    POSE_PARAM_DIM,
    STATE_DIM,
    HighLevelFdmSchema,
    pose_param_to_delta,
)


# ---------------------------------------------------------------------------
# Rollout wrapper
# ---------------------------------------------------------------------------

@dataclass
class HighLevelFdmRolloutResult:
    """Batched rollout prediction for one (history, map) and M control sequences."""

    pose_param: torch.Tensor  # (M, N, 4)  [dx, dy, sin, cos]
    pose_delta: torch.Tensor  # (M, N, 3)  [dx, dy, dtheta]
    risk_prob: torch.Tensor   # (M, N, K)  in [0, 1]


class HighLevelFdmRollout:
    """Batched High-Level FDM rollout.

    The model expects a *per-sample* history and map but MPPI usually runs
    M candidate control sequences at a single planning step. We therefore
    broadcast the history and map across M and run one batched forward
    pass per call.
    """

    def __init__(
        self,
        model: HighLevelFdm,
        *,
        schema: HighLevelFdmSchema,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.schema = schema
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        schema: HighLevelFdmSchema | None = None,
    ) -> "HighLevelFdmRollout":
        payload = torch.load(str(checkpoint_path), map_location=device)
        model = HighLevelFdm.from_checkpoint_payload(payload)
        if schema is None:
            schema = HighLevelFdmSchema(
                history_length=int(payload["config"]["history_length"]),
                horizon=int(payload["config"]["horizon"]),
                map_channels=int(payload["config"]["map_channels"]),
                map_height=int(payload["config"]["map_height"]),
                map_width=int(payload["config"]["map_width"]),
                map_resolution=float(payload.get("map_resolution", 0.1)),
            )
        return cls(model=model, schema=schema, device=device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        state_history: torch.Tensor | np.ndarray,
        map_patch: torch.Tensor | np.ndarray,
        control_sequences: torch.Tensor | np.ndarray,
    ) -> HighLevelFdmRolloutResult:
        """Predict `(pose_param, risk_prob)` for M control sequences.

        Args:
            state_history   (H, STATE_DIM) or (1, H, STATE_DIM)
            map_patch       (C, Gh, Gw) or (1, C, Gh, Gw)
            control_sequences (M, N, CONTROL_DIM)

        Returns:
            HighLevelFdmRolloutResult with shape prefix M on pose/risk tensors.
        """
        history_t = self._prepare_history(state_history)
        map_t = self._prepare_map(map_patch)
        controls_t = self._prepare_controls(control_sequences)
        num_samples = int(controls_t.shape[0])
        history_batched = history_t.expand(num_samples, -1, -1).contiguous()
        map_batched = map_t.expand(num_samples, -1, -1, -1).contiguous()

        with torch.no_grad():
            pose_param, risk_logits = self.model(
                history_batched, map_batched, controls_t
            )
            risk_prob = torch.sigmoid(risk_logits)

        pose_delta = self._pose_param_to_delta_tensor(pose_param)
        return HighLevelFdmRolloutResult(
            pose_param=pose_param,
            pose_delta=pose_delta,
            risk_prob=risk_prob,
        )

    # ------------------------------------------------------------------
    # Preparation helpers
    # ------------------------------------------------------------------

    def _prepare_history(
        self, state_history: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        tensor = torch.as_tensor(state_history, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != STATE_DIM:
            raise ValueError(
                f"state_history must be (H, {STATE_DIM}) or (1, H, {STATE_DIM}); "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != self.schema.history_length:
            raise ValueError(
                f"history length {tensor.shape[1]} != schema {self.schema.history_length}"
            )
        return tensor

    def _prepare_map(self, map_patch: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(map_patch, dtype=torch.float32, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                f"map_patch must be (C, Gh, Gw) or (1, C, Gh, Gw); got {tuple(tensor.shape)}"
            )
        expected_channels = self.schema.map_channels
        if tensor.shape[1] != expected_channels:
            raise ValueError(
                f"map_patch channels {tensor.shape[1]} != schema {expected_channels}"
            )
        return tensor

    def _prepare_controls(
        self, control_sequences: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            control_sequences, dtype=torch.float32, device=self.device
        )
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != CONTROL_DIM:
            raise ValueError(
                f"control_sequences must be (M, N, {CONTROL_DIM}); got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != self.schema.horizon:
            raise ValueError(
                f"horizon {tensor.shape[1]} != schema {self.schema.horizon}"
            )
        return tensor

    @staticmethod
    def _pose_param_to_delta_tensor(pose_param: torch.Tensor) -> torch.Tensor:
        if pose_param.shape[-1] != POSE_PARAM_DIM:
            raise ValueError(
                f"pose_param trailing dim must be {POSE_PARAM_DIM}; got {pose_param.shape}"
            )
        dx = pose_param[..., 0]
        dy = pose_param[..., 1]
        sin_dth = pose_param[..., 2]
        cos_dth = pose_param[..., 3]
        norm = torch.sqrt(sin_dth * sin_dth + cos_dth * cos_dth).clamp_min(1e-8)
        dtheta = torch.atan2(sin_dth / norm, cos_dth / norm)
        return torch.stack([dx, dy, dtheta], dim=-1)


# ---------------------------------------------------------------------------
# MPPI cost term
# ---------------------------------------------------------------------------

def high_level_fdm_cost(
    *,
    risk_prob: torch.Tensor,           # (M, N, K)
    pose_delta: torch.Tensor,          # (M, N, 3)
    risk_channel_weights: torch.Tensor | tuple[float, ...] | list[float],
    goal_xy_body: torch.Tensor | tuple[float, float] | list[float] | None = None,
    goal_weight: float = 0.0,
    step_weight_decay: float | None = None,
) -> torch.Tensor:
    """Additive MPPI cost term from High-Level FDM outputs.

    The cost is a weighted sum of two components:

    1. Risk: `sum_{n, k} w_k * r_{n, k}` over the horizon.
    2. Goal progress: `goal_weight * ||pose_t+N - goal_xy_body||_2` on the
       final-step predicted position in the body frame of t.

    MPPI's outer loop sums this with its own handwritten costs. Returning
    just `(M,)` keeps the caller in control of how to blend this with the
    existing cost terms.
    """
    if risk_prob.ndim != 3:
        raise ValueError(f"risk_prob must be (M, N, K); got {tuple(risk_prob.shape)}")
    if pose_delta.ndim != 3 or pose_delta.shape[-1] != 3:
        raise ValueError(f"pose_delta must be (M, N, 3); got {tuple(pose_delta.shape)}")

    num_samples, horizon, num_channels = risk_prob.shape
    if pose_delta.shape[0] != num_samples or pose_delta.shape[1] != horizon:
        raise ValueError("risk_prob and pose_delta must have matching (M, N) shape")

    weights = torch.as_tensor(
        list(risk_channel_weights), dtype=risk_prob.dtype, device=risk_prob.device
    )
    if weights.ndim != 1 or weights.numel() != num_channels:
        raise ValueError(
            f"risk_channel_weights must have length {num_channels}; got {weights.shape}"
        )

    if step_weight_decay is None:
        step_weights = torch.ones(
            horizon, dtype=risk_prob.dtype, device=risk_prob.device
        )
    else:
        decay = float(step_weight_decay)
        step_weights = torch.tensor(
            [decay ** k for k in range(horizon)],
            dtype=risk_prob.dtype,
            device=risk_prob.device,
        )
    step_weights = step_weights / step_weights.sum().clamp_min(1e-8)

    weighted_risk = (risk_prob * weights.view(1, 1, -1)).sum(dim=-1)  # (M, N)
    risk_cost = (weighted_risk * step_weights.view(1, -1)).sum(dim=-1)

    if goal_weight > 0.0 and goal_xy_body is not None:
        goal = torch.as_tensor(
            list(goal_xy_body), dtype=pose_delta.dtype, device=pose_delta.device
        )
        if goal.numel() != 2:
            raise ValueError("goal_xy_body must have exactly 2 entries")
        final_xy = pose_delta[:, -1, :2]
        goal_cost = torch.linalg.norm(final_xy - goal.view(1, 2), dim=-1)
        return risk_cost + float(goal_weight) * goal_cost

    return risk_cost


# ---------------------------------------------------------------------------
# Numpy-friendly convenience
# ---------------------------------------------------------------------------

def apply_pose_delta_to_world(
    current_state: np.ndarray,
    pose_delta_body: np.ndarray,
) -> np.ndarray:
    """Compose per-step body-frame deltas with the current world pose.

    Returns a `(M, N, 3)` world-frame [x, y, theta] trajectory given
    `current_state = (6,)` and `pose_delta_body = (M, N, 3)`.
    """
    current_state = np.asarray(current_state, dtype=np.float32).reshape(STATE_DIM)
    pose_delta = np.asarray(pose_delta_body, dtype=np.float32)
    if pose_delta.ndim != 3 or pose_delta.shape[-1] != 3:
        raise ValueError(
            f"pose_delta_body must be (M, N, 3); got {pose_delta.shape}"
        )
    origin_xy = current_state[:2]
    origin_theta = float(current_state[2])
    cos_t = float(np.cos(origin_theta))
    sin_t = float(np.sin(origin_theta))
    xy_body = pose_delta[..., :2]
    # Rotate body-frame xy into world frame.
    world_x = origin_xy[0] + cos_t * xy_body[..., 0] - sin_t * xy_body[..., 1]
    world_y = origin_xy[1] + sin_t * xy_body[..., 0] + cos_t * xy_body[..., 1]
    world_theta = origin_theta + pose_delta[..., 2]
    world_theta = np.arctan2(np.sin(world_theta), np.cos(world_theta)).astype(
        np.float32
    )
    trajectory = np.stack([world_x, world_y, world_theta], axis=-1)
    return trajectory.astype(np.float32)


__all__ = [
    "HighLevelFdmRollout",
    "HighLevelFdmRolloutResult",
    "high_level_fdm_cost",
    "apply_pose_delta_to_world",
]
