"""Torch implementation of the High-Level FDM.

The model takes:

    state_history    (B, H, STATE_DIM)           body frame of t
    map_patch        (B, C, Gh, Gw)              aligned to t
    control_sequence (B, N, CONTROL_DIM)         body frame

and returns:

    pose_param       (B, N, POSE_PARAM_DIM)      [dx, dy, sin_dth, cos_dth]
    risk_logits      (B, N, K)                   pre-sigmoid

Design principles:

* The map is only encoded once per forward pass (per MPPI optimization step).
* The history is encoded once and fused with the map to form a static
  context vector.
* A small GRUCell consumes per-step commands, emitting pose residuals that
  are *added* to the analytical nominal rollout computed in-graph. This
  keeps the network close to "zero residual = physics nominal" and greatly
  improves sample efficiency on the synthetic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from b2_fdm_mppi.high_level_fdm.schema import (
    CONTROL_DIM,
    POSE_PARAM_DIM,
    STATE_DIM,
    HighLevelFdmSchema,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HighLevelFdmModelConfig:
    history_length: int
    horizon: int
    map_channels: int
    map_height: int
    map_width: int
    dt: float
    num_risk_channels: int = 4
    d_hist: int = 64
    d_map: int = 64
    d_ctx: int = 128
    d_hidden: int = 128
    pose_head_hidden: int = 64
    risk_head_hidden: int = 64
    dropout: float = 0.0

    @classmethod
    def from_schema(
        cls,
        schema: HighLevelFdmSchema,
        *,
        dt: float,
        **overrides: Any,
    ) -> "HighLevelFdmModelConfig":
        return cls(
            history_length=schema.history_length,
            horizon=schema.horizon,
            map_channels=schema.map_channels,
            map_height=schema.map_height,
            map_width=schema.map_width,
            dt=float(dt),
            num_risk_channels=schema.num_risk_channels(),
            **overrides,
        )


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _HistoryEncoder(nn.Module):
    def __init__(self, history_length: int, state_dim: int, d_hist: int) -> None:
        super().__init__()
        hidden = max(32, d_hist)
        self.net = nn.Sequential(
            nn.Linear(history_length * state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_hist),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, H, D) -> flatten -> (B, d_hist)
        batch = history.shape[0]
        flat = history.reshape(batch, -1)
        return self.net(flat)


class _MapEncoder(nn.Module):
    def __init__(
        self,
        map_channels: int,
        map_height: int,
        map_width: int,
        d_map: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(32, d_map)
        # Store for shape logging; not used in forward.
        self.map_height = int(map_height)
        self.map_width = int(map_width)

    def forward(self, map_patch: torch.Tensor) -> torch.Tensor:
        # map_patch: (B, C, Gh, Gw) -> (B, d_map)
        features = self.conv(map_patch)
        pooled = self.pool(features).flatten(1)
        return self.proj(pooled)


class _PoseResidualHead(nn.Module):
    def __init__(self, d_hidden: int, pose_head_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, pose_head_hidden),
            nn.GELU(),
            nn.Linear(pose_head_hidden, POSE_PARAM_DIM),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class _RiskHead(nn.Module):
    def __init__(self, d_hidden: int, risk_head_hidden: int, num_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, risk_head_hidden),
            nn.GELU(),
            nn.Linear(risk_head_hidden, num_channels),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


# ---------------------------------------------------------------------------
# HighLevelFdm
# ---------------------------------------------------------------------------

class HighLevelFdm(nn.Module):
    """Recurrent high-level forward dynamics model with risk head."""

    def __init__(self, config: HighLevelFdmModelConfig) -> None:
        super().__init__()
        self.config = config

        self.history_encoder = _HistoryEncoder(
            history_length=config.history_length,
            state_dim=STATE_DIM,
            d_hist=config.d_hist,
        )
        self.map_encoder = _MapEncoder(
            map_channels=config.map_channels,
            map_height=config.map_height,
            map_width=config.map_width,
            d_map=config.d_map,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(config.d_hist + config.d_map, config.d_ctx),
            nn.GELU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
        )
        self.decoder_cell = nn.GRUCell(
            input_size=config.d_ctx + CONTROL_DIM,
            hidden_size=config.d_hidden,
        )
        self.pose_head = _PoseResidualHead(
            d_hidden=config.d_hidden,
            pose_head_hidden=config.pose_head_hidden,
        )
        self.risk_head = _RiskHead(
            d_hidden=config.d_hidden,
            risk_head_hidden=config.risk_head_hidden,
            num_channels=config.num_risk_channels,
        )
        # A learned initial hidden state lets the decoder "specialize" to
        # the first-step prediction rather than starting from zeros.
        self.initial_hidden = nn.Parameter(torch.zeros(1, config.d_hidden))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        state_history: torch.Tensor,
        map_patch: torch.Tensor,
        control_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(state_history, map_patch, control_sequence)

        context = self._encode_context(state_history, map_patch)
        pose_residuals, risk_logits = self._decode(context, control_sequence)

        nominal_pose_param = self._nominal_rollout_pose_param(control_sequence)
        pose_param = nominal_pose_param + pose_residuals
        return pose_param, risk_logits

    def predict(
        self,
        state_history: torch.Tensor,
        map_patch: torch.Tensor,
        control_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference variant: returns (pose_param, risk_prob) with no grad."""
        self.eval()
        with torch.no_grad():
            pose_param, risk_logits = self.forward(
                state_history, map_patch, control_sequence
            )
            risk_prob = torch.sigmoid(risk_logits)
        return pose_param, risk_prob

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        state_history: torch.Tensor,
        map_patch: torch.Tensor,
        control_sequence: torch.Tensor,
    ) -> None:
        cfg = self.config
        if state_history.ndim != 3 or state_history.shape[-1] != STATE_DIM:
            raise ValueError(
                f"state_history must be (B, H, {STATE_DIM}); got {tuple(state_history.shape)}"
            )
        if state_history.shape[1] != cfg.history_length:
            raise ValueError(
                f"state_history H={state_history.shape[1]} != config {cfg.history_length}"
            )
        if map_patch.ndim != 4 or map_patch.shape[1] != cfg.map_channels:
            raise ValueError(
                "map_patch must be (B, map_channels, Gh, Gw); "
                f"got {tuple(map_patch.shape)}, expected channels {cfg.map_channels}"
            )
        if control_sequence.ndim != 3 or control_sequence.shape[-1] != CONTROL_DIM:
            raise ValueError(
                f"control_sequence must be (B, N, {CONTROL_DIM}); got {tuple(control_sequence.shape)}"
            )
        if control_sequence.shape[1] != cfg.horizon:
            raise ValueError(
                f"control_sequence N={control_sequence.shape[1]} != config horizon {cfg.horizon}"
            )

    def _encode_context(
        self, state_history: torch.Tensor, map_patch: torch.Tensor
    ) -> torch.Tensor:
        hist = self.history_encoder(state_history)
        mapv = self.map_encoder(map_patch)
        fused = torch.cat([hist, mapv], dim=-1)
        return self.context_proj(fused)

    def _decode(
        self,
        context: torch.Tensor,
        control_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        horizon = control_sequence.shape[1]
        hidden = self.initial_hidden.expand(batch, -1).contiguous()
        pose_residuals = []
        risk_logits = []
        for step in range(horizon):
            u_step = control_sequence[:, step, :]
            decoder_input = torch.cat([context, u_step], dim=-1)
            hidden = self.decoder_cell(decoder_input, hidden)
            pose_residuals.append(self.pose_head(hidden))
            risk_logits.append(self.risk_head(hidden))
        pose_residuals_t = torch.stack(pose_residuals, dim=1)
        risk_logits_t = torch.stack(risk_logits, dim=1)
        return pose_residuals_t, risk_logits_t

    def _nominal_rollout_pose_param(
        self, control_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Torch-native re-implementation of schema.nominal_rollout in pose_param form."""
        dt = float(self.config.dt)
        batch, horizon, _ = control_sequence.shape
        device = control_sequence.device
        dtype = control_sequence.dtype
        pose_param = torch.zeros(
            batch, horizon, POSE_PARAM_DIM, device=device, dtype=dtype
        )
        x = torch.zeros(batch, device=device, dtype=dtype)
        y = torch.zeros(batch, device=device, dtype=dtype)
        theta = torch.zeros(batch, device=device, dtype=dtype)
        for step in range(horizon):
            vx = control_sequence[:, step, 0]
            vy = control_sequence[:, step, 1]
            wz = control_sequence[:, step, 2]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            x = x + (vx * cos_t - vy * sin_t) * dt
            y = y + (vx * sin_t + vy * cos_t) * dt
            theta = theta + wz * dt
            pose_param[:, step, 0] = x
            pose_param[:, step, 1] = y
            pose_param[:, step, 2] = torch.sin(theta)
            pose_param[:, step, 3] = torch.cos(theta)
        return pose_param

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.state_dict(),
            "config": {
                "history_length": self.config.history_length,
                "horizon": self.config.horizon,
                "map_channels": self.config.map_channels,
                "map_height": self.config.map_height,
                "map_width": self.config.map_width,
                "dt": self.config.dt,
                "num_risk_channels": self.config.num_risk_channels,
                "d_hist": self.config.d_hist,
                "d_map": self.config.d_map,
                "d_ctx": self.config.d_ctx,
                "d_hidden": self.config.d_hidden,
                "pose_head_hidden": self.config.pose_head_hidden,
                "risk_head_hidden": self.config.risk_head_hidden,
                "dropout": self.config.dropout,
            },
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> "HighLevelFdm":
        cfg = HighLevelFdmModelConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["model_state_dict"])
        return model
