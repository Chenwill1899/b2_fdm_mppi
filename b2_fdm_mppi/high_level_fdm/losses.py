"""Loss functions for the High-Level FDM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class HighLevelFdmLossConfig:
    lambda_pose: float = 1.0
    lambda_risk: float = 1.0
    lambda_smooth: float = 0.0
    huber_delta: float = 1.0
    # Later-steps are less important; first-step weight is 1.0, decaying by
    # this factor per step. If None, all steps are weighted equally.
    step_weight_decay: float | None = 0.98
    # Positive-class weight for the BCE risk loss. Useful when collision
    # labels are rare. If None, uniform weights.
    risk_pos_weight: tuple[float, ...] | None = None


class HighLevelFdmLoss(nn.Module):
    """Combined Huber + BCE loss with step weighting and optional smoothness."""

    def __init__(
        self,
        config: HighLevelFdmLossConfig,
        *,
        horizon: int,
        num_risk_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.horizon = int(horizon)
        self.num_risk_channels = int(num_risk_channels)

        if config.step_weight_decay is None:
            step_weights = torch.ones(self.horizon, dtype=torch.float32)
        else:
            decay = float(config.step_weight_decay)
            step_weights = torch.tensor(
                [decay ** k for k in range(self.horizon)], dtype=torch.float32
            )
        step_weights = step_weights / step_weights.sum().clamp_min(1e-8)
        self.register_buffer("step_weights", step_weights)

        if config.risk_pos_weight is None:
            pos_weight = torch.ones(self.num_risk_channels, dtype=torch.float32)
        else:
            if len(config.risk_pos_weight) != self.num_risk_channels:
                raise ValueError(
                    "risk_pos_weight length must equal num_risk_channels; "
                    f"got {len(config.risk_pos_weight)} vs {self.num_risk_channels}"
                )
            pos_weight = torch.tensor(
                config.risk_pos_weight, dtype=torch.float32
            )
        self.register_buffer("risk_pos_weight", pos_weight)

    def forward(
        self,
        pose_pred: torch.Tensor,   # (B, N, 4)
        pose_target: torch.Tensor, # (B, N, 4)
        risk_logits: torch.Tensor, # (B, N, K)
        risk_target: torch.Tensor, # (B, N, K)
        risk_mask: torch.Tensor,   # (B, N, K)
    ) -> dict[str, torch.Tensor]:
        cfg = self.config

        pose_err = F.huber_loss(
            pose_pred,
            pose_target,
            reduction="none",
            delta=float(cfg.huber_delta),
        ).mean(dim=-1)  # (B, N)
        step_weights = self.step_weights.view(1, -1)
        pose_loss = (pose_err * step_weights).sum(dim=-1).mean()

        risk_err = F.binary_cross_entropy_with_logits(
            risk_logits,
            risk_target,
            pos_weight=self.risk_pos_weight.view(1, 1, -1),
            reduction="none",
        )
        masked_sum = (risk_err * risk_mask).sum(dim=-1)
        valid = risk_mask.sum(dim=-1).clamp_min(1e-6)
        risk_per_step = masked_sum / valid
        risk_loss = (risk_per_step * step_weights).sum(dim=-1).mean()

        if cfg.lambda_smooth > 0.0 and pose_pred.shape[1] > 1:
            smooth = pose_pred[:, 1:, :] - pose_pred[:, :-1, :]
            smooth_target = pose_target[:, 1:, :] - pose_target[:, :-1, :]
            smooth_loss = F.huber_loss(
                smooth, smooth_target, reduction="mean", delta=float(cfg.huber_delta)
            )
        else:
            smooth_loss = pose_loss.new_zeros(())

        total = (
            cfg.lambda_pose * pose_loss
            + cfg.lambda_risk * risk_loss
            + cfg.lambda_smooth * smooth_loss
        )
        return {
            "total": total,
            "pose": pose_loss.detach(),
            "risk": risk_loss.detach(),
            "smooth": smooth_loss.detach(),
        }


def zero_residual_pose_loss(
    pose_target: torch.Tensor,
    nominal_pose: torch.Tensor,
    *,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Baseline pose loss: predict the analytical nominal rollout only.

    Any trained model that does not beat this baseline is not useful.
    """
    return F.huber_loss(
        nominal_pose, pose_target, reduction="mean", delta=float(huber_delta)
    )
