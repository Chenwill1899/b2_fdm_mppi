"""Training and evaluation loops for the High-Level FDM."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from b2_fdm_mppi.high_level_fdm.dataset import HighLevelFdmDataset, load_manifest
from b2_fdm_mppi.high_level_fdm.losses import (
    HighLevelFdmLoss,
    HighLevelFdmLossConfig,
    zero_residual_pose_loss,
)
from b2_fdm_mppi.high_level_fdm.model import (
    HighLevelFdm,
    HighLevelFdmModelConfig,
)
from b2_fdm_mppi.high_level_fdm.schema import HighLevelFdmSchema


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HighLevelFdmTrainerConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    min_learning_rate: float = 1e-5
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 123
    device: str = "cpu"
    amp: bool = False
    log_every: int = 50


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HighLevelFdmTrainer:
    """Minimal but complete training loop around HighLevelFdm."""

    def __init__(
        self,
        *,
        model: HighLevelFdm,
        loss_fn: HighLevelFdmLoss,
        config: HighLevelFdmTrainerConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(config.epochs),
            eta_min=float(config.min_learning_rate),
        )

    # ------------------------------------------------------------------
    # Full train entrypoint
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, float]] = []
        best_val = float("inf")
        best_epoch = 0
        for epoch in range(1, int(self.config.epochs) + 1):
            train_metrics = self._epoch(train_loader, training=True)
            if val_loader is not None:
                val_metrics = self._epoch(val_loader, training=False)
            else:
                val_metrics = {k: float("nan") for k in train_metrics}
            self.scheduler.step()
            entry = {
                "epoch": int(epoch),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            }
            history.append(entry)
            current_val = float(val_metrics.get("total", float("inf")))
            if val_loader is not None and current_val < best_val:
                best_val = current_val
                best_epoch = int(epoch)
                self._save_checkpoint(output_dir / "best_model.pt", epoch=epoch)

        self._save_checkpoint(output_dir / "model.pt", epoch=int(self.config.epochs))
        metrics = {
            "history": history,
            "best_epoch": int(best_epoch),
            "best_val_total": float(best_val if best_val != float("inf") else "nan"),
        }
        (output_dir / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        metrics = self._epoch(loader, training=False, collect_extras=True)
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        collect_extras: bool = False,
    ) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        running: dict[str, float] = {"total": 0.0, "pose": 0.0, "risk": 0.0, "smooth": 0.0}
        pose_abs_err = 0.0
        pose_theta_err = 0.0
        risk_brier = 0.0
        num_samples = 0
        pose_baseline_err = 0.0

        for batch in loader:
            state_history = batch["state_history"].to(self.device, non_blocking=True)
            map_patch = batch["map_patch"].to(self.device, non_blocking=True)
            control_sequence = batch["control_sequence"].to(self.device, non_blocking=True)
            pose_target = batch["pose_target"].to(self.device, non_blocking=True)
            risk_target = batch["risk_target"].to(self.device, non_blocking=True)
            risk_mask = batch["risk_mask"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(training):
                pose_pred, risk_logits = self.model(
                    state_history, map_patch, control_sequence
                )
                losses = self.loss_fn(
                    pose_pred=pose_pred,
                    pose_target=pose_target,
                    risk_logits=risk_logits,
                    risk_target=risk_target,
                    risk_mask=risk_mask,
                )

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if self.config.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.config.grad_clip)
                    )
                self.optimizer.step()

            batch_size = int(state_history.shape[0])
            num_samples += batch_size
            for k in running:
                running[k] += float(losses[k].detach()) * batch_size

            if collect_extras:
                with torch.no_grad():
                    # Pose error in [dx, dy] and angular error (radians) using
                    # the network's raw prediction (not standardized).
                    pose_abs = (pose_pred[..., :2] - pose_target[..., :2]).abs().mean()
                    dth_pred = torch.atan2(pose_pred[..., 2], pose_pred[..., 3])
                    dth_tgt = torch.atan2(pose_target[..., 2], pose_target[..., 3])
                    ang_err = torch.atan2(
                        torch.sin(dth_pred - dth_tgt), torch.cos(dth_pred - dth_tgt)
                    ).abs().mean()
                    probs = torch.sigmoid(risk_logits)
                    brier = ((probs - risk_target) ** 2 * risk_mask).sum() / risk_mask.sum().clamp_min(1e-6)
                    nominal_pose = self.model._nominal_rollout_pose_param(control_sequence)
                    baseline = zero_residual_pose_loss(
                        pose_target, nominal_pose, huber_delta=float(self.loss_fn.config.huber_delta)
                    )
                pose_abs_err += float(pose_abs) * batch_size
                pose_theta_err += float(ang_err) * batch_size
                risk_brier += float(brier) * batch_size
                pose_baseline_err += float(baseline) * batch_size

        num_samples = max(num_samples, 1)
        results = {k: running[k] / num_samples for k in running}
        if collect_extras:
            results["pose_xy_abs_err"] = pose_abs_err / num_samples
            results["pose_theta_abs_err"] = pose_theta_err / num_samples
            results["risk_brier"] = risk_brier / num_samples
            results["pose_zero_residual_baseline"] = pose_baseline_err / num_samples
        return results

    def _save_checkpoint(self, path: Path, *, epoch: int) -> None:
        payload = self.model.checkpoint_payload()
        payload["epoch"] = int(epoch)
        payload["trainer_config"] = asdict(self.config)
        payload["loss_config"] = asdict(self.loss_fn.config)
        torch.save(payload, str(path))


# ---------------------------------------------------------------------------
# Entry helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_model_from_manifest(
    manifest: dict[str, Any],
    *,
    dt: float,
    model_overrides: dict[str, Any] | None = None,
) -> tuple[HighLevelFdm, HighLevelFdmSchema]:
    schema = HighLevelFdmSchema.from_dict(manifest["schema"])
    cfg = HighLevelFdmModelConfig.from_schema(
        schema, dt=dt, **(model_overrides or {})
    )
    return HighLevelFdm(cfg), schema


def train_high_level_fdm(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    trainer_config: HighLevelFdmTrainerConfig,
    loss_config: HighLevelFdmLossConfig | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """High-level CLI entrypoint: load dataset, build model, train, evaluate."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(trainer_config.seed))

    manifest = load_manifest(dataset_dir)
    schema = HighLevelFdmSchema.from_dict(manifest["schema"])
    dt = float(manifest["dt"])

    train_ds = HighLevelFdmDataset(dataset_dir, split="train")
    val_ds = HighLevelFdmDataset(dataset_dir, split="val")
    test_ds = HighLevelFdmDataset(dataset_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(trainer_config.batch_size),
        shuffle=True,
        num_workers=int(trainer_config.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(trainer_config.batch_size),
        shuffle=False,
        num_workers=int(trainer_config.num_workers),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(trainer_config.batch_size),
        shuffle=False,
        num_workers=int(trainer_config.num_workers),
    )

    model, schema_used = build_model_from_manifest(
        manifest, dt=dt, model_overrides=model_overrides
    )
    loss_fn = HighLevelFdmLoss(
        loss_config or HighLevelFdmLossConfig(),
        horizon=schema_used.horizon,
        num_risk_channels=schema_used.num_risk_channels(),
    )
    trainer = HighLevelFdmTrainer(
        model=model, loss_fn=loss_fn, config=trainer_config
    )
    training_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
    )

    val_metrics = trainer.evaluate(val_loader)
    test_metrics = trainer.evaluate(test_loader)

    (output_dir / "schema.json").write_text(
        json.dumps(schema_used.to_dict(), indent=2), encoding="utf-8"
    )
    final = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "schema": schema_used.to_dict(),
        "dt": dt,
        "trainer_config": asdict(trainer_config),
        "loss_config": asdict(loss_fn.config),
        "training_metrics": training_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    return final


def evaluate_high_level_fdm(
    *,
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(dataset_dir)
    payload = torch.load(str(checkpoint_path), map_location=device)
    model = HighLevelFdm.from_checkpoint_payload(payload)
    model.to(device)
    loss_fn = HighLevelFdmLoss(
        HighLevelFdmLossConfig(**payload.get("loss_config", {})),
        horizon=int(payload["config"]["horizon"]),
        num_risk_channels=int(payload["config"]["num_risk_channels"]),
    )

    dataset = HighLevelFdmDataset(dataset_dir, split=split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    trainer = HighLevelFdmTrainer(
        model=model,
        loss_fn=loss_fn,
        config=HighLevelFdmTrainerConfig(
            epochs=1, batch_size=64, device=device
        ),
    )
    metrics = trainer.evaluate(loader)
    result = {
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "metrics": metrics,
        "schema": manifest["schema"],
    }
    (output_dir / f"{split}_eval_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
