#!/usr/bin/env python3
"""Train a baseline residual velocity FDM from oracle dataset splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FEATURE_NAMES = [
    "state_x",
    "state_y",
    "state_theta",
    "state_vx",
    "state_vy",
    "state_wz",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "terrain_slope_f",
    "terrain_slope_l",
    "terrain_roughness",
    "terrain_friction",
    "terrain_risk",
]
TARGET_NAMES = ["exec_du_vx", "exec_du_vy", "exec_du_wz"]
SPLITS = ("train", "val", "test")


class ResidualFdmMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(TARGET_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_residual_fdm_dataset(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    dataset_dir = Path(dataset_dir)
    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        with np.load(dataset_dir / f"{split}.npz") as data:
            features = _features_from_split(data)
            targets = np.asarray(data["exec_residuals"], dtype=np.float32)
        arrays[f"{split}_features"] = features
        arrays[f"{split}_targets"] = targets
    return arrays


def train_residual_fdm(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 50,
    batch_size: int = 256,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 123,
    device: str = "cpu",
) -> dict:
    _set_seed(seed)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_residual_fdm_dataset(dataset_dir)

    x_mean, x_std = _normalization(arrays["train_features"])
    y_mean, y_std = _normalization(arrays["train_targets"])
    np.savez(
        output_dir / "normalization.npz",
        feature_mean=x_mean,
        feature_std=x_std,
        target_mean=y_mean,
        target_std=y_std,
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
    )

    train_x = _standardize(arrays["train_features"], x_mean, x_std)
    train_y = _standardize(arrays["train_targets"], y_mean, y_std)
    val_x = _standardize(arrays["val_features"], x_mean, x_std)
    val_y = _standardize(arrays["val_targets"], y_mean, y_std)
    test_x = _standardize(arrays["test_features"], x_mean, x_std)
    test_y = _standardize(arrays["test_targets"], y_mean, y_std)

    torch_device = torch.device(device)
    model = ResidualFdmMlp(input_dim=train_x.shape[1], hidden_dim=int(hidden_dim)).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.MSELoss()

    train_tensor_x = torch.as_tensor(train_x, dtype=torch.float32, device=torch_device)
    train_tensor_y = torch.as_tensor(train_y, dtype=torch.float32, device=torch_device)
    val_tensor_x = torch.as_tensor(val_x, dtype=torch.float32, device=torch_device)
    val_tensor_y = torch.as_tensor(val_y, dtype=torch.float32, device=torch_device)
    test_tensor_x = torch.as_tensor(test_x, dtype=torch.float32, device=torch_device)
    test_tensor_y = torch.as_tensor(test_y, dtype=torch.float32, device=torch_device)

    history_train = []
    history_val = []
    n_train = int(train_tensor_x.shape[0])
    for _epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(n_train, device=torch_device)
        batch_losses = []
        for start in range(0, n_train, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            pred = model(train_tensor_x[idx])
            loss = loss_fn(pred, train_tensor_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        history_train.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
        history_val.append(_eval_loss(model, loss_fn, val_tensor_x, val_tensor_y))

    metrics = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "train_transitions": int(arrays["train_features"].shape[0]),
        "val_transitions": int(arrays["val_features"].shape[0]),
        "test_transitions": int(arrays["test_features"].shape[0]),
        "train_loss": history_train,
        "val_loss": history_val,
        "test_loss": _eval_loss(model, loss_fn, test_tensor_x, test_tensor_y),
        "val_mse": _eval_raw_mse(
            model,
            val_tensor_x,
            arrays["val_targets"],
            y_mean,
            y_std,
        ),
        "test_mse": _eval_raw_mse(
            model,
            test_tensor_x,
            arrays["test_targets"],
            y_mean,
            y_std,
        ),
        "zero_residual_val_mse": _zero_residual_mse(arrays["val_targets"]),
        "zero_residual_test_mse": _zero_residual_mse(arrays["test_targets"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": int(hidden_dim),
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )
    return metrics


def _features_from_split(data) -> np.ndarray:
    terrain_risk = np.asarray(data["terrain_risk"], dtype=np.float32).reshape(-1, 1)
    return np.concatenate(
        [
            np.asarray(data["states"], dtype=np.float32),
            np.asarray(data["cmd_controls"], dtype=np.float32),
            np.asarray(data["terrain_features"], dtype=np.float32),
            terrain_risk,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32, copy=False)


def _eval_loss(model: nn.Module, loss_fn: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> float:
    if int(features.shape[0]) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(features), targets).detach().cpu())


def _eval_raw_mse(
    model: nn.Module,
    features: torch.Tensor,
    targets: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> float:
    if int(features.shape[0]) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        pred = model(features).detach().cpu().numpy()
    pred_raw = pred * target_std + target_mean
    diff = pred_raw - np.asarray(targets, dtype=np.float32)
    return float(np.mean(diff * diff))


def _zero_residual_mse(targets: np.ndarray) -> float:
    if targets.size == 0:
        return 0.0
    return float(np.mean(np.asarray(targets, dtype=np.float32) ** 2))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metrics = train_residual_fdm(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
