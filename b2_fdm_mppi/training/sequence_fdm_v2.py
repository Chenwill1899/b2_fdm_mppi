"""Training utilities for Sequence FDM V2 with curriculum learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SequenceFdmDataset(Dataset):
    """Sequence FDM V2 training windows pre-stacked into contiguous tensors.

    Pre-stacks at construction so __getitem__ is a tensor slice. Call .to(device)
    to move everything to GPU once and avoid per-batch host→device transfer.
    If norm dict is provided, inputs and state targets are normalized in-place
    so the model learns on zero-mean unit-variance data.
    """

    def __init__(
        self,
        windows: list[dict],
        horizon_steps: int,
        norm: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.horizon_steps = horizon_steps
        states = np.stack([w["state"] for w in windows]).astype(np.float32)
        controls = np.stack([w["controls"] for w in windows]).astype(np.float32)
        terrain_grids = np.stack([w["terrain_grid"] for w in windows]).astype(np.float32)
        target_states = np.stack([w["target_states"] for w in windows]).astype(np.float32)
        target_risk = np.stack([w["target_risk"] for w in windows]).astype(np.float32)

        if norm is not None:
            states = (states - norm["state_mean"]) / norm["state_std"]
            controls = (controls - norm["control_mean"]) / norm["control_std"]
            target_states = (target_states - norm["state_target_mean"]) / norm["state_target_std"]
            # target_risk stays in [0,1] for BCEWithLogitsLoss

        self.states = torch.from_numpy(states)
        self.controls = torch.from_numpy(controls)
        self.terrain_grids = torch.from_numpy(terrain_grids)
        self.target_states = torch.from_numpy(target_states)
        self.target_risk = torch.from_numpy(target_risk)

    def to(self, device: torch.device) -> "SequenceFdmDataset":
        self.states = self.states.to(device)
        self.controls = self.controls.to(device)
        self.terrain_grids = self.terrain_grids.to(device)
        self.target_states = self.target_states.to(device)
        self.target_risk = self.target_risk.to(device)
        return self

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.states[idx],
            self.controls[idx],
            self.terrain_grids[idx],
            self.target_states[idx],
            self.target_risk[idx],
        )


def compute_normalization(windows: list[dict], horizon_steps: int) -> dict[str, np.ndarray]:
    """Compute mean/std for states, controls, and state targets from training windows.

    Risk targets are kept in [0,1] for BCEWithLogitsLoss and are NOT normalized.
    """
    states = np.stack([w["state"] for w in windows])
    controls = np.stack([w["controls"] for w in windows])
    target_states = np.stack([w["target_states"] for w in windows])

    return {
        "state_mean": np.mean(states, axis=0).astype(np.float32),
        "state_std": np.std(states, axis=0).astype(np.float32) + 1e-8,
        "control_mean": np.mean(controls, axis=(0, 1)).astype(np.float32),
        "control_std": np.std(controls, axis=(0, 1)).astype(np.float32) + 1e-8,
        "state_target_mean": np.mean(target_states, axis=(0, 1)).astype(np.float32),
        "state_target_std": np.std(target_states, axis=(0, 1)).astype(np.float32) + 1e-8,
        "horizon_steps": np.array([horizon_steps], dtype=np.int32),
    }


def train_sequence_fdm_v2(
    windows: list[dict],
    output_dir: str | Path,
    hidden_dims: list[int] | None = None,
    curriculum_phases: list[tuple[int, int, float]] | None = None,
    batch_size: int = 64,
    w_traj: float = 1.0,
    w_risk: float = 0.5,
    val_ratio: float = 0.15,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_tensorboard: bool = True,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train SequenceFdmMlpV2 with curriculum learning.

    Args:
        windows: List of training windows from build_sequence_fdm_windows
        output_dir: Directory to save checkpoints and normalization
        hidden_dims: MLP hidden layer sizes
        curriculum_phases: List of (horizon_steps, epochs, lr) tuples
        batch_size: Training batch size
        w_traj: Trajectory loss weight
        w_risk: Risk loss weight
        val_ratio: Fraction of windows for validation
        patience: Early stopping patience (epochs)
        device: "cuda" or "cpu"
        use_tensorboard: Whether to log metrics to TensorBoard
        resume_from: Checkpoint directory to resume from (loads best_model.pt + optimizer.pt)
    """
    from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hidden_dims is None:
        hidden_dims = [256, 256, 256]
    if curriculum_phases is None:
        curriculum_phases = [(5, 50, 1e-3), (10, 50, 5e-4), (20, 100, 1e-4), (20, 50, 5e-5)]

    # Split train/val
    n_val = int(len(windows) * val_ratio)
    rng = np.random.default_rng(42)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    val_indices = set(indices[:n_val].tolist())
    train_windows = [w for i, w in enumerate(windows) if i not in val_indices]
    val_windows = [w for i, w in enumerate(windows) if i in val_indices]

    # Compute normalization on training set only (use max horizon)
    max_horizon = max(h for h, _, _ in curriculum_phases)
    norm = compute_normalization(train_windows, horizon_steps=max_horizon)
    np.savez(output_dir / "normalization.npz", **norm)

    torch_device = torch.device(device)
    best_ckpt_path = None
    best_val_loss = float("inf")
    history: list[dict] = []

    # TensorBoard setup
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(output_dir / "runs"))
        except ImportError:
            print("Warning: tensorboard not available, skipping TB logging")

    global_step = 0

    for phase_idx, (horizon, epochs, lr) in enumerate(curriculum_phases):
        print(f"\n=== Curriculum Phase {phase_idx + 1}/{len(curriculum_phases)}: H={horizon}, lr={lr} ===")

        # Filter and truncate windows to current horizon
        phase_train = [w for w in train_windows if w["controls"].shape[0] >= horizon]
        phase_val = [w for w in val_windows if w["controls"].shape[0] >= horizon]

        if not phase_train:
            raise ValueError(f"No training windows with horizon >= {horizon}")

        def truncate(w: dict) -> dict:
            return {
                "state": w["state"],
                "controls": w["controls"][:horizon],
                "terrain_grid": w["terrain_grid"],
                "target_states": w["target_states"][:horizon],
                "target_risk": w["target_risk"][:horizon],
            }

        phase_train = [truncate(w) for w in phase_train]
        phase_val = [truncate(w) for w in phase_val]

        train_dataset = SequenceFdmDataset(phase_train, horizon_steps=horizon, norm=norm).to(torch_device)
        val_dataset = SequenceFdmDataset(phase_val, horizon_steps=horizon, norm=norm).to(torch_device)
        n_train = len(train_dataset)
        n_val = len(val_dataset)

        # Create or load model
        model = SequenceFdmMlpV2(horizon_steps=horizon, hidden_dims=hidden_dims).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Resume logic: load model + optimizer from previous training round
        if resume_from is not None:
            resume_path = Path(resume_from)
            resume_model_path = resume_path / "best_model.pt"
            resume_opt_path = resume_path / "optimizer.pt"
            if resume_model_path.exists():
                ckpt = torch.load(resume_model_path, map_location=device, weights_only=False)
                if ckpt.get("horizon_steps") == horizon:
                    model.load_state_dict(ckpt["model_state_dict"])
                    print(f"  Resumed model from {resume_model_path}")
                else:
                    print(f"  Warning: checkpoint horizon {ckpt.get('horizon_steps')} != current {horizon}, starting fresh")
            if resume_opt_path.exists():
                opt_ckpt = torch.load(resume_opt_path, map_location=device, weights_only=False)
                if opt_ckpt.get("horizon_steps") == horizon:
                    optimizer.load_state_dict(opt_ckpt["optimizer_state_dict"])
                    print(f"  Resumed optimizer state from {resume_opt_path}")
                else:
                    print(f"  Warning: optimizer horizon {opt_ckpt.get('horizon_steps')} != current {horizon}, skipping optimizer resume")
            resume_from = None  # only resume on first phase
        elif best_ckpt_path is not None and best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("horizon_steps") == horizon:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"  Loaded checkpoint from phase {phase_idx}")
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        phase_best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_loss_sum = torch.zeros((), device=torch_device)
            train_traj_sum = torch.zeros((), device=torch_device)
            train_risk_sum = torch.zeros((), device=torch_device)
            n_train_batches = 0
            perm = torch.randperm(n_train, device=torch_device)
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                state = train_dataset.states[idx].contiguous()
                controls = train_dataset.controls[idx].contiguous()
                grid = train_dataset.terrain_grids[idx].contiguous()
                target_states = train_dataset.target_states[idx].contiguous()
                target_risk = train_dataset.target_risk[idx].contiguous()

                pred_states, pred_risk_logits = model(state, controls, grid)

                loss_traj = mse_loss(pred_states, target_states)
                loss_risk = bce_loss(pred_risk_logits, target_risk)
                loss = w_traj * loss_traj + w_risk * loss_risk

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.detach()
                train_traj_sum += loss_traj.detach()
                train_risk_sum += loss_risk.detach()
                n_train_batches += 1
                global_step += 1

            # Validation
            model.eval()
            val_loss_sum = torch.zeros((), device=torch_device)
            val_traj_sum = torch.zeros((), device=torch_device)
            val_risk_sum = torch.zeros((), device=torch_device)
            n_val_batches = 0
            with torch.no_grad():
                for i in range(0, n_val, batch_size):
                    state = val_dataset.states[i : i + batch_size].contiguous()
                    controls = val_dataset.controls[i : i + batch_size].contiguous()
                    grid = val_dataset.terrain_grids[i : i + batch_size].contiguous()
                    target_states = val_dataset.target_states[i : i + batch_size].contiguous()
                    target_risk = val_dataset.target_risk[i : i + batch_size].contiguous()

                    pred_states, pred_risk_logits = model(state, controls, grid)
                    loss_traj = mse_loss(pred_states, target_states)
                    loss_risk = bce_loss(pred_risk_logits, target_risk)
                    loss = w_traj * loss_traj + w_risk * loss_risk
                    val_loss_sum += loss
                    val_traj_sum += loss_traj
                    val_risk_sum += loss_risk
                    n_val_batches += 1

            avg_train = float(train_loss_sum.item() / max(n_train_batches, 1))
            avg_val = float(val_loss_sum.item() / max(n_val_batches, 1))
            avg_train_traj = float(train_traj_sum.item() / max(n_train_batches, 1))
            avg_train_risk = float(train_risk_sum.item() / max(n_train_batches, 1))
            avg_val_traj = float(val_traj_sum.item() / max(n_val_batches, 1))
            avg_val_risk = float(val_risk_sum.item() / max(n_val_batches, 1))

            print(f"  Epoch {epoch + 1}/{epochs}: train={avg_train:.6f}, val={avg_val:.6f}")

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar(f"Phase{phase_idx}/Loss/train", avg_train, epoch)
                writer.add_scalar(f"Phase{phase_idx}/Loss/val", avg_val, epoch)
                writer.add_scalar(f"Phase{phase_idx}/TrajLoss/train", avg_train_traj, epoch)
                writer.add_scalar(f"Phase{phase_idx}/TrajLoss/val", avg_val_traj, epoch)
                writer.add_scalar(f"Phase{phase_idx}/RiskLoss/train", avg_train_risk, epoch)
                writer.add_scalar(f"Phase{phase_idx}/RiskLoss/val", avg_val_risk, epoch)
                writer.add_scalar(f"Phase{phase_idx}/LearningRate", lr, epoch)

            history.append({
                "phase": phase_idx,
                "epoch": epoch,
                "horizon": horizon,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "train_traj_loss": avg_train_traj,
                "train_risk_loss": avg_train_risk,
                "val_traj_loss": avg_val_traj,
                "val_risk_loss": avg_val_risk,
            })

            if avg_val < phase_best_loss:
                phase_best_loss = avg_val
                no_improve = 0
                ckpt_path = output_dir / f"best_model_h{horizon}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "horizon_steps": horizon,
                    "hidden_dims": hidden_dims,
                    "input_dim": 6 + 3 * horizon + 81,
                    "target_dim": 6 * horizon + horizon,
                    "phase": phase_idx,
                }, ckpt_path)
                best_ckpt_path = ckpt_path
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "horizon_steps": horizon,
                        "hidden_dims": hidden_dims,
                        "input_dim": 6 + 3 * horizon + 81,
                        "target_dim": 6 * horizon + horizon,
                        "phase": phase_idx,
                    }, output_dir / "best_model.pt")
                    torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "horizon_steps": horizon,
                        "lr": lr,
                    }, output_dir / "optimizer.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    if writer is not None:
        writer.close()

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "output_dir": str(output_dir),
    }
