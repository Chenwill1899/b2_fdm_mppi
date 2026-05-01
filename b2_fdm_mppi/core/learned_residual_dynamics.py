"""Learned residual dynamics wrapper for Stage 5 NumPy MPPI rollout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.residual_fdm_model import (
    FEATURE_NAMES,
    TARGET_NAMES,
    ResidualFdmMlp,
    build_feature_vector,
)
from b2_fdm_mppi.core.terrain import TerrainField


@dataclass(frozen=True)
class LearnedResidualStep:
    next_state: np.ndarray
    real_control: np.ndarray
    predicted_residual: np.ndarray
    terrain_features: np.ndarray
    terrain_risk: float


class LearnedResidualDynamics:
    """Loads a residual FDM checkpoint and applies it as velocity correction."""

    def __init__(
        self,
        *,
        model: ResidualFdmMlp,
        robot: OmniB2,
        terrain: TerrainField,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device: torch.device,
        checkpoint_path: Path,
        normalization_path: Path,
    ) -> None:
        self.model = model
        self.robot = robot
        self.terrain = terrain
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.normalization_path = Path(normalization_path)

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        *,
        robot: OmniB2,
        terrain: TerrainField,
        checkpoint: str | Path = "best_model.pt",
        normalization: str | Path = "normalization.npz",
        device: str = "cpu",
    ) -> "LearnedResidualDynamics":
        model_dir = Path(model_dir)
        checkpoint_path = _resolve_artifact_path(model_dir, checkpoint)
        normalization_path = _resolve_artifact_path(model_dir, normalization)
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        input_dim = int(checkpoint_data["input_dim"])
        hidden_dim = int(checkpoint_data["hidden_dim"])
        if input_dim != len(FEATURE_NAMES):
            raise ValueError(f"Expected input_dim={len(FEATURE_NAMES)}, checkpoint has input_dim={input_dim}")
        _validate_names(checkpoint_data.get("feature_names"), FEATURE_NAMES, "feature_names")
        _validate_names(checkpoint_data.get("target_names"), TARGET_NAMES, "target_names")

        normalizer = np.load(normalization_path)
        feature_mean = np.asarray(normalizer["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(normalizer["feature_std"], dtype=np.float32)
        target_mean = np.asarray(normalizer["target_mean"], dtype=np.float32)
        target_std = np.asarray(normalizer["target_std"], dtype=np.float32)
        if feature_mean.shape != (len(FEATURE_NAMES),) or feature_std.shape != (len(FEATURE_NAMES),):
            raise ValueError("normalization feature_mean/feature_std shape does not match residual FDM input schema")
        if target_mean.shape != (len(TARGET_NAMES),) or target_std.shape != (len(TARGET_NAMES),):
            raise ValueError("normalization target_mean/target_std shape does not match residual FDM target schema")
        if "feature_names" in normalizer:
            _validate_names(normalizer["feature_names"], FEATURE_NAMES, "normalization feature_names")
        if "target_names" in normalizer:
            _validate_names(normalizer["target_names"], TARGET_NAMES, "normalization target_names")

        torch_device = torch.device(device)
        model = ResidualFdmMlp(input_dim=input_dim, hidden_dim=hidden_dim).to(torch_device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            robot=robot,
            terrain=terrain,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            device=torch_device,
            checkpoint_path=checkpoint_path,
            normalization_path=normalization_path,
        )

    def predict_residual(
        self,
        state: np.ndarray,
        command: np.ndarray,
        terrain_features: np.ndarray | None = None,
        terrain_risk: float | None = None,
    ) -> np.ndarray:
        residuals = self.predict_residual_batch(
            np.asarray(state, dtype=np.float32).reshape(1, 6),
            np.asarray(command, dtype=np.float32).reshape(1, 3),
            None if terrain_features is None else np.asarray(terrain_features, dtype=np.float32).reshape(1, 4),
            None if terrain_risk is None else np.asarray([terrain_risk], dtype=np.float32),
        )
        return residuals[0]

    def predict_residual_batch(
        self,
        states: np.ndarray,
        commands: np.ndarray,
        terrain_features: np.ndarray | None = None,
        terrain_risk: np.ndarray | None = None,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32).reshape(-1, 6)
        commands = np.asarray(commands, dtype=np.float32).reshape(-1, 3)
        if len(states) != len(commands):
            raise ValueError("states and commands must have the same batch length")
        features, risks = self._terrain_batch(states, terrain_features, terrain_risk)
        fdm_features = np.concatenate([states, commands, features, risks[:, None]], axis=1).astype(
            np.float32,
            copy=False,
        )
        standardized = ((fdm_features - self.feature_mean) / self.feature_std).astype(np.float32, copy=False)
        tensor = torch.as_tensor(standardized, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(tensor).detach().cpu().numpy()
        return (pred * self.target_std + self.target_mean).astype(np.float32, copy=False)

    def step(self, state: np.ndarray, command: np.ndarray) -> LearnedResidualStep:
        state = np.asarray(state, dtype=np.float32).reshape(6)
        command = self.robot.clip_control(command)
        terrain_features = self.terrain.feature(float(state[0]), float(state[1]))
        terrain_risk = self.terrain.risk_cost(float(state[0]), float(state[1]), features=terrain_features)
        residual = self.predict_residual(state, command, terrain_features, terrain_risk)
        real_control = self.robot.clip_control(command + residual)
        next_state = self.robot.update_state(state, real_control)
        return LearnedResidualStep(
            next_state=next_state,
            real_control=real_control,
            predicted_residual=residual,
            terrain_features=terrain_features,
            terrain_risk=float(terrain_risk),
        )

    def _terrain_batch(
        self,
        states: np.ndarray,
        terrain_features: np.ndarray | None,
        terrain_risk: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if terrain_features is None:
            features = np.asarray(
                [self.terrain.feature(float(state[0]), float(state[1])) for state in states],
                dtype=np.float32,
            )
        else:
            features = np.asarray(terrain_features, dtype=np.float32).reshape(len(states), 4)
        if terrain_risk is None:
            risks = np.asarray(
                [
                    self.terrain.risk_cost(float(state[0]), float(state[1]), features=features[idx])
                    for idx, state in enumerate(states)
                ],
                dtype=np.float32,
            )
        else:
            risks = np.asarray(terrain_risk, dtype=np.float32).reshape(len(states))
        return features, risks


def _resolve_artifact_path(model_dir: Path, artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return model_dir / path


def _validate_names(actual, expected: list[str], label: str) -> None:
    if actual is None:
        return
    actual_list = [str(item) for item in np.asarray(actual).tolist()]
    if actual_list != list(expected):
        raise ValueError(f"{label} do not match residual FDM schema")
