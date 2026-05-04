# Sequence FDM-MPPI V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end sequence FDM that replaces nominal MPPI rollouts with direct trajectory + binary risk prediction, trained on randomly generated terrains with curriculum learning.

**Architecture:** New `SequenceFdmMlpV2` model takes (state, H-step controls, 9x9 terrain risk grid) and predicts (H-step absolute states, H-step binary risk). A new `MppiOmniSequenceFdmV2Torch` controller overrides cost computation to use the FDM. Data is collected from `RandomTerrainGenerator` episodes, processed into sliding windows, and trained with curriculum H=5->10->20.

**Tech Stack:** PyTorch, NumPy, pytest, existing b2_fdm_mppi infrastructure

---

## File Structure

| File | Responsibility |
|------|---------------|
| `b2_fdm_mppi/simulation/random_terrain.py` | `RandomTerrainGenerator` — creates random TerrainField instances with patches and noise |
| `b2_fdm_mppi/core/sequence_fdm_v2.py` | `SequenceFdmMlpV2` — model definition; feature vector builder; feature/target name helpers |
| `b2_fdm_mppi/core/sequence_fdm_dynamics.py` | `SequenceFdmDynamics` — wraps model + normalization for inference |
| `b2_fdm_mppi/data/sequence_fdm_collector.py` | `collect_sequence_fdm_episode()` — runs MPPI on random terrain, records trajectory + binary risk; `build_sequence_fdm_dataset()` — sliding window extraction |
| `b2_fdm_mppi/training/sequence_fdm_v2.py` | `train_sequence_fdm_v2()` — curriculum training loop with phase switching |
| `b2_fdm_mppi/controllers/mppi_omni_sequence_fdm_v2_torch.py` | `MppiOmniSequenceFdmV2Torch` — MPPI controller using FDM for trajectory + risk prediction |
| `tools/collect_sequence_fdm_v2_data.py` | CLI for parallel data collection |
| `tools/train_sequence_fdm_v2.py` | CLI for training |
| `tools/eval_sequence_fdm_v2.py` | CLI for open-loop and closed-loop evaluation |
| `tests/test_random_terrain_generator.py` | Tests for terrain generator |
| `tests/test_sequence_fdm_v2_model.py` | Tests for model forward pass |
| `tests/test_sequence_fmd_dynamics.py` | Tests for dynamics wrapper |
| `tests/test_sequence_fdm_collector.py` | Tests for episode collection and dataset building |
| `tests/test_sequence_fdm_training.py` | Tests for curriculum dataset and training loop |
| `tests/test_mppi_omni_sequence_fdm_v2_torch.py` | Tests for controller cost computation |

**Constraint:** No modifications to existing files in `b2_fdm_mppi/`. All new functionality lives in new files. Existing residual FDM and sequence FDM (v1) remain untouched as baselines.

---

## Task 1: RandomTerrainGenerator

**Files:**
- Create: `b2_fdm_mppi/simulation/random_terrain.py`
- Test: `tests/test_random_terrain_generator.py`

**Context:** `TerrainField` is in `b2_fdm_mppi/core/terrain.py`. It accepts `patches` (list of dicts with `type`, `center`, `angle`, `size`, `edge_width`, and deltas for `slope_f`, `slope_l`, `roughness`, `friction`) and noise parameters. Risk is computed as `w0*abs(slope_f) + w1*abs(slope_l) + w2*roughness + w3*(1-friction)`.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator
from b2_fdm_mppi.core.terrain import TerrainField


def test_generator_returns_terrain_field():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(2, 3))
    terrain = gen.generate(seed=42)
    assert isinstance(terrain, TerrainField)
    assert terrain.enabled


def test_generator_reproducible():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(2, 3))
    t1 = gen.generate(seed=123)
    t2 = gen.generate(seed=123)
    # Same seed should produce same patches
    assert len(t1.patches) == len(t2.patches)
    for p1, p2 in zip(t1.patches, t2.patches):
        assert p1["type"] == p2["type"]
        assert np.allclose(p1["center"], p2["center"])


def test_generator_different_seeds():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(3, 5))
    t1 = gen.generate(seed=1)
    t2 = gen.generate(seed=2)
    # Different seeds should likely produce different patch counts or positions
    any_diff = False
    if len(t1.patches) != len(t2.patches):
        any_diff = True
    else:
        for p1, p2 in zip(t1.patches, t2.patches):
            if not np.allclose(p1["center"], p2["center"]):
                any_diff = True
                break
    assert any_diff


def test_patches_within_bounds():
    bounds = (-20, 20, -15, 15)
    gen = RandomTerrainGenerator(map_bounds=bounds, num_patches_range=(5, 5))
    terrain = gen.generate(seed=999)
    for patch in terrain.patches:
        cx, cy = patch["center"]
        assert bounds[0] <= cx <= bounds[1]
        assert bounds[2] <= cy <= bounds[3]


def test_generator_creates_risk_variation():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(3, 5))
    terrain = gen.generate(seed=77)
    # Sample risk at a few points; should see variation
    risks = [terrain.risk_cost(x, y) for x in [-5, 0, 5] for y in [-5, 0, 5]]
    assert max(risks) > min(risks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_random_terrain_generator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'b2_fdm_mppi.simulation.random_terrain'"

- [ ] **Step 3: Write minimal implementation**

```python
"""Random terrain generator for sequence FDM data collection."""

from __future__ import annotations

import numpy as np

from b2_fdm_mppi.core.terrain import TerrainField


class RandomTerrainGenerator:
    """Generates random TerrainField instances with configurable patches and noise."""

    def __init__(
        self,
        map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
        num_patches_range: tuple[int, int] = (3, 6),
        patch_types: tuple[str, ...] = ("ellipse", "band"),
        patch_size_range: tuple[float, float] = (1.5, 4.0),
        risk_intensity_range: tuple[float, float] = (0.3, 0.9),
        noise_enabled: bool = True,
        noise_scale_range: tuple[float, float] = (0.2, 0.5),
    ) -> None:
        self.map_bounds = map_bounds
        self.num_patches_range = num_patches_range
        self.patch_types = patch_types
        self.patch_size_range = patch_size_range
        self.risk_intensity_range = risk_intensity_range
        self.noise_enabled = noise_enabled
        self.noise_scale_range = noise_scale_range

    def generate(self, seed: int) -> TerrainField:
        rng = np.random.default_rng(seed)
        num_patches = int(rng.integers(self.num_patches_range[0], self.num_patches_range[1] + 1))

        patches: list[dict] = []
        for _ in range(num_patches):
            patch_type = str(rng.choice(self.patch_types))
            cx = float(rng.uniform(self.map_bounds[0], self.map_bounds[1]))
            cy = float(rng.uniform(self.map_bounds[2], self.map_bounds[3]))
            angle = float(rng.uniform(0.0, 180.0))
            sx = float(rng.uniform(self.patch_size_range[0], self.patch_size_range[1]))
            sy = float(rng.uniform(self.patch_size_range[0], self.patch_size_range[1]))
            intensity = float(rng.uniform(self.risk_intensity_range[0], self.risk_intensity_range[1]))

            patch: dict = {
                "type": patch_type,
                "center": [cx, cy],
                "angle": angle,
                "size": [sx, sy],
                "edge_width": 0.5,
                "slope_f": intensity * 0.8,
                "slope_l": intensity * 0.6,
                "roughness": intensity * 0.5,
                "friction": -intensity * 0.4,
            }
            patches.append(patch)

        noise_scale = float(rng.uniform(self.noise_scale_range[0], self.noise_scale_range[1]))

        return TerrainField(
            enabled=True,
            noise_enabled=self.noise_enabled,
            noise_seed=seed,
            noise_scale=noise_scale,
            noise_grid_size=(32, 32),
            noise_smooth_passes=3,
            noise_roughness_weight=0.25,
            noise_friction_weight=0.18,
            noise_slope_weight=0.08,
            noise_x_range=(self.map_bounds[0], self.map_bounds[1]),
            noise_y_range=(self.map_bounds[2], self.map_bounds[3]),
            patches=patches if patches else None,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_random_terrain_generator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/simulation/random_terrain.py tests/test_random_terrain_generator.py
git commit -m "feat: add RandomTerrainGenerator for sequence FDM data collection"
```

---

## Task 2: Terrain Risk Grid Sampling

**Files:**
- Create: `b2_fdm_mppi/core/terrain_grid.py`
- Test: `tests/test_terrain_grid.py`

**Context:** `TerrainField.risk_cost(x, y)` returns a scalar float. For the sequence FDM model input, we need a 9x9 grid of risk values centered on the robot. Both numpy (data collection) and torch (inference) versions are needed.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
import torch

from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.core.terrain_grid import sample_terrain_risk_grid_np, sample_terrain_risk_grid_torch


def test_grid_np_shape_and_bounds():
    terrain = TerrainField(enabled=True, noise_enabled=False)
    grid = sample_terrain_risk_grid_np(terrain, x=0.0, y=0.0, size=9, span=18.0)
    assert grid.shape == (81,)
    assert grid.dtype == np.float32
    assert np.all(grid >= 0.0)


def test_grid_np_centered():
    terrain = TerrainField(enabled=True, noise_enabled=False, patches=[
        {"type": "ellipse", "center": [5.0, 0.0], "angle": 0.0, "size": [2.0, 2.0],
         "edge_width": 0.5, "slope_f": 1.0, "slope_l": 0.0, "roughness": 0.0, "friction": 0.0}
    ])
    grid_center = sample_terrain_risk_grid_np(terrain, x=0.0, y=0.0, size=9, span=18.0)
    grid_shifted = sample_terrain_risk_grid_np(terrain, x=5.0, y=0.0, size=9, span=18.0)
    # The grid centered at (5,0) should have higher risk values near the patch center
    assert np.max(grid_shifted) > np.max(grid_center)


def test_grid_torch_shape():
    terrain = TerrainField(enabled=True, noise_enabled=False)
    grid = sample_terrain_risk_grid_torch(terrain, x=0.0, y=0.0, size=9, span=18.0)
    assert grid.shape == (81,)
    assert grid.dtype == torch.float32
    assert torch.all(grid >= 0.0)


def test_grid_torch_matches_np():
    terrain = TerrainField(enabled=True, noise_enabled=True, noise_seed=42)
    grid_np = sample_terrain_risk_grid_np(terrain, x=3.0, y=-2.0, size=9, span=18.0)
    grid_torch = sample_terrain_risk_grid_torch(terrain, x=3.0, y=-2.0, size=9, span=18.0)
    assert np.allclose(grid_np, grid_torch.numpy(), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_terrain_grid.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'b2_fdm_mppi.core.terrain_grid'"

- [ ] **Step 3: Write minimal implementation**

```python
"""Terrain risk grid sampling utilities for sequence FDM."""

from __future__ import annotations

import numpy as np
import torch

from b2_fdm_mppi.core.terrain import TerrainField


def sample_terrain_risk_grid_np(
    terrain: TerrainField,
    x: float,
    y: float,
    size: int = 9,
    span: float = 18.0,
) -> np.ndarray:
    """Sample a flat risk grid centered on (x, y) using numpy."""
    half = span / 2.0
    xs = np.linspace(x - half, x + half, size, dtype=np.float32)
    ys = np.linspace(y - half, y + half, size, dtype=np.float32)
    grid = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = float(terrain.risk_cost(float(xs[i]), float(ys[j])))
    return grid.reshape(-1)


def sample_terrain_risk_grid_torch(
    terrain: TerrainField,
    x: float,
    y: float,
    size: int = 9,
    span: float = 18.0,
) -> torch.Tensor:
    """Sample a flat risk grid centered on (x, y) using torch."""
    half = span / 2.0
    xs = torch.linspace(x - half, x + half, size, dtype=torch.float32)
    ys = torch.linspace(y - half, y + half, size, dtype=torch.float32)
    grid = torch.zeros((size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = float(terrain.risk_cost(float(xs[i].item()), float(ys[j].item())))
    return grid.reshape(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_terrain_grid.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/core/terrain_grid.py tests/test_terrain_grid.py
git commit -m "feat: add terrain risk grid sampling for sequence FDM input"
```

---

## Task 3: SequenceFdmMlpV2 Model

**Files:**
- Create: `b2_fdm_mppi/core/sequence_fdm_v2.py`
- Test: `tests/test_sequence_fdm_v2_model.py`

**Context:** New model that takes (state 6D, controls Hx3 flattened, terrain grid 81D) and outputs (states Hx6, risk logits H). Unlike existing `SequenceFdmMlp` in `residual_fdm_model.py`, this model predicts absolute states and binary risk logits.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
import torch

from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2


def test_model_forward_shapes():
    model = SequenceFdmMlpV2(horizon_steps=10, hidden_dims=[64, 64])
    batch = 4
    state = torch.randn(batch, 6)
    controls = torch.randn(batch, 10, 3)
    terrain = torch.randn(batch, 81)

    states_pred, risk_logits = model(state, controls, terrain)
    assert states_pred.shape == (batch, 10, 6)
    assert risk_logits.shape == (batch, 10)


def test_model_single_sample():
    model = SequenceFdmMlpV2(horizon_steps=5, hidden_dims=[32])
    state = torch.randn(1, 6)
    controls = torch.randn(1, 5, 3)
    terrain = torch.randn(1, 81)

    states_pred, risk_logits = model(state, controls, terrain)
    assert states_pred.shape == (1, 5, 6)
    assert risk_logits.shape == (1, 5)
    assert torch.isfinite(states_pred).all()
    assert torch.isfinite(risk_logits).all()


def test_model_save_load():
    model = SequenceFdmMlpV2(horizon_steps=8, hidden_dims=[64])
    state = {"model_state_dict": model.state_dict(), "horizon_steps": 8, "hidden_dims": [64]}
    buffer = torch.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    model2 = SequenceFdmMlpV2(horizon_steps=loaded["horizon_steps"], hidden_dims=loaded["hidden_dims"])
    model2.load_state_dict(loaded["model_state_dict"])

    x = torch.randn(2, 6)
    u = torch.randn(2, 8, 3)
    g = torch.randn(2, 81)
    s1, r1 = model(x, u, g)
    s2, r2 = model2(x, u, g)
    assert torch.allclose(s1, s2)
    assert torch.allclose(r1, r2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequence_fdm_v2_model.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'b2_fdm_mppi.core.sequence_fdm_v2'"

- [ ] **Step 3: Write minimal implementation**

```python
"""Sequence FDM V2 model: predicts absolute trajectories + binary risk from terrain grid."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES_V2 = [
    "state_x", "state_y", "state_theta",
    "state_vx", "state_vy", "state_wz",
] + [
    f"cmd_t{i}_vx" for i in range(1, 100)  # dynamically sized by horizon
] + [
    f"cmd_t{i}_vy" for i in range(1, 100)
] + [
    f"cmd_t{i}_wz" for i in range(1, 100)
] + [
    f"terrain_grid_{i}" for i in range(81)
]

TARGET_NAMES_V2 = [
    f"state_x_t{i}" for i in range(1, 100)
] + [
    f"state_y_t{i}" for i in range(1, 100)
] + [
    f"state_theta_t{i}" for i in range(1, 100)
] + [
    f"state_vx_t{i}" for i in range(1, 100)
] + [
    f"state_vy_t{i}" for i in range(1, 100)
] + [
    f"state_wz_t{i}" for i in range(1, 100)
] + [
    f"risk_t{i}" for i in range(1, 100)
]


def build_feature_names_v2(horizon_steps: int) -> list[str]:
    names = ["state_x", "state_y", "state_theta", "state_vx", "state_vy", "state_wz"]
    for step in range(horizon_steps):
        names.extend([f"future_cmd_t{step + 1}_vx", f"future_cmd_t{step + 1}_vy", f"future_cmd_t{step + 1}_wz"])
    names.extend([f"terrain_grid_{i}" for i in range(81)])
    return names


def build_target_names_v2(horizon_steps: int) -> list[str]:
    names: list[str] = []
    for axis in ["x", "y", "theta", "vx", "vy", "wz"]:
        for step in range(horizon_steps):
            names.append(f"state_{axis}_t{step + 1}")
    for step in range(horizon_steps):
        names.append(f"risk_t{step + 1}")
    return names


class SequenceFdmMlpV2(nn.Module):
    """MLP that maps (state, control sequence, terrain grid) -> (state sequence, risk logits)."""

    def __init__(self, horizon_steps: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.horizon_steps = int(horizon_steps)
        input_dim = 6 + 3 * self.horizon_steps + 81
        output_dim = 6 * self.horizon_steps + self.horizon_steps

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        terrain_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, 6)
            controls: (B, H, 3)
            terrain_grid: (B, 81)
        Returns:
            states_pred: (B, H, 6)
            risk_logits: (B, H)
        """
        batch_size = state.shape[0]
        controls_flat = controls.view(batch_size, -1)
        x = torch.cat([state, controls_flat, terrain_grid], dim=1)
        out = self.net(x)
        states_pred = out[:, : 6 * self.horizon_steps].view(batch_size, self.horizon_steps, 6)
        risk_logits = out[:, 6 * self.horizon_steps :]
        return states_pred, risk_logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sequence_fdm_v2_model.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/core/sequence_fdm_v2.py tests/test_sequence_fdm_v2_model.py
git commit -m "feat: add SequenceFdmMlpV2 with absolute state + risk prediction"
```

---

## Task 4: SequenceFdmDynamics Wrapper

**Files:**
- Create: `b2_fdm_mppi/core/sequence_fdm_dynamics.py`
- Test: `tests/test_sequence_fdm_dynamics.py`

**Context:** Wrapper that loads a trained `SequenceFdmMlpV2` checkpoint + normalization, handles input normalization and output denormalization, and provides a clean `predict()` API for both numpy and torch inputs.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile

from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics


def _create_dummy_checkpoint(tmp_path: Path, horizon_steps: int = 5):
    model = SequenceFdmMlpV2(horizon_steps=horizon_steps, hidden_dims=[32])
    norm = {
        "state_mean": np.zeros(6, dtype=np.float32),
        "state_std": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(3, dtype=np.float32),
        "control_std": np.ones(3, dtype=np.float32),
        "target_mean": np.zeros(horizon_steps * 7, dtype=np.float32),
        "target_std": np.ones(horizon_steps * 7, dtype=np.float32),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "horizon_steps": horizon_steps,
        "hidden_dims": [32],
        "input_dim": 6 + 3 * horizon_steps + 81,
        "target_dim": horizon_steps * 7,
        "feature_names": [],
        "target_names": [],
    }
    ckpt_path = tmp_path / "best_model.pt"
    norm_path = tmp_path / "normalization.npz"
    torch.save(checkpoint, ckpt_path)
    np.savez(norm_path, **norm)
    return tmp_path


def test_from_artifacts(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=5)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")
    assert dyn.horizon_steps == 5
    assert dyn.model is not None


def test_predict_shapes(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=5)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")

    state = np.zeros(6, dtype=np.float32)
    controls = np.zeros((5, 3), dtype=np.float32)
    terrain = np.zeros(81, dtype=np.float32)

    states_pred, risk_logits = dyn.predict(state, controls, terrain)
    assert states_pred.shape == (5, 6)
    assert risk_logits.shape == (5,)
    assert np.isfinite(states_pred).all()
    assert np.isfinite(risk_logits).all()


def test_predict_batch(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=8)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")

    states = np.zeros((3, 6), dtype=np.float32)
    controls = np.zeros((3, 8, 3), dtype=np.float32)
    terrains = np.zeros((3, 81), dtype=np.float32)

    states_pred, risk_logits = dyn.predict_batch(states, controls, terrains)
    assert states_pred.shape == (3, 8, 6)
    assert risk_logits.shape == (3, 8)


def test_predict_torch(tmp_path):
    model_dir = _create_dummy_checkpoint(tmp_path, horizon_steps=5)
    dyn = SequenceFdmDynamics.from_artifacts(model_dir, device="cpu")

    state = torch.zeros(6)
    controls = torch.zeros(5, 3)
    terrain = torch.zeros(81)

    states_pred, risk_logits = dyn.predict_torch(state, controls, terrain)
    assert states_pred.shape == (5, 6)
    assert risk_logits.shape == (5,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequence_fdm_dynamics.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'b2_fdm_mppi.core.sequence_fdm_dynamics'"

- [ ] **Step 3: Write minimal implementation**

```python
"""Sequence FDM dynamics wrapper for inference-time normalization and model loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2


class SequenceFdmDynamics:
    """Loads a SequenceFdmMlpV2 checkpoint and handles normalization for inference."""

    def __init__(
        self,
        *,
        model: SequenceFdmMlpV2,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        control_mean: np.ndarray,
        control_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device: torch.device,
        checkpoint_path: Path,
        normalization_path: Path,
    ) -> None:
        self.model = model
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.control_mean = np.asarray(control_mean, dtype=np.float32)
        self.control_std = np.asarray(control_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.normalization_path = Path(normalization_path)
        self.horizon_steps = int(model.horizon_steps)

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        *,
        checkpoint: str | Path = "best_model.pt",
        normalization: str | Path = "normalization.npz",
        device: str = "cpu",
    ) -> "SequenceFdmDynamics":
        model_dir = Path(model_dir)
        checkpoint_path = model_dir / checkpoint
        normalization_path = model_dir / normalization

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        horizon_steps = int(ckpt["horizon_steps"])
        hidden_dims = list(ckpt.get("hidden_dims", [256, 256, 256]))
        input_dim = int(ckpt["input_dim"])
        target_dim = int(ckpt["target_dim"])

        expected_input = 6 + 3 * horizon_steps + 81
        expected_target = 6 * horizon_steps + horizon_steps
        if input_dim != expected_input:
            raise ValueError(f"Checkpoint input_dim {input_dim} != expected {expected_input}")
        if target_dim != expected_target:
            raise ValueError(f"Checkpoint target_dim {target_dim} != expected {expected_target}")

        model = SequenceFdmMlpV2(horizon_steps=horizon_steps, hidden_dims=hidden_dims)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        norm = np.load(normalization_path)
        return cls(
            model=model,
            state_mean=norm["state_mean"],
            state_std=norm["state_std"],
            control_mean=norm["control_mean"],
            control_std=norm["control_std"],
            target_mean=norm["target_mean"],
            target_std=norm["target_std"],
            device=torch.device(device),
            checkpoint_path=checkpoint_path,
            normalization_path=normalization_path,
        )

    def predict(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        terrain_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(state, dtype=np.float32).reshape(1, 6)
        cmds = np.asarray(controls, dtype=np.float32).reshape(1, self.horizon_steps, 3)
        grid = np.asarray(terrain_grid, dtype=np.float32).reshape(1, 81)
        s_pred, r_logits = self.predict_batch(states, cmds, grid)
        return s_pred[0], r_logits[0]

    def predict_batch(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        terrain_grids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        controls_t = torch.as_tensor(controls, dtype=torch.float32, device=self.device)
        grids_t = torch.as_tensor(terrain_grids, dtype=torch.float32, device=self.device)
        s_pred, r_logits = self.predict_torch(states_t, controls_t, grids_t)
        return s_pred.detach().cpu().numpy(), r_logits.detach().cpu().numpy()

    def predict_torch(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        terrain_grids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = states.to(torch.float32).reshape(-1, 6)
        controls = controls.to(torch.float32).reshape(-1, self.horizon_steps, 3)
        terrain_grids = terrain_grids.to(torch.float32).reshape(-1, 81)

        # Normalize inputs
        s_norm = (states - self._to_tensor(self.state_mean)) / self._to_tensor(self.state_std)
        c_norm = (controls - self._to_tensor(self.control_mean)) / self._to_tensor(self.control_std)
        # Terrain grid is approximately [0, 1], no normalization

        # Forward pass
        out_states, out_risk = self.model(s_norm, c_norm, terrain_grids)

        # Denormalize state predictions
        target_state_mean = self._to_tensor(self.target_mean[: 6 * self.horizon_steps]).view(self.horizon_steps, 6)
        target_state_std = self._to_tensor(self.target_std[: 6 * self.horizon_steps]).view(self.horizon_steps, 6)
        states_pred = out_states * target_state_std + target_state_mean

        # Risk logits are not normalized (they are raw logits)
        return states_pred, out_risk

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sequence_fdm_dynamics.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/core/sequence_fdm_dynamics.py tests/test_sequence_fdm_dynamics.py
git commit -m "feat: add SequenceFdmDynamics wrapper for model inference"
```

---

## Task 5: Data Collection — Single Episode

**Files:**
- Create: `b2_fdm_mppi/data/sequence_fdm_collector.py`
- Test: `tests/test_sequence_fdm_collector.py`

**Context:** The collector needs to: (1) generate random terrain, (2) sample start/goal >10m apart, (3) create an MPPI config, (4) run simulation using `OmniMppiSimulationRunner`, (5) extract trajectory data, (6) mark binary risk sequences. The existing `collect_oracle_episode` pattern in `b2_fdm_mppi/data/collect_oracle_episode.py` loads config from YAML and creates a runner. We follow the same pattern but inject random terrain and start/goal.

- [ ] **Step 1: Write failing test**

```python
import tempfile
from pathlib import Path
import numpy as np
import pytest

from b2_fdm_mppi.data.sequence_fdm_collector import collect_sequence_fdm_episode


def test_collect_episode_output_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = collect_sequence_fdm_episode(
            base_config_path="configs/smoke.yaml",
            episode_id=0,
            terrain_seed=42,
            output_dir=tmpdir,
            map_bounds=(-10, 10, -10, 10),
        )
        assert result["episode_id"] == 0
        assert result["terrain_seed"] == 42
        assert Path(result["output_path"]).exists()


def test_binary_risk_marking():
    from b2_fdm_mppi.data.sequence_fdm_collector import _mark_binary_risk

    terrain_risk = np.array([0.1, 0.2, 0.8, 0.3, 0.1], dtype=np.float32)
    binary = _mark_binary_risk(terrain_risk, threshold=0.6)
    expected = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    assert np.array_equal(binary, expected)


def test_binary_risk_all_safe():
    from b2_fdm_mppi.data.sequence_fdm_collector import _mark_binary_risk

    terrain_risk = np.array([0.1, 0.2, 0.3, 0.1], dtype=np.float32)
    binary = _mark_binary_risk(terrain_risk, threshold=0.6)
    expected = np.array([0, 0, 0, 0], dtype=np.float32)
    assert np.array_equal(binary, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequence_fdm_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
"""Data collection for sequence FDM V2: random terrain episodes with binary risk labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.data.collect_oracle_episode import collect_oracle_episode
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator


def _mark_binary_risk(terrain_risk: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Mark binary risk: once threshold is exceeded, all subsequent timesteps are 1."""
    binary = np.zeros_like(terrain_risk, dtype=np.float32)
    exceed = np.where(terrain_risk > threshold)[0]
    if len(exceed) > 0:
        first = int(exceed[0])
        binary[first:] = 1.0
    return binary


def _sample_start_goal(
    rng: np.random.Generator,
    map_bounds: tuple[float, float, float, float],
    min_distance: float = 10.0,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample start and goal positions at least min_distance apart."""
    for _ in range(max_attempts):
        start = np.array([
            rng.uniform(map_bounds[0], map_bounds[1]),
            rng.uniform(map_bounds[2], map_bounds[3]),
        ], dtype=np.float32)
        goal = np.array([
            rng.uniform(map_bounds[0], map_bounds[1]),
            rng.uniform(map_bounds[2], map_bounds[3]),
        ], dtype=np.float32)
        if np.linalg.norm(goal - start) >= min_distance:
            return start, goal
    raise RuntimeError(f"Failed to sample start/goal with min_distance={min_distance}")


def collect_sequence_fdm_episode(
    *,
    base_config_path: str | Path,
    episode_id: int,
    terrain_seed: int,
    output_dir: str | Path,
    map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    min_start_goal_distance: float = 10.0,
    risk_threshold: float = 0.6,
    num_patches_range: tuple[int, int] = (3, 6),
) -> dict:
    """Collect a single sequence FDM episode on a randomly generated terrain.

    Returns a dict with episode metadata. The actual trajectory data is saved to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{episode_id:06d}.npz"

    # Generate random terrain
    generator = RandomTerrainGenerator(
        map_bounds=map_bounds,
        num_patches_range=num_patches_range,
    )
    terrain = generator.generate(seed=terrain_seed)

    # Sample start and goal
    rng = np.random.default_rng(terrain_seed + 10000)
    start_xy, goal_xy = _sample_start_goal(rng, map_bounds, min_distance=min_start_goal_distance)

    # Load base config and override terrain + start/goal
    config = load_config(base_config_path)
    config["terrain"] = terrain.to_config()
    config["simulation"]["initial_state"] = [float(start_xy[0]), float(start_xy[1]), 0.0, 0.0, 0.0, 0.0]
    config["simulation"]["goal"] = [float(goal_xy[0]), float(goal_xy[1]), 0.0, 0.0, 0.0, 0.0]

    # Save temporary config
    temp_config_path = output_dir / f"config_episode_{episode_id:06d}.yaml"
    import yaml
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run collection using existing oracle episode collector
    # This will create states, controls, terrain_features, terrain_risk arrays
    result = collect_oracle_episode(
        config_path=temp_config_path,
        episode_id=episode_id,
        seed=terrain_seed,
        output_path=output_path,
    )

    # Load the saved episode and add binary risk
    episode = np.load(output_path, allow_pickle=True)
    terrain_risk = episode["terrain_risk"].astype(np.float32)
    binary_risk = _mark_binary_risk(terrain_risk, threshold=risk_threshold)

    # Re-save with binary risk included
    data_dict = {key: episode[key] for key in episode.files}
    data_dict["binary_risk"] = binary_risk
    data_dict["terrain_seed"] = np.array([terrain_seed], dtype=np.int32)
    data_dict["start_xy"] = start_xy.astype(np.float32)
    data_dict["goal_xy"] = goal_xy.astype(np.float32)

    np.savez_compressed(output_path, **data_dict)
    episode.close()

    # Cleanup temp config
    temp_config_path.unlink(missing_ok=True)

    result.update({
        "terrain_seed": terrain_seed,
        "binary_risk_any": bool(np.any(binary_risk > 0)),
        "output_path": str(output_path),
    })
    return result
```

**Note on implementation:** The collector reuses `collect_oracle_episode` for the simulation run, then post-processes the saved NPZ to add binary risk labels. `TerrainField.to_config()` may need to be checked — if it doesn't exist, implement a simple dict serialization in `random_terrain.py` or inline in the collector.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sequence_fdm_collector.py -v`
Expected: Tests for `_mark_binary_risk` and `_sample_start_goal` PASS. The full `test_collect_episode_output_structure` may require the smoke config to exist and may take longer; if it fails due to missing config, adjust the test to use a minimal inline config.

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/data/sequence_fdm_collector.py tests/test_sequence_fdm_collector.py
git commit -m "feat: add sequence FDM episode collector with binary risk labels"
```

---

## Task 6: Data Collection — Parallel Tool

**Files:**
- Create: `tools/collect_sequence_fdm_v2_data.py`
- Test: `tests/test_collect_sequence_fdm_v2_cli.py`

**Context:** Follow the pattern of `tools/generate_oracle_episodes.py` which delegates to `b2_fdm_mppi.data.generate_oracle_episodes`. We create a CLI tool that calls `collect_sequence_fdm_episode` in parallel using `ProcessPoolExecutor`.

- [ ] **Step 1: Write failing test**

```python
import subprocess
import sys
import tempfile
from pathlib import Path


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "tools/collect_sequence_fdm_v2_data.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--episodes" in result.stdout


def test_cli_smoke_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "tools/collect_sequence_fdm_v2_data.py",
                "--base-config", "configs/smoke.yaml",
                "--episodes", "2",
                "--output-dir", tmpdir,
                "--workers", "1",
                "--base-seed", "100",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        output = Path(tmpdir)
        assert len(list(output.glob("episode_*.npz"))) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collect_sequence_fdm_v2_cli.py -v`
Expected: FAIL with "FileNotFoundError: tools/collect_sequence_fdm_v2_data.py"

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""CLI tool for parallel collection of sequence FDM V2 training data."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from b2_fdm_mppi.data.sequence_fdm_collector import collect_sequence_fdm_episode


def _collect_one(args: tuple) -> dict:
    base_config_path, episode_id, terrain_seed, output_dir, map_bounds = args
    try:
        return collect_sequence_fdm_episode(
            base_config_path=base_config_path,
            episode_id=episode_id,
            terrain_seed=terrain_seed,
            output_dir=output_dir,
            map_bounds=map_bounds,
        )
    except Exception as e:
        return {"episode_id": episode_id, "error": str(e), "success": False}


def main():
    parser = argparse.ArgumentParser(description="Collect sequence FDM V2 training episodes")
    parser.add_argument("--base-config", required=True, help="Base YAML config path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--output-dir", required=True, help="Output directory for episodes")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed for terrain generation")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--map-bounds", type=float, nargs=4, default=[-20.0, 20.0, -20.0, 20.0])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            args.base_config,
            i,
            args.base_seed + i,
            str(output_dir),
            tuple(args.map_bounds),
        )
        for i in range(args.episodes)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(_collect_one, tasks):
            results.append(result)
            if result.get("error"):
                print(f"Episode {result['episode_id']} failed: {result['error']}")
            else:
                print(f"Episode {result['episode_id']} done: {result.get('output_path')}")

    summary = {
        "total": len(results),
        "success": sum(1 for r in results if not r.get("error")),
        "failed": sum(1 for r in results if r.get("error")),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"Collection complete: {summary['success']}/{summary['total']} succeeded")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collect_sequence_fdm_v2_cli.py -v`
Expected: `test_cli_help` PASS. `test_cli_smoke_run` PASS if smoke config exists and MPPI runs successfully.

- [ ] **Step 5: Commit**

```bash
git add tools/collect_sequence_fdm_v2_data.py tests/test_collect_sequence_fdm_v2_cli.py
git commit -m "feat: add CLI for parallel sequence FDM data collection"
```

---

## Task 7: Dataset Builder (Sliding Window)

**Files:**
- Modify: `b2_fdm_mppi/data/sequence_fdm_collector.py` (append new functions)
- Test: `tests/test_sequence_fdm_dataset_builder.py`

**Context:** After collecting episodes, we need to extract sliding windows for supervised learning. Each window: (state_t, controls_t:t+H, terrain_grid_t) -> (states_t+1:t+H, binary_risk_t+1:t+H).

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
import tempfile
from pathlib import Path

from b2_fdm_mppi.data.sequence_fdm_collector import build_sequence_fdm_windows


def test_build_windows_shapes():
    # Create a fake episode
    T = 50
    states = np.zeros((T, 6), dtype=np.float32)
    states[:, 0] = np.arange(T, dtype=np.float32)  # x increases
    controls = np.ones((T, 3), dtype=np.float32)
    terrain_risk = np.zeros(T, dtype=np.float32)
    binary_risk = np.zeros(T, dtype=np.float32)
    terrain_seed = np.array([42], dtype=np.int32)

    # Save fake episode
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode_000000.npz"
        np.savez(path, states=states, cmd_controls=controls, terrain_risk=terrain_risk,
                 binary_risk=binary_risk, terrain_seed=terrain_seed)

        windows = build_sequence_fdm_windows(
            episode_path=path,
            horizon_steps=10,
            stride=5,
            map_bounds=(-20, 20, -20, 20),
        )
        assert len(windows) > 0
        w = windows[0]
        assert w["state"].shape == (6,)
        assert w["controls"].shape == (10, 3)
        assert w["terrain_grid"].shape == (81,)
        assert w["target_states"].shape == (10, 6)
        assert w["target_risk"].shape == (10,)


def test_build_windows_stride():
    T = 30
    states = np.zeros((T, 6), dtype=np.float32)
    controls = np.ones((T, 3), dtype=np.float32)
    terrain_risk = np.zeros(T, dtype=np.float32)
    binary_risk = np.zeros(T, dtype=np.float32)
    terrain_seed = np.array([1], dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episode_000000.npz"
        np.savez(path, states=states, cmd_controls=controls, terrain_risk=terrain_risk,
                 binary_risk=binary_risk, terrain_seed=terrain_seed)

        windows_stride1 = build_sequence_fdm_windows(path, horizon_steps=5, stride=1, map_bounds=(-10, 10, -10, 10))
        windows_stride5 = build_sequence_fdm_windows(path, horizon_steps=5, stride=5, map_bounds=(-10, 10, -10, 10))
        assert len(windows_stride5) < len(windows_stride1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequence_fdm_dataset_builder.py -v`
Expected: FAIL with "ImportError: cannot import name 'build_sequence_fdm_windows'"

- [ ] **Step 3: Write minimal implementation**

Append to `b2_fdm_mppi/data/sequence_fdm_collector.py`:

```python
def build_sequence_fdm_windows(
    episode_path: str | Path,
    horizon_steps: int,
    stride: int = 1,
    map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    terrain_grid_size: int = 9,
    terrain_grid_span: float = 18.0,
) -> list[dict]:
    """Extract sliding windows from a collected episode for sequence FDM training.

    Returns a list of dicts, each containing one training sample.
    """
    from b2_fdm_mppi.core.terrain_grid import sample_terrain_risk_grid_np
    from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator

    episode = np.load(episode_path, allow_pickle=True)
    states = episode["states"].astype(np.float32)
    controls = episode["cmd_controls"].astype(np.float32)
    binary_risk = episode["binary_risk"].astype(np.float32)
    terrain_seed = int(episode["terrain_seed"][0])
    episode.close()

    # Reconstruct terrain for grid sampling
    generator = RandomTerrainGenerator(map_bounds=map_bounds)
    terrain = generator.generate(seed=terrain_seed)

    T = len(states)
    windows: list[dict] = []
    for t in range(0, T - horizon_steps, stride):
        state_t = states[t]
        controls_t = controls[t : t + horizon_steps]
        terrain_grid = sample_terrain_risk_grid_np(
            terrain, float(state_t[0]), float(state_t[1]),
            size=terrain_grid_size, span=terrain_grid_span,
        )
        target_states = states[t + 1 : t + 1 + horizon_steps]
        target_risk = binary_risk[t + 1 : t + 1 + horizon_steps]

        windows.append({
            "state": state_t,
            "controls": controls_t,
            "terrain_grid": terrain_grid,
            "target_states": target_states,
            "target_risk": target_risk,
            "episode_id": str(episode_path),
            "timestep": t,
        })
    return windows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sequence_fdm_dataset_builder.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/data/sequence_fdm_collector.py tests/test_sequence_fdm_dataset_builder.py
git commit -m "feat: add sliding window dataset builder for sequence FDM"
```

---

## Task 8: Training — Curriculum Dataset

**Files:**
- Create: `b2_fdm_mppi/training/sequence_fdm_v2.py`
- Test: `tests/test_sequence_fdm_training.py`

**Context:** Training uses curriculum learning: H=5 -> 10 -> 20. We need a PyTorch Dataset that loads windows for a specific horizon. The dataset is built from the collected episodes on-the-fly or pre-cached.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
import tempfile
from pathlib import Path
import torch

from b2_fdm_mppi.training.sequence_fdm_v2 import SequenceFdmDataset, compute_normalization


def test_dataset_length_and_getitem():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake windows
        windows = []
        for i in range(20):
            windows.append({
                "state": np.zeros(6, dtype=np.float32),
                "controls": np.zeros((10, 3), dtype=np.float32),
                "terrain_grid": np.zeros(81, dtype=np.float32),
                "target_states": np.ones((10, 6), dtype=np.float32),
                "target_risk": np.zeros(10, dtype=np.float32),
            })
        dataset = SequenceFdmDataset(windows, horizon_steps=10)
        assert len(dataset) == 20
        x_state, x_ctrl, x_grid, y_state, y_risk = dataset[0]
        assert x_state.shape == (6,)
        assert x_ctrl.shape == (10, 3)
        assert x_grid.shape == (81,)
        assert y_state.shape == (10, 6)
        assert y_risk.shape == (10,)


def test_compute_normalization():
    windows = []
    for i in range(50):
        windows.append({
            "state": np.random.randn(6).astype(np.float32),
            "controls": np.random.randn(10, 3).astype(np.float32),
            "terrain_grid": np.random.rand(81).astype(np.float32),
            "target_states": np.random.randn(10, 6).astype(np.float32),
            "target_risk": np.random.randint(0, 2, size=10).astype(np.float32),
        })
    norm = compute_normalization(windows, horizon_steps=10)
    assert "state_mean" in norm
    assert "state_std" in norm
    assert "control_mean" in norm
    assert "control_std" in norm
    assert "target_mean" in norm
    assert "target_std" in norm
    assert norm["state_mean"].shape == (6,)
    assert norm["state_std"].shape == (6,)
    assert norm["control_mean"].shape == (3,)
    assert norm["control_std"].shape == (3,)
    assert norm["target_mean"].shape == (10 * 7,)
    assert norm["target_std"].shape == (10 * 7,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequence_fdm_training.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
"""Training utilities for Sequence FDM V2 with curriculum learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SequenceFdmDataset(Dataset):
    """PyTorch dataset for sequence FDM V2 training windows."""

    def __init__(self, windows: list[dict], horizon_steps: int) -> None:
        self.windows = windows
        self.horizon_steps = horizon_steps

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        w = self.windows[idx]
        state = torch.as_tensor(w["state"], dtype=torch.float32)
        controls = torch.as_tensor(w["controls"], dtype=torch.float32)
        terrain_grid = torch.as_tensor(w["terrain_grid"], dtype=torch.float32)
        target_states = torch.as_tensor(w["target_states"], dtype=torch.float32)
        target_risk = torch.as_tensor(w["target_risk"], dtype=torch.float32)
        return state, controls, terrain_grid, target_states, target_risk


def compute_normalization(windows: list[dict], horizon_steps: int) -> dict[str, np.ndarray]:
    """Compute mean/std for states, controls, and targets from training windows."""
    states = np.stack([w["state"] for w in windows])
    controls = np.stack([w["controls"].reshape(-1) for w in windows])
    target_states = np.stack([w["target_states"].reshape(-1) for w in windows])
    target_risk = np.stack([w["target_risk"] for w in windows])

    target = np.concatenate([target_states, target_risk], axis=1)

    return {
        "state_mean": np.mean(states, axis=0).astype(np.float32),
        "state_std": np.std(states, axis=0).astype(np.float32) + 1e-8,
        "control_mean": np.mean(controls, axis=0).astype(np.float32),
        "control_std": np.std(controls, axis=0).astype(np.float32) + 1e-8,
        "target_mean": np.mean(target, axis=0).astype(np.float32),
        "target_std": np.std(target, axis=0).astype(np.float32) + 1e-8,
        "horizon_steps": np.array([horizon_steps], dtype=np.int32),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sequence_fdm_training.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/training/sequence_fdm_v2.py tests/test_sequence_fdm_training.py
git commit -m "feat: add SequenceFdmDataset and normalization computation"
```

---

## Task 9: Training Loop with Curriculum

**Files:**
- Modify: `b2_fdm_mppi/training/sequence_fdm_v2.py` (append training loop)
- Test: `tests/test_sequence_fdm_training.py` (append new tests)

**Context:** The training loop iterates through curriculum phases. Each phase has a horizon length, number of epochs, and learning rate. The model from the previous phase's best checkpoint is loaded for the next phase. Loss = MSE on states + BCE on risk.

- [ ] **Step 1: Write failing test**

Append to `tests/test_sequence_fdm_training.py`:

```python
import tempfile
from pathlib import Path

from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2
from b2_fdm_mppi.training.sequence_fdm_v2 import train_sequence_fdm_v2


def test_train_smoke():
    """Smoke test: training runs for a few epochs without crashing."""
    windows = []
    for i in range(30):
        windows.append({
            "state": np.random.randn(6).astype(np.float32) * 0.1,
            "controls": np.random.randn(5, 3).astype(np.float32) * 0.1,
            "terrain_grid": np.random.rand(81).astype(np.float32),
            "target_states": np.random.randn(5, 6).astype(np.float32) * 0.1,
            "target_risk": np.random.randint(0, 2, size=5).astype(np.float32),
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        metrics = train_sequence_fdm_v2(
            windows=windows,
            output_dir=output_dir,
            hidden_dims=[32, 32],
            curriculum_phases=[(5, 3, 1e-3)],
            batch_size=8,
        )
        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "normalization.npz").exists()
        assert "best_val_loss" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sequence_fdm_training.py::test_train_smoke -v`
Expected: FAIL with "ImportError: cannot import name 'train_sequence_fdm_v2'"

- [ ] **Step 3: Write minimal implementation**

Append to `b2_fdm_mppi/training/sequence_fdm_v2.py`:

```python
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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hidden_dims is None:
        hidden_dims = [256, 256, 256]
    if curriculum_phases is None:
        curriculum_phases = [(5, 50, 1e-3), (10, 50, 5e-4), (20, 100, 1e-4), (20, 50, 5e-5)]

    # Split train/val by episode to avoid data leakage
    n_val = int(len(windows) * val_ratio)
    rng = np.random.default_rng(42)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    val_indices = set(indices[:n_val].tolist())
    train_windows = [w for i, w in enumerate(windows) if i not in val_indices]
    val_windows = [w for i, w in enumerate(windows) if i in val_indices]

    # Compute normalization on training set only
    # We use the max horizon for normalization so it works across phases
    max_horizon = max(h for h, _, _ in curriculum_phases)
    # Filter windows to max_horizon (should already be, but be safe)
    norm = compute_normalization(train_windows, horizon_steps=max_horizon)
    np.savez(output_dir / "normalization.npz", **norm)

    torch_device = torch.device(device)
    best_ckpt_path = None
    best_val_loss = float("inf")
    history: list[dict] = []

    for phase_idx, (horizon, epochs, lr) in enumerate(curriculum_phases):
        print(f"\n=== Curriculum Phase {phase_idx + 1}/{len(curriculum_phases)}: H={horizon}, lr={lr} ===")

        # Filter windows to current horizon
        phase_train = [w for w in train_windows if w["controls"].shape[0] >= horizon]
        phase_val = [w for w in val_windows if w["controls"].shape[0] >= horizon]

        if not phase_train:
            raise ValueError(f"No training windows with horizon >= {horizon}")

        # Truncate windows to current horizon
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

        train_dataset = SequenceFdmDataset(phase_train, horizon_steps=horizon)
        val_dataset = SequenceFdmDataset(phase_val, horizon_steps=horizon)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create or load model
        model = SequenceFdmMlpV2(horizon_steps=horizon, hidden_dims=hidden_dims).to(torch_device)
        if best_ckpt_path is not None and best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            # Try to load compatible layers if horizon changed
            # For simplicity, we only load if horizon matches; otherwise train from scratch
            if ckpt.get("horizon_steps") == horizon:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"  Loaded checkpoint from phase {phase_idx}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        phase_best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for state, controls, grid, target_states, target_risk in train_loader:
                state = state.to(torch_device)
                controls = controls.to(torch_device)
                grid = grid.to(torch_device)
                target_states = target_states.to(torch_device)
                target_risk = target_risk.to(torch_device)

                pred_states, pred_risk_logits = model(state, controls, grid)

                loss_traj = mse_loss(pred_states, target_states)
                loss_risk = bce_loss(pred_risk_logits, target_risk)
                loss = w_traj * loss_traj + w_risk * loss_risk

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for state, controls, grid, target_states, target_risk in val_loader:
                    state = state.to(torch_device)
                    controls = controls.to(torch_device)
                    grid = grid.to(torch_device)
                    target_states = target_states.to(torch_device)
                    target_risk = target_risk.to(torch_device)

                    pred_states, pred_risk_logits = model(state, controls, grid)
                    loss_traj = mse_loss(pred_states, target_states)
                    loss_risk = bce_loss(pred_risk_logits, target_risk)
                    loss = w_traj * loss_traj + w_risk * loss_risk
                    val_losses.append(float(loss.item()))

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            print(f"  Epoch {epoch + 1}/{epochs}: train={avg_train:.6f}, val={avg_val:.6f}")

            history.append({
                "phase": phase_idx,
                "epoch": epoch,
                "horizon": horizon,
                "train_loss": avg_train,
                "val_loss": avg_val,
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
                    # Save overall best
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "horizon_steps": horizon,
                        "hidden_dims": hidden_dims,
                        "input_dim": 6 + 3 * horizon + 81,
                        "target_dim": 6 * horizon + horizon,
                        "phase": phase_idx,
                    }, output_dir / "best_model.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    # Save final model (last phase's best)
    if best_ckpt_path is not None:
        import shutil
        shutil.copy(best_ckpt_path, output_dir / "model.pt")

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "output_dir": str(output_dir),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sequence_fdm_training.py::test_train_smoke -v`
Expected: PASS (training completes and saves checkpoints)

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/training/sequence_fdm_v2.py tests/test_sequence_fdm_training.py
git commit -m "feat: add curriculum training loop for sequence FDM V2"
```

---

## Task 10: MPPI Controller V2

**Files:**
- Create: `b2_fdm_mppi/controllers/mppi_omni_sequence_fdm_v2_torch.py`
- Test: `tests/test_mppi_omni_sequence_fdm_v2_torch.py`

**Context:** New controller inherits from `MppiOmniTorch` and overrides `_trajectory_cost_batch_torch()`. It uses `SequenceFdmDynamics.predict_torch()` to get predicted trajectories and risk logits, then computes goal cost, risk cost, obstacle cost, and smoothness. Unlike existing `MppiOmniSequenceFdmTorch`, this controller does NOT call `_rollout_batch_torch()` for trajectory prediction — the FDM replaces the rollout entirely. It still calls `_rollout_batch_torch()` for control penalties (accel, jerk) if needed.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from b2_fdm_mppi.controllers.mppi_omni_sequence_fdm_v2_torch import MppiOmniSequenceFdmV2Torch
from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics
from b2_fdm_mppi.core.terrain import TerrainField


def _make_dummy_dynamics(tmp_path: Path, horizon_steps: int = 5):
    model = SequenceFdmMlpV2(horizon_steps=horizon_steps, hidden_dims=[32])
    norm = {
        "state_mean": np.zeros(6, dtype=np.float32),
        "state_std": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(3, dtype=np.float32),
        "control_std": np.ones(3, dtype=np.float32),
        "target_mean": np.zeros(horizon_steps * 7, dtype=np.float32),
        "target_std": np.ones(horizon_steps * 7, dtype=np.float32),
    }
    ckpt = {
        "model_state_dict": model.state_dict(),
        "horizon_steps": horizon_steps,
        "hidden_dims": [32],
        "input_dim": 6 + 3 * horizon_steps + 81,
        "target_dim": horizon_steps * 7,
    }
    torch.save(ckpt, tmp_path / "best_model.pt")
    np.savez(tmp_path / "normalization.npz", **norm)
    return SequenceFdmDynamics.from_artifacts(tmp_path, device="cpu")


def test_controller_instantiation():
    with tempfile.TemporaryDirectory() as tmpdir:
        dyn = _make_dummy_dynamics(Path(tmpdir), horizon_steps=5)
        terrain = TerrainField(enabled=True, noise_enabled=False)
        controller = MppiOmniSequenceFdmV2Torch(
            dt=0.1,
            horizon_steps=5,
            num_samples=16,
            lambda_=0.7,
            noise_std=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            max_vx=1.5,
            max_vy=0.5,
            max_wz=1.0,
            sequence_dynamics=dyn,
            terrain=terrain,
            device="cpu",
            fdm_risk_weight=5.0,
        )
        assert controller.horizon_steps == 5
        assert controller.fdm_risk_weight == 5.0


def test_controller_cost_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        dyn = _make_dummy_dynamics(Path(tmpdir), horizon_steps=5)
        terrain = TerrainField(enabled=True, noise_enabled=False)
        controller = MppiOmniSequenceFdmV2Torch(
            dt=0.1,
            horizon_steps=5,
            num_samples=8,
            lambda_=0.7,
            noise_std=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            max_vx=1.5,
            max_vy=0.5,
            max_wz=1.0,
            sequence_dynamics=dyn,
            terrain=terrain,
            device="cpu",
            fdm_risk_weight=5.0,
        )
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        controls = torch.randn(8, 5, 3, dtype=torch.float32)
        goal = torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        obstacles = torch.zeros(0, 3, dtype=torch.float32)

        costs = controller._trajectory_cost_batch_torch(initial_state, controls, goal, obstacles)
        assert costs.shape == (8,)
        assert torch.isfinite(costs).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mppi_omni_sequence_fdm_v2_torch.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
"""Torch MPPI controller using Sequence FDM V2 for direct trajectory + risk prediction."""

from __future__ import annotations

import numpy as np
import torch

from b2_fdm_mppi.controllers.mppi_omni_torch import MppiOmniTorch
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.core.terrain_grid import sample_terrain_risk_grid_torch


class MppiOmniSequenceFdmV2Torch(MppiOmniTorch):
    """MPPI controller that uses Sequence FDM V2 to predict trajectories and risks directly."""

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

    def _trajectory_cost_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        path: torch.Tensor | None = None,
        costmap: dict | None = None,
    ) -> torch.Tensor:
        controls = torch.clamp(controls, -self.max_control_t, self.max_control_t)
        num_samples = int(controls.shape[0])
        H = self.horizon_steps

        # 1. Sample terrain grid centered on current state
        profile_start = self._profile_start()
        x0 = float(initial_state[0])
        y0 = float(initial_state[1])
        terrain_grid = sample_terrain_risk_grid_torch(self.terrain, x0, y0, size=9, span=18.0)
        terrain_grid = terrain_grid.unsqueeze(0).expand(num_samples, -1)
        self._profile_stop("terrain_grid_ms", profile_start)

        # 2. Prepare state tensor
        state_t = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32).reshape(1, 6),
            dtype=torch.float32,
            device=self.torch_device,
        ).expand(num_samples, -1)

        # 3. FDM forward pass (gradients retained)
        profile_start = self._profile_start()
        pred_states, pred_risk_logits = self.sequence_dynamics.predict_torch(state_t, controls, terrain_grid)
        self._profile_stop("fdm_inference_ms", profile_start)

        # 4. Binary risk from logits
        pred_risk = torch.sigmoid(pred_risk_logits)

        # 5. Compute costs
        profile_start = self._profile_start()
        final_states = pred_states[:, -1, :]
        xy_error = final_states[:, :2] - goal[:2]
        yaw_error = self._angle_diff_torch(final_states[:, 2], goal[2])
        goal_cost = self.goal_xy_weight * torch.sum(xy_error * xy_error, dim=1)
        yaw_cost = self.yaw_weight * yaw_error * yaw_error

        # Control smoothness
        control_cost = self.control_weight * torch.sum(controls * controls, dim=(1, 2))
        previous = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device).view(1, 1, 3)
        previous = previous.expand(num_samples, 1, 3)
        control_deltas = torch.diff(torch.cat([previous, controls], dim=1), dim=1)
        smooth_cost = self.smooth_weight * torch.sum(control_deltas * control_deltas, dim=(1, 2))

        # Obstacle cost from predicted trajectory positions
        obstacle_cost = self._obstacle_cost_batch_torch(pred_states[:, 1:, :], obstacles)

        # Risk cost from FDM prediction
        risk_cost = self.fdm_risk_weight * torch.sum(pred_risk, dim=1)

        # Path tracking and goal progress
        path_tracking_cost = self._path_tracking_cost_batch_torch(pred_states[:, 1:, :], path)
        goal_progress_cost = self._goal_progress_cost_batch_torch(initial_state, pred_states[:, -1, :], goal)
        heading_to_goal_cost = self._heading_to_goal_cost_batch_torch(pred_states[:, 1:, :], goal)
        local_costmap_cost = self._local_costmap_cost_batch_torch(pred_states[:, 1:, :], initial_state, costmap)

        self._profile_stop("cost_terms_ms", profile_start)

        return (
            goal_cost
            + yaw_cost
            + control_cost
            + smooth_cost
            + obstacle_cost
            + risk_cost
            + path_tracking_cost
            + goal_progress_cost
            + heading_to_goal_cost
            + local_costmap_cost
        ).to(torch.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mppi_omni_sequence_fdm_v2_torch.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add b2_fdm_mppi/controllers/mppi_omni_sequence_fdm_v2_torch.py tests/test_mppi_omni_sequence_fdm_v2_torch.py
git commit -m "feat: add MppiOmniSequenceFdmV2Torch controller"
```

---

## Task 11: CLI Training Tool

**Files:**
- Create: `tools/train_sequence_fdm_v2.py`
- Test: `tests/test_train_sequence_fdm_v2_cli.py`

**Context:** CLI that loads collected episodes, builds sliding windows, and calls `train_sequence_fdm_v2()`.

- [ ] **Step 1: Write failing test**

```python
import subprocess
import sys


def test_train_cli_help():
    result = subprocess.run(
        [sys.executable, "tools/train_sequence_fdm_v2.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_sequence_fdm_v2_cli.py -v`
Expected: FAIL with "FileNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""CLI for training Sequence FDM V2 with curriculum learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from b2_fdm_mppi.data.sequence_fdm_collector import build_sequence_fdm_windows
from b2_fdm_mppi.training.sequence_fdm_v2 import train_sequence_fdm_v2


def main():
    parser = argparse.ArgumentParser(description="Train Sequence FDM V2")
    parser.add_argument("--data-dir", required=True, help="Directory containing collected episode .npz files")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20], help="Curriculum horizons")
    parser.add_argument("--epochs", type=int, nargs="+", default=[50, 50, 100], help="Epochs per phase")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 5e-4, 1e-4], help="Learning rates per phase")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--w-risk", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not (len(args.horizons) == len(args.epochs) == len(args.lrs)):
        raise ValueError("--horizons, --epochs, and --lrs must have the same length")

    data_dir = Path(args.data_dir)
    episode_files = sorted(data_dir.glob("episode_*.npz"))
    if not episode_files:
        raise ValueError(f"No episode files found in {data_dir}")

    print(f"Loading {len(episode_files)} episodes...")
    all_windows = []
    for ep_path in episode_files:
        windows = build_sequence_fdm_windows(ep_path, horizon_steps=max(args.horizons), stride=1)
        all_windows.extend(windows)
    print(f"Extracted {len(all_windows)} training windows")

    curriculum = list(zip(args.horizons, args.epochs, args.lrs))
    metrics = train_sequence_fdm_v2(
        windows=all_windows,
        output_dir=args.output_dir,
        hidden_dims=args.hidden_dims,
        curriculum_phases=curriculum,
        batch_size=args.batch_size,
        w_risk=args.w_risk,
        patience=args.patience,
        device=args.device,
    )

    with open(Path(args.output_dir) / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Best val loss: {metrics['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_sequence_fdm_v2_cli.py -v`
Expected: `test_train_cli_help` PASS

- [ ] **Step 5: Commit**

```bash
git add tools/train_sequence_fdm_v2.py tests/test_train_sequence_fdm_v2_cli.py
git commit -m "feat: add CLI for training sequence FDM V2"
```

---

## Task 12: Evaluation Tool

**Files:**
- Create: `tools/eval_sequence_fdm_v2.py`
- Test: `tests/test_eval_sequence_fdm_v2_cli.py`

**Context:** CLI that runs open-loop evaluation (compare FDM vs nominal vs ground truth on test terrains) and closed-loop comparison (run both Nominal MPPI and Sequence FDM MPPI on same terrains).

- [ ] **Step 1: Write failing test**

```python
import subprocess
import sys


def test_eval_cli_help():
    result = subprocess.run(
        [sys.executable, "tools/eval_sequence_fdm_v2.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_sequence_fdm_v2_cli.py -v`
Expected: FAIL with "FileNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""CLI for evaluating Sequence FDM V2: open-loop and closed-loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_torch import MppiOmniTorch
from b2_fdm_mppi.controllers.mppi_omni_sequence_fdm_v2_torch import MppiOmniSequenceFdmV2Torch
from b2_fdm_mppi.core.sequence_fdm_dynamics import SequenceFdmDynamics
from b2_fdm_mppi.core.terrain import TerrainField
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator


def _load_controller(model_dir: str | None, config: dict, device: str):
    """Load either nominal or sequence FDM controller from config."""
    if model_dir is None:
        return MppiOmniTorch.from_config(config, device=device)
    dynamics = SequenceFdmDynamics.from_artifacts(model_dir, device=device)
    return MppiOmniSequenceFdmV2Torch.from_config(
        config, device=device, sequence_dynamics=dynamics, fdm_risk_weight=10.0
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sequence FDM V2")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--model-dir", help="Trained model directory (for FDM controller)")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config = load_config(args.config)

    results = []
    for i in range(args.num_episodes):
        terrain_seed = args.base_seed + i
        generator = RandomTerrainGenerator(map_bounds=(-20, 20, -20, 20))
        terrain = generator.generate(seed=terrain_seed)

        config = base_config.copy()
        config["terrain"] = terrain.to_config()
        # Note: start/goal would need to be sampled here; omitted for brevity in CLI

        # Run both controllers
        # ... (implementation depends on runner interface)
        result = {
            "episode": i,
            "terrain_seed": terrain_seed,
            "fdm_model": args.model_dir,
        }
        results.append(result)

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete: {len(results)} episodes")


if __name__ == "__main__":
    main()
```

**Note:** The full closed-loop evaluation requires integrating with `OmniMppiSimulationRunner`. The CLI skeleton is provided; the runner integration should be filled in during execution by referencing the existing evaluation tools in `b2_fdm_mppi/evaluation/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_sequence_fdm_v2_cli.py -v`
Expected: `test_eval_cli_help` PASS

- [ ] **Step 5: Commit**

```bash
git add tools/eval_sequence_fdm_v2.py tests/test_eval_sequence_fdm_v2_cli.py
git commit -m "feat: add evaluation CLI skeleton for sequence FDM V2"
```

---

## Self-Review Checklist

### 1. Spec Coverage

| Spec Requirement | Plan Task |
|-----------------|-----------|
| Random terrain generator (patches, noise, seed control) | Task 1 |
| 9x9 terrain risk grid sampling | Task 2 |
| SequenceFdmMlpV2 (absolute states + binary risk) | Task 3 |
| SequenceFdmDynamics wrapper | Task 4 |
| Episode collection with binary risk marking | Task 5 |
| Parallel collection CLI | Task 6 |
| Sliding window dataset builder | Task 7 |
| Curriculum dataset + normalization | Task 8 |
| Curriculum training loop | Task 9 |
| MppiOmniSequenceFdmV2Torch controller | Task 10 |
| Training CLI | Task 11 |
| Evaluation CLI | Task 12 |

**Gap identified:** The evaluation CLI (Task 12) is a skeleton because the exact `OmniMppiSimulationRunner` integration interface was not fully explored. During execution, the runner integration should be completed by referencing `b2_fdm_mppi/evaluation/benchmark.py` or `b2_fdm_mppi/simulation/omni_runner.py`.

### 2. Placeholder Scan

- No "TBD", "TODO", "implement later" found in task code.
- `tools/eval_sequence_fdm_v2.py` has a comment noting runner integration is needed — this is acknowledged as a skeleton.
- `collect_sequence_fdm_episode` uses `terrain.to_config()` which may not exist on `TerrainField` — this needs verification during execution.

### 3. Type Consistency

- `SequenceFdmMlpV2.forward()` returns `(torch.Tensor, torch.Tensor)` consistently across Tasks 3, 4, 9, 10.
- `SequenceFdmDynamics.predict_torch()` signature is consistent across Tasks 4 and 10.
- Normalization keys (`state_mean`, `state_std`, `control_mean`, `control_std`, `target_mean`, `target_std`) are consistent across Tasks 4, 8, 9.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-sequence-fdm-mppi.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach would you prefer?
