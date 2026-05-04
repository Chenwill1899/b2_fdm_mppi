# Sequence FDM-MPPI Design Spec

**Date**: 2026-05-04
**Status**: Approved
**Approach**: End-to-end sequence FDM (Approach A) with curriculum learning

---

## 1. Problem Statement

Current learned FDM is a single-step residual compensator: it predicts `Δu_t` at each step, applies it to the control input, then feeds the corrected control to the nominal kinematic model for integration. MPPI still fundamentally relies on the nominal rollout, only with slightly corrected controls.

**Goal**: Build a sequence-level forward dynamics model that directly evaluates entire candidate control sequences. Given current state, future H-step controls, and terrain features, the model outputs:
- Future trajectory `X̂_{t+1:t+H}`
- Binary risk sequence `R̂_{t+1:t+H}`

MPPI then uses these predictions directly to compute candidate costs, replacing the nominal model rollout.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Prediction horizon H | Equal to MPPI `horizon_steps` | Direct replacement for rollout |
| State prediction | Absolute coordinates | No nominal model ambiguity; supervise against real trajectories |
| Risk representation | Binary (0/1), triggered on threshold exceedance | RL-style trial-and-error signal; once triggered, all subsequent timesteps are 1 |
| Risk trigger | Enter terrain high-risk area (`risk_cost > threshold`) | Core scenario: MPPI avoids obstacles but may plan into dangerous terrain |
| Terrain encoding | 9x9 risk grid, 18m x 18m (2m resolution), centered on robot | Spatial awareness of nearby risk zones; compact enough for MLP |
| Model type | Pure MLP + curriculum learning (Approach A) | Minimal baseline; fast single-pass inference; upgrade path to temporal encoder (Approach C) |
| Gradient mode | Training + inference both retain gradients | Enables future end-to-end MPPI parameter tuning |
| Obstacle cost | Independent computation, NOT from FDM | Explicit safety guarantee; FDM focuses on terrain dynamics |

---

## 3. Random Terrain Generation & Data Collection

### 3.1 RandomTerrainGenerator

Wrapper around existing `TerrainField` that produces varied terrains per episode:

```python
class RandomTerrainGenerator:
    def __init__(self, base_config: dict, num_patches_range=(2, 5),
                 patch_types=("ellipse", "band"), map_bounds=(-10, 10, -10, 10)):
        ...

    def generate(self, seed: int) -> TerrainField:
        # 1. Random base noise parameters (seed, scale, weights)
        # 2. Random N patches (position, size, angle, risk intensity)
        # 3. Patches distributed within map_bounds, avoiding start/goal safe zones
        # 4. Return configured TerrainField
```

**Constraints**:
- Start and goal positions must be > 10m apart
- Obstacle density high enough that every trajectory must avoid at least 1 obstacle
- All randomness controlled by `seed` for reproducibility

### 3.2 Episode Data Collection

For each random terrain:
1. Randomize start and goal positions (> 10m apart, within map bounds)
2. Run MPPI with nominal `MppiOmniTorch` closed-loop until goal reached or step limit
3. Record:
   - `states`: T x 6 (x, y, theta, vx, vy, wz)
   - `controls`: T x 3 (vx_cmd, vy_cmd, wz_cmd)
   - `terrain_features`: T x 4 (slope_f, slope_l, roughness, friction)
   - `terrain_risk`: T x 1 (from `TerrainField.risk_cost()`)
4. Mark binary risk sequence:
   - If `terrain_risk[t] > risk_threshold` (e.g., 0.6): `binary_risk[t:] = 1`
   - Else: `binary_risk = 0`
5. Mark success: `final_dist < goal_tol` AND `max(binary_risk) == 0`

**Trajectory retention**: Both success and failure trajectories are kept. Failures are critical for the model to learn "which control sequences lead to risk."

### 3.3 Training Dataset Construction

Sliding window extraction from episodes:

**Input**:
- Current state `s_t`: 6D
- Future H-step controls `U_{t:t+H-1}`: H x 3, flattened to 3H
- Local terrain encoding `E_t`: 9 x 9 = 81D (18m x 18m risk grid, 2m resolution)

**Target**:
- Future H-step states `X_{t+1:t+H}`: H x 6 (absolute coordinates)
- Future H-step binary risk `R_{t+1:t+H}`: H x 1

**Data split**: By terrain seed (not by window), ensuring validation terrains are unseen during training.

---

## 4. SequenceFdmMlp Architecture

```python
class SequenceFdmMlp(nn.Module):
    def __init__(self, horizon_steps: int,
                 terrain_grid_size: int = 81,
                 hidden_dims: list[int] = [256, 256, 256]):
        input_dim = 6 + 3 * horizon_steps + terrain_grid_size   # state + controls + terrain
        output_dim = 6 * horizon_steps + 1 * horizon_steps       # trajectories + risks

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(h, h), nn.ReLU()) for h in hidden_dims],
            nn.Linear(hidden_dims[-1], output_dim)
        )
```

**Output splitting**:
- `X_pred` (first `6*H` elements): reshaped to `(H, 6)`, no activation (regression)
- `R_logits` (last `H` elements): sigmoid for probability, threshold at 0.5 for binary

**Design note**: No separate risk head in the baseline. A shared encoder + single output layer is simplest for Approach A. If risk prediction underperforms, a separate risk head can be added as an upgrade.

---

## 5. Training Protocol

### 5.1 Loss Function

Multi-task loss with per-step supervision:

```python
loss = w_traj * L_traj + w_risk * L_risk

L_traj = (1/H) * sum(MSE(x_pred[k], x_target[k]) for k in 1..H)
L_risk = (1/H) * sum(BCEWithLogitsLoss(r_logits[k], r_target[k]) for k in 1..H)
```

**Initial weights**: `w_traj = 1.0`, `w_risk = 0.5`. Increase `w_risk` to 1.0 if F1 remains low.

**Critical**: `L_traj` computes MSE independently at each of the H steps, not just at the final step. This forces the network to learn the correct intermediate dynamics composition.

### 5.2 Curriculum Learning

| Phase | Horizon H | Epochs | Learning Rate | Description |
|-------|-----------|--------|---------------|-------------|
| 1 | 5 | 50 | 1e-3 | Warm-up: learn short-horizon mapping |
| 2 | 10 | 50 | 5e-4 | Extend mid-horizon composition |
| 3 | 20 | 100 | 1e-4 | Target horizon, fine-tuning |
| 4 | 20 | 50 | 5e-5 | Decay convergence |

Each phase initializes from the previous phase's best validation checkpoint. If validation loss plateaus for 10 epochs, advance to the next phase early.

### 5.3 Normalization

Standardization based on training set statistics:
- States: `state_mean`, `state_std`
- Controls: `control_mean`, `control_std`
- Terrain grid: approximately [0, 1], no additional normalization

Normalization parameters are saved alongside the model checkpoint and used during inference.

### 5.4 Validation Metrics

| Metric | Description |
|--------|-------------|
| Val Traj MSE | Predicted vs ground-truth trajectory MSE |
| Val Risk Accuracy | Binary risk classification accuracy |
| Val Risk F1 | Handles class imbalance (more safe trajectories than risky) |
| Endpoint Error | Position error at step H only |

**Early stopping**: patience=10, monitoring `val_traj_mse`.

---

## 6. MPPI Integration

### 6.1 Class Design

```python
class MppiOmniSequenceFdmTorch(MppiOmniTorch):
    def __init__(self, *, sequence_fdm: SequenceFdmMlp,
                 norm_params: dict, terrain_field: TerrainField,
                 fdm_risk_weight: float = 10.0, ...):
        super().__init__(...)
        self.sequence_fdm = sequence_fdm  # gradients retained
        self.norm = norm_params
        self.terrain = terrain_field
        self.fdm_risk_weight = fdm_risk_weight
```

**Reused unchanged**:
- `_sample_control_sequences()` — candidate sampling
- `_update_control_sequence()` — softmin weighting + control update
- `_control_cost()` / `_obstacle_cost_batch()` — control smoothness + obstacle avoidance

**Overridden**:
- `_trajectory_cost_batch_torch()` — FDM replaces nominal rollout

### 6.2 Cost Computation Flow

```python
def _trajectory_cost_batch_torch(self, candidates, state, ...):
    # candidates: (num_samples, H, 3)
    # state: (6,)

    # 1. Sample 18m x 18m terrain risk grid centered on robot
    terrain_grid = self._sample_terrain_grid(state[0], state[1])  # (81,)

    # 2. Normalize inputs
    s_norm = (state - self.norm.state_mean) / self.norm.state_std
    U_norm = (candidates - self.norm.control_mean) / self.norm.control_std

    # 3. Batch expand for FDM
    s_batch = s_norm.unsqueeze(0).expand(N, -1)      # (N, 6)
    g_batch = terrain_grid.unsqueeze(0).expand(N, -1)  # (N, 81)

    # 4. FDM forward (gradients retained)
    x_pred, r_logits = self.sequence_fdm(s_batch, U_norm, g_batch)
    x_pred = x_pred.view(N, H, 6)
    x_pred = x_pred * self.norm.state_std + self.norm.state_mean
    r_pred = (torch.sigmoid(r_logits) > 0.5).float()  # (N, H)

    # 5. Compute costs from predictions
    goal_cost = self._goal_cost_batch(x_pred)
    risk_cost = r_pred.sum(dim=1) * self.fdm_risk_weight
    obstacle_cost = self._obstacle_cost_from_trajectories(x_pred[..., :2])
    smooth_cost = self._control_smoothness(candidates)

    return goal_cost + risk_cost + obstacle_cost + smooth_cost
```

### 6.3 Terrain Grid Sampling

```python
def _sample_terrain_grid(self, x: float, y: float) -> torch.Tensor:
    xs = torch.linspace(x - 9, x + 9, 9)
    ys = torch.linspace(y - 9, y + 9, 9)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    risks = self.terrain.risk_cost_grid(grid_x.numpy(), grid_y.numpy())
    return torch.from_numpy(risks).float().flatten()
```

### 6.4 Cost Composition Summary

| Cost Term | Source | Role |
|-----------|--------|------|
| Goal | FDM predicted trajectory | Target convergence |
| Risk | FDM predicted binary risk | **Core innovation**: terrain risk predicted by FDM |
| Obstacle | Predicted trajectory positions | Explicit safety (independent of FDM) |
| Smooth | Control sequence itself | Control quality |

---

## 7. Evaluation Protocol

### 7.1 Open-Loop FDM Evaluation

On a fixed test terrain set (unseen during training), compare three trajectories from the same initial state and control sequence:

| Source | Method |
|--------|--------|
| FDM prediction | `sequence_fdm(state, controls, terrain_grid)` |
| Nominal model | Existing `MppiOmniTorch` rollout logic |
| Ground truth | Oracle residual world closed-loop rollout |

**Metrics**:
- `ADE`: Average displacement error (per-step position Euclidean distance)
- `FDE`: Final displacement error (step H only)
- `Risk Accuracy`: FDM risk vs ground-truth risk match rate
- `Risk F1`: Handles class imbalance

### 7.2 Risk Prediction Evaluation

Dedicated test for risk awareness:
1. Construct "must-cross high-risk zone" scenarios where the only path between start and goal intersects high-risk terrain
2. FDM should predict risk=1 for control sequences passing through the zone
3. Compare against nominal model rollout which lacks real dynamics awareness

**Metrics**: Precision, Recall, F1 (focus on minimizing False Negatives).

### 7.3 Closed-Loop MPPI Comparison

On identical random terrain sets, run:

| Controller | Description |
|------------|-------------|
| Nominal MPPI | Existing `MppiOmniTorch` with nominal rollout |
| Sequence FDM MPPI | `MppiOmniSequenceFdmTorch` |

**Comparison dimensions**:
- Success rate (reach goal without triggering risk)
- Final distance error
- Average path length
- Risk exposure ratio (time in high-risk zones)
- Minimum obstacle clearance
- Per-iteration MPPI computation time

### 7.4 Minimum Viable Product Criteria

Sequence FDM MPPI is considered effective when:

1. FDM open-loop ADE < 0.5m (at H=20)
2. FDM risk F1 > 0.7
3. Closed-loop success rate >= Nominal MPPI on same terrain set
4. Closed-loop risk exposure ratio < Nominal MPPI
5. Per-iteration MPPI time overhead < 50% vs Nominal MPPI

---

## 8. Upgrade Path

After Approach A baseline is validated:

1. **Temporal encoder (Approach C)**: Replace MLP encoder with 1D-CNN or small GRU over the control sequence to better model temporal structure
2. **Separate risk head**: If risk F1 remains low, add a dedicated risk prediction head with auxiliary supervision
3. **End-to-end MPPI tuning**: Use retained gradients to backpropagate from closed-loop performance into MPPI hyperparameters or FDM weights

---

## 9. Open Questions

1. **Risk threshold**: The 0.6 threshold for binary risk marking is a first guess. It may need tuning based on the distribution of `terrain_risk` values across random terrains.
2. **Patch density**: "At least 1 obstacle per trajectory" needs quantification (e.g., minimum 3 patches per map) to ensure the data collection pipeline enforces this.
3. **FDM vs nominal on low-risk terrain**: If terrain is mostly safe, Sequence FDM may not show clear advantage. Evaluation must include sufficiently challenging terrain distributions.
